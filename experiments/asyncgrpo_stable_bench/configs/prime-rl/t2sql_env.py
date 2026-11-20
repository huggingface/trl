"""miles' text-to-SQL episode and reward as a verifiers v1 taskset + harness for prime-rl.

The episode logic is TRL's port (workloads/text2sql/t2sql_episode.py, itself miles' sql_episode.py on
River's `text_to_sql_recipe` modules at river-recipes 64394037). The reward and SQL helpers below are copied from it
verbatim; that file imports TRL, which prime-rl's venv does not have, so it can't be imported here.

What is the same as TRL: prompt (`prompts.opening_messages(task, few_shot=True)` + the `execute_sql_query` schema,
thinking on), up to 5 tool turns + 1 answer turn, 3,072 tokens per turn, 29,696 per episode, first SQL call executed
read-only through the recipe's `SqlWork(16)` (50 rows, 1 MB, 30 s), malformed call -> `MALFORMED_ACTION_PROMPT`, unsafe
query ends the rollout, observations as user messages with the turns-left line and `SQL_LIMIT_PROMPT` after the 5th
tool turn, reward `reward.rollout_reward` with the `qwen3.6-35b-a3b` preset minus VeriEQL and the re-draft penalty.

What differs (plumbing): the model is called through verifiers' interception server (chat completions), so prime-rl's
renderer turns messages into tokens and parses each completion into `reasoning_content` / `content` / `tool_calls`.
The recipe reads structured `tool_calls` the same way as the `<tool_call>` text TRL parses (`parsing.parse_tool_calls`,
`reward.observe_tool_calls`); the process rules get the turn's reasoning + `</think>` + content, as TRL passes the raw
turn text. The per-turn cap is the run's `max_completion_tokens` (3,072): the train client imposes the config's
sampling on every request, so the episode budget can't shrink a request the way TRL's `min(3072, 29696 - len)` does.
Instead a turn sampled when fewer than 3,072 tokens were left that runs past the budget scores as cut by the context
(TRL's `context_cut`), and the observation check before each turn is TRL's, with the observation ids TRL appends.
"""

import asyncio
import dataclasses
import functools
import json
import os

import httpx
import verifiers.v1 as vf
from text_to_sql_recipe import config, grading, parsing, prompts, reward, sqlite_exec
from text_to_sql_recipe.config import RewardConfig
from transformers import AutoTokenizer
from verifiers.v1.runtimes import ProgramResult


# miles' launcher settings (run_qwen3_6_35b_a3b_text_to_sql.py): the recipe's qwen3.6-35b-a3b preset
MAX_TOOL_TURNS = 5  # --text-to-sql-max-tool-turns
MAX_TOKENS_PER_TURN = 3072  # --rollout-max-response-len
MAX_SEQ_LEN = 29696  # --max-seq-len
RECIPE = "qwen3.6-35b-a3b"  # --text-to-sql-recipe
MODEL = os.environ.get("T2SQL_TOKENIZER", "Qwen/Qwen3.5-35B-A3B")

# SQLite runs in worker threads; this bounds tool calls, grading and gold queries together, and lets a grading
# retry drain the other work, as the recipe's grading runtime does
_SQL_WORK = None
# gold SQL is slow on the largest databases, so each task's result is computed once per process
_GOLD_RESULTS: dict[str, asyncio.Task] = {}


def _sql_work():
    # created lazily: an asyncio.Semaphore binds to the event loop it is first used on
    global _SQL_WORK
    if _SQL_WORK is None:
        _SQL_WORK = sqlite_exec.SqlWork(16)
    return _SQL_WORK


@functools.cache
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


# --- verifiers side -------------------------------------------------------------------------------------------------


class T2SQLData(vf.TaskData):
    task: str
    """The recipe's task row as JSON (convert.py's `task` column, database_path absolute)."""


class T2SQLTask(vf.Task[T2SQLData]):
    @vf.reward
    async def t2sql_reward(self, trace: vf.Trace) -> float:
        scores = await reward_func(json.loads(self.data.task), trace.info["t2sql"], evaluation=False)
        trace.info["t2sql_scores"] = scores
        return scores["score"]

    @vf.metric
    async def t2sql(self, trace: vf.Trace) -> dict[str, float]:
        episode = trace.info["t2sql"]
        return {"tool_calls": float(episode["tool_calls"]), "unsafe": float(episode["unsafe"]),
                "context_cut": float(episode["status"] != "completed")}


class T2SQLConfig(vf.TasksetConfig):
    data: str = "/fsx/amine_dirhoussi/stable-bench-src/text2sql/data/train.jsonl"
    """convert.py output, one row per task."""


class T2SQLTaskset(vf.Taskset[T2SQLTask, T2SQLConfig]):
    def load(self):
        for line in open(self.config.data):
            row = json.loads(line)
            yield T2SQLTask(T2SQLData(id=row["id"], prompt=row["prompt"], task=row["task"]), self.config.task)


class T2SQLHarnessConfig(vf.HarnessConfig):
    pass


class T2SQLHarness(vf.Harness[T2SQLHarnessConfig]):
    """Runs the episode loop in the env worker itself; every model call goes through the interception endpoint."""

    APPENDS_SYSTEM_PROMPT = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data, tool_interception_url=None):
        async with httpx.AsyncClient(timeout=None) as client:

            async def chat(messages):
                r = await client.post(
                    f"{endpoint.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {secret}"},
                    json={"model": ctx.model, "messages": messages, "tools": [prompts.sql_tool_schema({})]},
                )
                r.raise_for_status()
                out = r.json()
                return out["choices"][0], out["usage"]

            trace.info["t2sql"] = await run_episode(chat, json.loads(data.task))
        return ProgramResult(exit_code=0, stdout="", stderr="")


__all__ = ["T2SQLTaskset", "T2SQLHarness"]


# --- episode (t2sql_episode.run_episode over chat completions) ------------------------------------------------------


async def run_episode(chat, task: dict) -> dict:
    """One episode (miles' `generate`). `chat(messages)` returns one completion choice and its usage."""
    messages = prompts.opening_messages(task, few_shot=True)
    n_tokens = 0
    episode = {"turns": [], "final_text": "", "unsafe": False, "tool_calls": 0, "tool_protocol_events": [],
               "status": "completed", "finish_reasons": []}

    for turn in range(1, MAX_TOOL_TURNS + 2):
        choice, usage = await chat(messages)
        budget = MAX_SEQ_LEN - usage["prompt_tokens"]
        n_tokens = usage["prompt_tokens"] + usage["completion_tokens"]
        message, finish_reason = choice["message"], choice["finish_reason"]
        # TRL asks for min(3072, budget) tokens; a turn that needed more than the budget is cut there
        if budget < MAX_TOKENS_PER_TURN and usage["completion_tokens"] >= budget:
            finish_reason = "length"
        episode["finish_reasons"].append(finish_reason)
        if finish_reason == "abort":
            break

        visible = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            visible["tool_calls"] = message["tool_calls"]
        reasoning = message.get("reasoning_content")
        episode["turns"].append((f"{reasoning}</think>" if reasoning is not None else "") + visible["content"])
        tools_allowed = turn <= MAX_TOOL_TURNS
        reward.observe_tool_calls(episode["tool_protocol_events"], visible, turn=turn, tools_allowed=tools_allowed)
        if finish_reason == "length" and budget < MAX_TOKENS_PER_TURN:
            # the context budget cut this turn; the recipe never grades such a turn, so it scores as no answer
            episode["status"] = "context_cut"
            break
        if not tools_allowed:
            episode["final_text"] = visible["content"].strip()
            break

        call = next((c for c in parsing.parse_tool_calls(visible) if c.name == prompts.SQL_TOOL_NAME), None)
        if call is not None:
            episode["tool_calls"] += 1
            observation = await _sql_observation(task, call.arguments, turns_left=MAX_TOOL_TURNS - turn)
            if observation is None:
                episode["unsafe"] = True
                break
        elif parsing.has_tool_call_intent(visible):
            observation = parsing.MALFORMED_ACTION_PROMPT
        else:
            episode["final_text"] = visible["content"].strip()
            break

        new = [{"role": "user", "content": observation}]
        if turn == MAX_TOOL_TURNS:
            new.append({"role": "user", "content": prompts.SQL_LIMIT_PROMPT})
        last_token_id = tokenizer().eos_token_id if finish_reason == "stop" else None
        if n_tokens + len(_observation_ids(tokenizer(), new, last_token_id=last_token_id)) >= MAX_SEQ_LEN:
            episode["status"] = "truncated"
            break
        messages = [*messages, message, *new]

    episode["total_tokens"] = n_tokens
    return episode


# --- copied verbatim from t2sql_episode.py --------------------------------------------------------------------------


async def reward_func(task: dict, episode: dict, *, evaluation: bool) -> dict:
    if episode["unsafe"]:
        grade = {"grade": "INCORRECT", "error_category": "unsafe_sql", "format_compliant": False}
    else:
        graded_task, expected = await _gold_result(task)
        grade = await _grade(episode["final_text"], graded_task, expected)

    scored = reward.rollout_reward(
        grade,
        _reward_config(RECIPE, evaluation=evaluation),
        task=task,
        messages=[{"role": "assistant", "content": text} for text in episode["turns"]],
        prompt_message_count=0,
        unsafe=episode["unsafe"],
        refuted=False,
        redrafts=0.0,
        loop_stopped=False,
        tool_protocol_events=episode["tool_protocol_events"],
    )
    return {
        "score": scored.reward,
        "correct": float(grade["grade"] == "CORRECT"),
        "format_compliant": float(bool(grade.get("format_compliant"))),
        "tool_calls": float(episode["tool_calls"]),
        "process_penalty": scored.metrics["process_penalty"],
        "invalid_tool_calls": scored.metrics["tool_protocol/invalid_calls"],
        "error_category": grade.get("error_category"),
    }


@functools.cache
def _reward_config(recipe: str, *, evaluation: bool) -> RewardConfig:
    """The recipe preset's reward terms, without the VeriEQL verifier and re-draft guard this example does not port."""
    train = dataclasses.replace(config.load(recipe).reward, verifier="none", redraft_penalty=0.0)
    if not evaluation:
        return train
    # evaluation scores the outcome only, as the recipe's evaluation does
    return dataclasses.replace(train, process_penalty_weight=0.0, tool_format_penalty=0.0)


def _observation_ids(tokenizer, messages: list[dict], *, last_token_id: int) -> list[int]:
    """The ids that close the sampled turn and open the next one, as the recipe's incremental framing does."""
    stop = "" if last_token_id == tokenizer.eos_token_id else tokenizer.eos_token
    framing = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(stop + "\n" + framing, add_special_tokens=False)


async def _grade(final_text: str, task: dict, expected) -> dict:
    """Grade the answer, retrying a timeout with the other SQL work drained, as the recipe's `GradingRuntime` does."""
    for attempt in range(grading.GRADING_TIMEOUT_RETRIES + 1):
        last_attempt = attempt == grading.GRADING_TIMEOUT_RETRIES
        try:
            grade = await _sql_work().execute(
                grading.grade_with_expected, final_text, task, expected, exclusive=attempt > 0
            )
        except sqlite_exec.SQLiteInfrastructureError as error:
            if last_attempt or not grading.numeric_comparison_timed_out(error):
                raise
            continue
        if last_attempt or grade.get("error_category") != "timeout":
            return grade


def _run_sql_tool(task: dict, arguments: dict):
    # building the executor hashes and may snapshot the whole database, so it must stay off the event loop
    return sqlite_exec.execute_sql_tool(task, sqlite_exec.tool_executor(task), arguments)


async def _sql_observation(task: dict, arguments: dict, *, turns_left: int) -> str | None:
    """The user message reporting one SQL call, or None when the query was unsafe."""
    try:
        results = await _sql_work().execute(_run_sql_tool, task, arguments)
    except sqlite_exec.SQLiteInfrastructureError:
        raise
    except sqlite_exec.SQL_TOOL_TERMINAL_ERRORS:
        return None
    except Exception as exc:  # as in the recipe's SqlEnv: any other error is shown to the model
        error = (
            str(exc).encode("utf-8", "ignore")[: prompts.SQL_TOOL_CONTEXT_ERROR_MAX_BYTES].decode("utf-8", "ignore")
        )
        return (
            f"Results for {prompts.SQL_TOOL_NAME}:\nError: {error} {prompts.SQL_TOOL_CONTEXT_ERROR_GUIDANCE}"
            f"\n\n{prompts.tool_result_prompt()}"
        )
    body = "\n\n".join(result.format() for result in results) or "(no output)"
    return (
        f"Results for {prompts.SQL_TOOL_NAME}:\n{body}\n\n{prompts.tool_result_prompt()}"
        f"\n\nYou have {turns_left} turns left to complete the task."
    )


async def _gold_result(task: dict):
    """The task as graded and its gold result."""
    task_id = str(task["id"])
    if task_id not in _GOLD_RESULTS:
        pending = asyncio.create_task(_sql_work().execute(_execute_gold, dict(task)))
        _GOLD_RESULTS[task_id] = pending
        pending.add_done_callback(functools.partial(_forget_failed_gold, task_id))
    return await asyncio.shield(_GOLD_RESULTS[task_id])


def _forget_failed_gold(task_id: str, pending: asyncio.Task) -> None:
    """Drop a failed gold result so the next sample of the task runs the gold query again."""
    if (pending.cancelled() or pending.exception() is not None) and _GOLD_RESULTS.get(task_id) is pending:
        del _GOLD_RESULTS[task_id]


def _execute_gold(task: dict):
    grading.prepare_streaming_grading_storage([task])
    failures = {}
    results = grading.gold_results([task], concurrency=1, failures=failures)
    if not results:
        raise RuntimeError(f"gold SQL failed for task {task['id']}: {failures}")
    return task, results[str(task["id"])]
