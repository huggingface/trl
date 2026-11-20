"""miles' text-to-SQL episode and reward (examples/text_to_sql/sql_episode.py, miles 778794a8), ported to TRL.

The prompt, the read-only SQLite tool, the grader and the reward terms come from River's `text_to_sql_recipe` package
(river-recipes 64394037, the commit miles pins), imported exactly as miles imports it. Each turn either calls
`execute_sql_query` once or answers; after the last tool turn the next turn is graded as the answer whatever it
contains. A query that tries to write or escape the sandbox ends the rollout as unsafe.

Observations are appended to the sampled ids as the chat template frames a new user message, never re-rendered, so
the trainer sees exactly the tokens the engine sampled (one training row per episode).

What changed from miles, and only that: SGLang `/generate` -> vLLM `/v1/completions` (`sample` below), miles'
`Sample` -> plain lists, and the episode is one function shared by the TRL rollout loop (`Text2SQLLoop`) and the
vLLM-only probe/eval driver (`probe_eval.py`).
"""

import asyncio
import dataclasses
import functools
import json
import os
import random
import time

import numpy as np
from text_to_sql_recipe import config, grading, parsing, prompts, reward, sqlite_exec
from text_to_sql_recipe.config import RewardConfig

from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutLoop, Rollout, TurnRecord


# miles' launcher settings (run_qwen3_6_35b_a3b_text_to_sql.py): the recipe's qwen3.6-35b-a3b preset
MAX_TOOL_TURNS = 5  # --text-to-sql-max-tool-turns
MAX_TOKENS_PER_TURN = 3072  # --rollout-max-response-len
MAX_SEQ_LEN = 29696  # --max-seq-len
RECIPE = "qwen3.6-35b-a3b"  # --text-to-sql-recipe

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


def opening_prompt_ids(tokenizer, task: dict) -> list[int]:
    """miles' prompt: convert_dataset.py's opening messages and tool schema, rendered by --apply-chat-template."""
    messages = prompts.opening_messages(task, few_shot=True)
    tools = [prompts.sql_tool_schema({})]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    return tokenizer.encode(text, add_special_tokens=False)


async def run_episode(sample, tokenizer, task: dict, *, evaluation: bool) -> dict:
    """One episode (miles' `generate` + `reward_func`).

    `sample(prompt_ids, max_tokens)` returns `(output_ids, logprobs, text, finish_reason)` for one sampling request.
    Returns the turns as TRL `TurnRecord`s, the reward and the episode record.
    """
    tokens = opening_prompt_ids(tokenizer, task)
    prompt_len = len(tokens)
    turns = []
    episode = {"turns": [], "final_text": "", "unsafe": False, "tool_calls": 0, "tool_protocol_events": []}
    finish_reasons = []
    status = "completed"

    for turn in range(1, MAX_TOOL_TURNS + 2):
        max_new_tokens = min(MAX_TOKENS_PER_TURN, MAX_SEQ_LEN - len(tokens))
        if max_new_tokens <= 0:
            status = "truncated"
            break
        output_ids, logprobs, text, finish_reason = await sample(tokens, max_new_tokens)
        turns.append(TurnRecord(list(tokens), output_ids, logprobs))
        tokens = tokens + output_ids
        finish_reasons.append(finish_reason)
        if finish_reason == "abort":
            break

        episode["turns"].append(text)
        message = {"role": "assistant", "content": _visible_text(text)}
        tools_allowed = turn <= MAX_TOOL_TURNS
        reward.observe_tool_calls(episode["tool_protocol_events"], message, turn=turn, tools_allowed=tools_allowed)
        if finish_reason == "length" and max_new_tokens < MAX_TOKENS_PER_TURN:
            # the context budget cut this turn; the recipe never grades such a turn, so it scores as no answer
            status = "context_cut"
            break
        if not tools_allowed:
            episode["final_text"] = message["content"].strip()
            break

        call = next((c for c in parsing.parse_tool_calls(message) if c.name == prompts.SQL_TOOL_NAME), None)
        if call is not None:
            episode["tool_calls"] += 1
            observation = await _sql_observation(task, call.arguments, turns_left=MAX_TOOL_TURNS - turn)
            if observation is None:
                episode["unsafe"] = True
                break
        elif parsing.has_tool_call_intent(message):
            observation = parsing.MALFORMED_ACTION_PROMPT
        else:
            episode["final_text"] = message["content"].strip()
            break

        messages = [{"role": "user", "content": observation}]
        if turn == MAX_TOOL_TURNS:
            messages.append({"role": "user", "content": prompts.SQL_LIMIT_PROMPT})
        observation_ids = _observation_ids(tokenizer, messages, last_token_id=tokens[-1])
        if len(tokens) + len(observation_ids) >= MAX_SEQ_LEN:
            status = "truncated"
            break
        tokens = tokens + observation_ids

    scores = await reward_func(task, episode, evaluation=evaluation)
    record = {
        "id": str(task["id"]),
        "evaluation": evaluation,
        "status": status,
        "turns": len(turns),
        "finish_reasons": finish_reasons,
        "prompt_tokens": prompt_len,
        "total_tokens": len(tokens),
        "sampled_tokens": sum(len(t.output_ids) for t in turns),
        **scores,
    }
    return {"turns": turns, "reward": scores["score"], "record": record, "completion_text": "".join(episode["turns"])}


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


def _visible_text(text: str) -> str:
    """The turn without its reasoning, which is what the recipe parses tool calls and the answer from.

    The generation prompt opens a `<think>` block, so a turn that never closes it is all reasoning.
    """
    _, closed, tail = text.partition("</think>")
    return tail if closed else ""


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


# --- TRL side -------------------------------------------------------------------------------------------------------


def completion_payload(
    model: str, prompt_ids: list[int], max_tokens: int, temperature: float, cache_salt: str | None = None
) -> dict:
    """AsyncRolloutLoop.generate_one_turn's /v1/completions payload, with a per-request `max_tokens` (the episode's
    context budget) and miles' sampling (full vocabulary)."""
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "n": 1,
        "logprobs": 0,
        "seed": random.getrandbits(63),
    }
    if cache_salt is not None:
        # Per-version prefix-cache salt (TRL #8212): the server pauses with clear_cache=false across weight syncs.
        payload["cache_salt"] = cache_salt
    return payload


def parse_choice(output: dict) -> tuple[list[int], list[float], str, str]:
    choice = output["choices"][0]
    return choice["token_ids"], choice["logprobs"]["token_logprobs"], choice["text"], choice["finish_reason"]


class Text2SQLLoop(AsyncRolloutLoop):
    """Runs miles' text-to-SQL episode per rollout and scores it inline (`Rollout.reward`).

    Advantage: reward minus the group mean, no std normalization (miles: `--advantage-estimator grpo
    --disable-grpo-std-normalization`). Every finished episode is appended to `T2SQL_EPISODES_FILE`.
    """

    async def setup(self):
        await super().setup()
        # generate_one only receives the prompt messages; the row's task is found by its prompt (unique, checked by
        # convert.py)
        self._tasks = {row["prompt"][-1]["content"]: json.loads(row["task"]) for row in self.dataset}
        self._episodes = open(os.environ["T2SQL_EPISODES_FILE"], "a", buffering=1)

    async def close(self):
        self._episodes.close()
        await super().close()

    async def generate_one(self, prompt, tools, group_id=0):
        task = self._tasks[prompt[-1]["content"]]

        async def sample(prompt_ids, max_tokens):
            payload = completion_payload(
                self._request_model, prompt_ids, max_tokens, self.config.temperature, f"v{self.model_version}"
            )
            output = await self._retry(
                lambda: self._post("/v1/completions", payload, self.config.vllm_server_timeout),
                max_attempts=30,
                label="vllm /v1/completions",
            )
            return parse_choice(output)

        t0 = time.time()
        result = await run_episode(sample, self.tokenizer, task, evaluation=False)
        record = {
            **result["record"],
            "group_id": group_id,
            "model_version": self.model_version,
            "start": t0,
            "end": time.time(),
        }
        self._episodes.write(json.dumps(record) + "\n")
        turns = result["turns"]
        completion_ids = [i for t in turns for i in t.output_ids]
        completion = [{"role": "assistant", "content": result["completion_text"]}]
        return Rollout(completion, completion_ids, turns, tool_call_count=int(record["tool_calls"]),
                       reward=result["reward"])

    def compute_advantages(self, group):
        rewards = group.rewards
        scored = ~np.isnan(rewards)
        advantages = np.zeros_like(rewards)
        if scored.any():
            advantages[scored] = rewards[scored] - rewards[scored].mean()
        return advantages
