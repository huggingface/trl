"""CPU check of configs/prime-rl/t2sql_env.py against TRL's port (workloads/text2sql).

1. The taskset loads the 2,033 train tasks.
2. Reward, as TRL's test_reward.py on the same tasks (first N per split): gold SQL in <solution> scores 1.0 (eval
   config), a wrong query 0, no <solution> -1, unsafe -1.
3. Episode loop with a scripted model (gold query as a structured tool call, then the gold answer; and the same with the
   call as <tool_call> text, which is how TRL sees it): both reach the same reward.
4. Tokens: prime-rl's qwen3.5 renderer (thinking on, thinking_retention "all", as the run's config) renders the opening
   prompt with the tool schema exactly as TRL's apply_chat_template, parses a sampled tool call into the structured call
   the recipe reads, and bridges an observation onto the sampled ids exactly as TRL appends them.

Usage: python test_t2sql_env.py <data dir> <n tasks per split>
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import t2sql_env as env
from renderers import create_renderer
from renderers.configs import Qwen35RendererConfig
from text_to_sql_recipe import prompts


def episode(text, unsafe=False):
    return {"turns": [text], "final_text": text, "unsafe": unsafe, "tool_calls": 0, "tool_protocol_events": []}


async def check(task):
    gold = episode(f"<solution>{task['gold_sql']}</solution>")
    out = {
        "gold_eval": await env.reward_func(task, gold, evaluation=True),
        "wrong": await env.reward_func(task, episode("<solution>SELECT 123456789</solution>"), evaluation=True),
        "none": await env.reward_func(task, episode("I think the answer is 3."), evaluation=True),
        "unsafe": await env.reward_func(task, episode("", unsafe=True), evaluation=True),
    }
    return {k: (v["score"], v["correct"], v["error_category"]) for k, v in out.items()}


def scripted_chat(task, structured):
    """A model that queries the gold SQL once, then answers with it (reasoning split off, as the renderer does)."""
    sql = task["gold_sql"]
    replies = []
    if structured:
        call = {"id": "call_0", "type": "function",
                "function": {"name": "execute_sql_query", "arguments": json.dumps({"query": sql})}}
        replies.append({"role": "assistant", "content": "", "reasoning_content": "Let me check.\n", "tool_calls": [call]})
    else:
        text = f"<tool_call>\n<function=execute_sql_query>\n<parameter=query>\n{sql}\n</parameter>\n</function>\n</tool_call>"
        replies.append({"role": "assistant", "content": text, "reasoning_content": "Let me check.\n"})
    replies.append({"role": "assistant", "content": f"<solution>{sql}</solution>", "reasoning_content": "Done.\n"})
    seen = []

    async def chat(messages):
        seen.append(messages)
        message = replies[len(seen) - 1]
        return {"message": message, "finish_reason": "stop"}, {"prompt_tokens": 4000, "completion_tokens": 100}

    return chat, seen


async def main(data_dir, n):
    failures = 0
    tok = env.tokenizer()

    tasks = list(env.T2SQLTaskset(env.T2SQLConfig(data=str(data_dir / "train.jsonl"))).load())
    print(f"1. taskset: {len(tasks)} train tasks, first id {tasks[0].data.id}", flush=True)
    failures += len(tasks) != 2033

    print("2. reward on the first", n, "tasks per split (TRL: validation 100/100, arcwise 100/100, hard 99/100, train 96/100)")
    for split in ("validation", "train", "hard", "arcwise"):
        rows = [json.loads(json.loads(line)["task"]) for line in open(data_dir / f"{split}.jsonl")]
        t0 = time.time()
        results = await asyncio.gather(*(check(t) for t in rows[:n]))
        gold_ok = sum(r["gold_eval"][0] == 1.0 for r in results)
        other_bad = [(t["id"], r) for t, r in zip(rows, results)
                     if r["wrong"][0] != 0.0 or r["none"][0] != -1.0 or r["unsafe"][0] != -1.0]
        failures += len(other_bad)
        print(f"   {split}: gold eval score 1.0 on {gold_ok}/{len(results)} ({time.time() - t0:.0f}s); wrong/none/unsafe "
              f"failures {len(other_bad)}; gold misses {[t['id'] for t, r in zip(rows, results) if r['gold_eval'][0] != 1.0]}",
              flush=True)

    print("3. episode loop, scripted gold model, 20 train tasks")
    rows = [json.loads(json.loads(line)["task"]) for line in open(data_dir / "train.jsonl")][:20]
    diffs = []
    for task in rows:
        scores = {}
        for structured in (True, False):
            chat, seen = scripted_chat(task, structured)
            ep = await env.run_episode(chat, task)
            scores[structured] = (await env.reward_func(task, ep, evaluation=False))["score"]
            correct = (await env.reward_func(task, ep, evaluation=True))["correct"]
            assert len(seen) == 2 and ep["tool_calls"] == 1 and seen[1][-1]["role"] == "user", ep
            assert seen[1][-1]["content"].startswith("Results for execute_sql_query"), seen[1][-1]
            assert "You have 4 turns left" in seen[1][-1]["content"]
        diffs.append(abs(scores[True] - scores[False]))
        failures += correct != 1.0
    print(f"   structured vs text tool call: max |reward diff| {max(diffs):.3f}; all correct: {failures == 0}")
    failures += max(diffs) > 1e-9

    print("4. tokens: renderer vs TRL's apply_chat_template / observation framing")
    renderer = create_renderer(tok, Qwen35RendererConfig(enable_thinking=True, thinking_retention="all"))
    tools = [prompts.sql_tool_schema({})]
    mismatch = 0
    lens = []
    for task in rows[:20] + [json.loads(json.loads(line)["task"]) for line in open(data_dir / "arcwise.jsonl")][:20]:
        messages = prompts.opening_messages(task, few_shot=True)
        trl = tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools),
                         add_special_tokens=False)
        ours = renderer.render_ids(messages, tools=tools, add_generation_prompt=True)
        mismatch += trl != ours
        lens.append(len(ours))
    print(f"   opening prompt: {40 - mismatch}/40 identical; tokens mean {statistics.mean(lens):.0f}; "
          f"tail {tok.decode(ours[-12:])!r}")
    failures += mismatch

    prompt_ids = renderer.render_ids(messages, tools=tools, add_generation_prompt=True)
    sql = task["gold_sql"]
    sampled = (f"Let me look.\n</think>\n\n<tool_call>\n<function=execute_sql_query>\n<parameter=query>\n{sql}\n"
               f"</parameter>\n</function>\n</tool_call>")
    completion_ids = tok.encode(sampled, add_special_tokens=False) + [tok.eos_token_id]
    parsed = renderer.parse_response(completion_ids, tools=tools, prompt_ids=prompt_ids)
    print(f"   parsed sampled turn: reasoning {parsed.reasoning_content!r}, content {parsed.content!r}, "
          f"tool_calls {[(c.name, c.arguments) for c in parsed.tool_calls]}")
    call_ok = len(parsed.tool_calls) == 1 and parsed.tool_calls[0].name == "execute_sql_query"
    failures += not call_ok
    for new in ([{"role": "user", "content": "Results for execute_sql_query:\nSQL execution: 1 row(s)\nx\n1"}],
                [{"role": "user", "content": "obs"}, {"role": "user", "content": prompts.SQL_LIMIT_PROMPT}]):
        bridged = renderer.bridge_to_next_turn(prompt_ids, completion_ids, new, tools=tools)
        trl = prompt_ids + completion_ids + env._observation_ids(tok, new, last_token_id=completion_ids[-1])
        same = bridged is not None and bridged.token_ids == trl
        print(f"   bridge with {len(new)} user message(s): identical to TRL's appended ids: {same}"
              + ("" if same else f"\n     TRL tail {tok.decode(trl[-40:])!r}\n     ours {tok.decode(bridged.token_ids[-40:]) if bridged else None!r}"))
        failures += not same
    print("5. verifiers scoring: Task.score() on a trace carrying the scripted gold episode")
    import verifiers.v1 as vf

    task = tasks[0]
    chat, _ = scripted_chat(json.loads(task.data.task), structured=True)
    trace = vf.Trace(agent=vf.AgentInfo(config=vf.AgentConfig()),
                     task=vf.TraceTask(type="T2SQLTask", data=task.data, key="k", hash="h"))
    trace.info["t2sql"] = await env.run_episode(chat, json.loads(task.data.task))
    await task.score(trace)
    expected = (await env.reward_func(json.loads(task.data.task), trace.info["t2sql"], evaluation=False))["score"]
    print(f"   rewards {trace.rewards}, reward {trace.reward}, metrics {trace.metrics}, expected {expected}")
    failures += abs(trace.reward - expected) > 1e-9
    print("ALL OK" if failures == 0 else f"{failures} FAILURES")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), int(sys.argv[2])))
