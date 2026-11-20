"""CPU check for configs/openrlhf/w2_agent.py (launch/openrlhf/w2_check.sbatch), no GPU, no OpenRLHF trainer.

  1. Template: does OpenRLHF's shipped OpenAI-server executor's prefix assertion hold on Qwen3.5-9B for a terminus-2
     style second call? And what does our delta tokenization append?
  2. Scripted episodes through the real Harbor server (local Apptainer boxes) with a fake llm_engine:
     replay of solved trajectories from the verl W2 run (expect the original reward 1) and a nop reply (expect 0).
Usage: python w2_check.py <model_dir> <trials_dir> <split_dir>
"""
import asyncio, glob, json, os, sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../configs/openrlhf"))
import w2_agent
from openrlhf.utils.utils import get_tokenizer
from vllm import SamplingParams

model, trials, split = sys.argv[1:4]
proc = get_tokenizer(model, None)
tok = getattr(proc, "tokenizer", proc)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
render = lambda m: proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)

# 1. template
msgs = [{"role": "user", "content": "Do the task. Reply in JSON."}]
reply = '{"analysis": "look", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}], "task_complete": false}'
p1 = tok(render(msgs), add_special_tokens=False)["input_ids"]
buf = p1 + tok(reply, add_special_tokens=False)["input_ids"] + [im_end]
msgs2 = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "New Terminal Output:\n$ ls\nfoo"}]
full2 = tok(render(msgs2), add_special_tokens=False)["input_ids"]
ok = full2[: len(buf)] == buf
print(f"TEMPLATE shipped OpenAI-executor prefix assertion holds on call 2: {ok}")
if not ok:
    i = next(k for k in range(min(len(buf), len(full2))) if buf[k] != full2[k])
    print(f"  first mismatch at token {i}: sampled ...{tok.decode(buf[i-5:i+8])!r} vs re-render ...{tok.decode(full2[i-5:i+8])!r}")
print(f"  render of call 1 ends: {render(msgs)[-60:]!r}")
print(f"  render of call 2: {render(msgs2)!r}")


class Out:
    def __init__(self, ids):
        self.outputs = [SimpleNamespace(token_ids=ids, logprobs=None, finish_reason="stop", text="")]


class FakeEngine:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    async def generate(self, ids, sp):
        self.prompts.append(list(ids))
        r = self.replies.pop(0) if self.replies else '{"analysis": "done", "commands": [], "task_complete": true}'
        return Out(tok(r, add_special_tokens=False)["input_ids"] + [im_end])


ep = w2_agent.Episode(SamplingParams(max_tokens=8192), 65536, proc, FakeEngine([reply]))
asyncio.run(ep.chat({"messages": msgs}))
asyncio.run(ep.chat({"messages": msgs2}))
print(f"DELTA appended text after call 1: {tok.decode(ep.ids[ep.ranges[0][1]:])!r}")
print(f"DELTA sequence decodes to: {tok.decode(ep.ids)!r}")

# 2. scripted episodes
names = sorted(n for n in os.listdir(os.path.join(split, "tasks")) if not n.startswith("."))
solved = []
for res in sorted(glob.glob(os.path.join(trials, "*", "result.json"))):
    r = json.load(open(res))
    rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    traj = os.path.join(os.path.dirname(res), "agent", "trajectory.json")
    name = (r.get("task_name") or "").split("/")[-1]
    if rew == 1 and os.path.exists(traj) and name in names:
        # terminus-2 stores parsed tool calls, not the raw reply: rebuild the JSON reply it parsed them from
        texts = []
        for st in json.load(open(traj))["steps"]:
            if st.get("source") != "agent":
                continue
            calls = st.get("tool_calls") or []
            cmds = [c["arguments"] for c in calls if c.get("function_name") == "bash_command"]
            done = any(c.get("function_name") == "mark_task_complete" for c in calls)
            texts.append(json.dumps({"analysis": "replay", "plan": "replay", "commands": cmds, "task_complete": done}))
        if texts:
            solved.append((name, texts))
seen, cases = set(), []
for name, texts in solved:
    if name not in seen and len(cases) < 3:
        seen.add(name); cases.append(("replay", name, texts, 1.0))
print(f"found {len(solved)} solved trajectories with text messages; replaying {len(cases)}")
for name in names[:2]:
    cases.append(("nop", name, ['{"analysis": "nothing", "commands": [], "task_complete": true}'], 0.0))


async def run(kind, name, texts, expect):
    ex = w2_agent.AgentExecutor()
    eng = FakeEngine(texts)
    out = await ex.execute(prompt="x", label=str(names.index(name)), sampling_params=SamplingParams(max_tokens=8192, logprobs=1),
                           max_length=65536, hf_tokenizer=proc, llm_engine=eng)
    acts = sum(e - s for s, e in out["action_ranges"])
    good = out["reward"] == expect and out["extra_logs"]["w2_unscored"] == 0
    print(f"EPISODE {kind} {name}: reward {out['reward']} (expect {expect}) calls {len(out['action_ranges'])} "
          f"seq {len(out['observation_tokens'])} action_tokens {acts} logprobs {len(out['rollout_log_probs'])} {'OK' if good else 'MISMATCH'}")
    return good


async def main():
    return await asyncio.gather(*[run(*c) for c in cases])

res = asyncio.run(main())
print("ALL OK" if all(res) else "SOME FAILED")
