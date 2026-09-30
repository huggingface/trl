# AsyncGRPO on the General domain of `XiaomiMiMo/MiMo-V2.6-RL-oss`

Trains MiMo's own agent on MiMo's own knowledge-work environments, with [`AsyncGRPOTrainer`](https://huggingface.co/docs/trl/async_grpo_trainer) and one [Hugging Face sandbox](https://huggingface.co/docs/huggingface_hub/guides/sandbox) per rollout. A port of [`XiaomiMiMo/verl`](https://github.com/XiaomiMiMo/verl)'s `recipes/general` onto TRL's loop-owning OpenEnv path.

## The environment

The General domain is 925 enterprise tasks: accounting and tax, legal compliance, healthcare operations, HR, consulting. Each task is a directory holding

- `workspace/` — the office documents the agent reads, xlsx, docx, pdf, pptx, csv,
- `system/` and `tools/` — a few business systems (a general ledger, a procurement hub, a case tracker) served to the agent as MCP tools over SQLite state,
- `verify.py` and `verifier_meta.json` — a rubric of five to nine assertions, most judged by an LLM against a hidden answer key, scored as a weighted value in `[0, 1]`.

The agent is [`mimoagent`](https://github.com/XiaomiMiMo/MiMo-Agent)'s Claude-Code-style agent: Bash, Read, Write, Edit, Grep and Glob, plus one tool per function of each business system. Prompts, tool config, step limit and the anti-reward-hacking guard are the ones the dataset was trained with (`config/agent/general/s3k.yaml` upstream).

```
 trainer (AsyncGRPOTrainer, FSDP2)                                vLLM
 ┌────────────────────────────────────────────────┐              ┌──────────────────────────┐
 │ HarnessRolloutWorker                           │  chat        │ Qwen3-8B (+ adapter)     │
 │  └ TrainingSessionFactory.create()             │  completions │                          │
 │      ├ Sandbox.create(general-agent-env-0)     ├─────────────▶│                          │
 │      ├ install the task, start its MCP servers │              └──────────────────────────┘
 │      ├ mimoagent CCAgent ──── Bash/Read/... ───┼───┐
 │      │                  └──── mcp__<system>__* ┼───┤
 │      └ verify(): the task's rubric + LLM judge ┼───┤
 └────────────────────────────────────────────────┘   ▼
                                    one HF sandbox per rollout, cpu-basic
```

- **The agent owns the loop.** `mimoagent` renders the prompts, calls the model, runs its tools in the sandbox and stops when it answers without a tool call. TRL never samples a turn.
- **TRL reads the trace.** `TracingModel` records every model call the way OpenEnv's interception proxy would: the request as sent, the generated token ids, their logprobs. `HarnessRolloutWorker` rebuilds one training row per turn and trains with GRPO on the outcome reward.
- **The reward is the task's own.** After the agent stops, its final reply is written as `workspace/answer.md`, the verifier materials are uploaded (only then, so the agent never sees the answer key), and the rubric is scored in the sandbox. `--reward binary` reproduces upstream's all-or-nothing reward; `--reward score` keeps the weighted value, which is what a small policy can climb.

## Running

```sh
# once, from a login node: the task directories are ~50 small files each
HF_HUB_CACHE=/fsx/$USER/hf-hub python -c "from probe_general import load_tasks; load_tasks(1000, 0, language='en')"

# measure the base policy, no training
python probe_general.py --model Qwen/Qwen3-8B --n-tasks 64 --samples-per-task 4 --output probe.jsonl

# train, one node: vLLM on the first GPUs, the trainer on the rest
./run_slurm.sh                       # both packings, so their curves can be read side by side
PACKINGS=tree ./run_slurm.sh         # just one
LORA_RANK=32 TRAIN_GPUS=2 ./run_slurm.sh
```

`probe_general.py` prints the reward distribution, the per-task spread that GRPO trains on, and an estimate of what tree packing would fold. Its `trainable_instances.json` is the task list where the base policy scores above zero, which is what a short run should train on.

## Two things this example fixes in the environment

Both are applied to the copies uploaded into each sandbox, so upstream's grading is unchanged.

- **The MCP bridge imported a library per tool call.** Every tool call spawned a Python process whose `import mcp.client.streamable_http` alone measures ~12 s inside a sandbox, against 0.12 s for the interpreter. `mcp_bridge_fast.py` speaks the same streamable-http JSON-RPC with the standard library: identical tool lists, results and errors, 2.9× faster per call.
- **The verifier judged rubric items one at a time.** A task has a median of five LLM-judged items, each a separate call to the judge, and the verifier runs inside the sandbox while the rollout still holds its slot. `PARALLEL_JUDGE` appends a concurrent override of that loop to the uploaded `verify.py`; weights, gates and votes are untouched.

Together they roughly halve rollout wall time, which matters because the trainer is otherwise starved: a rollout spends about 2% of its life inside a model call.

## Notes

- The judge is any OpenAI-compatible endpoint reachable from the sandbox, by default Qwen3-235B on the Hugging Face router with GLM as a fallback chain. Use a **non-reasoning** judge: the verifier caps a verdict at 4000 tokens and cannot disable thinking, so a reasoning model loses about 13% of verdicts.
- The public image ships MCP 2.x while the task scripts are written for 1.x, so each sandbox builds a small `mcp<2` venv at the path the entrypoint hardcodes.
- `SandboxPool` does not work for these tasks: a pooled sandbox is an unprivileged home on a shared host, so it cannot write the `/work` paths a task installs itself into.
- A full fine-tune of an 8B on 45k-token rows needs seven FSDP ranks; on fewer, use `LORA_RANK`.
