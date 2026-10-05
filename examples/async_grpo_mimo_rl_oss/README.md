# AsyncGRPO on `XiaomiMiMo/MiMo-V2.6-RL-oss`

Trains MiMo's own agent on MiMo's own environments, with [`AsyncGRPOTrainer`](https://huggingface.co/docs/trl/async_grpo_trainer) and one [Hugging Face sandbox](https://huggingface.co/docs/huggingface_hub/guides/sandbox) per rollout. A port of [`XiaomiMiMo/verl`](https://github.com/XiaomiMiMo/verl)'s `recipes/general`, `recipes/code` and `recipes/arvo` onto TRL's loop-owning OpenEnv path.

The default policy is [`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B), the SFT checkpoint MiMo released as the starting point for RL on these environments.

## The domains

| `--domain` | tasks | the agent works on | the reward |
|---|---|---|---|
| `general` | 925 | a workspace of office documents plus a few business systems served as MCP tools | a rubric of five to nine items, most judged by an LLM against a hidden answer key, weighted into `[0, 1]` |
| `code` | 2698 | a repository in its broken state | a hidden test patch is applied and one command is run: 1 iff it exits zero |
| `cyber` | 1000 | a sanitizer-instrumented binary and its source | 1 iff the submitted input crashes in the function the task names |

Each task ships its own Docker image, and the agent is [`mimoagent`](https://github.com/XiaomiMiMo/MiMo-Agent)'s Claude-Code-style agent: Bash, Read, Write, Edit, Grep and Glob, plus one tool per function of each business system in General. Prompts, tool config, step limits and the anti-reward-hacking guard are the ones the dataset was trained with (`config/agent/general/s3k.yaml`, `config/agent/code/mini-claude-code.yaml` and `config/agent/arvo/arvo.yaml` upstream).

```
 trainer (AsyncGRPOTrainer, FSDP2)                                vLLM
 ┌────────────────────────────────────────────────┐              ┌──────────────────────────┐
 │ HarnessRolloutWorker                           │  chat        │ MiMo-9B (+ adapter)      │
 │  └ TrainingSessionFactory.create()             │  completions │                          │
 │      ├ Sandbox.create(the task's image)        ├─────────────▶│                          │
 │      ├ install the task                        │              └──────────────────────────┘
 │      ├ mimoagent CCAgent ──── Bash/Read/... ───┼───┐
 │      │                  └──── mcp__<system>__* ┼───┤
 │      └ verify(): the task's own verifier ──────┼───┤
 └────────────────────────────────────────────────┘   ▼
                                    one HF sandbox per rollout, cpu-basic
```

- **The agent owns the loop.** `mimoagent` renders the prompts, calls the model, runs its tools in the sandbox and stops when it answers without a tool call. TRL never samples a turn.
- **TRL reads the trace.** `TracingModel` records every model call the way OpenEnv's interception proxy would: the request as sent, the generated token ids, their logprobs. `HarnessRolloutWorker` rebuilds one training row per turn and trains with GRPO on the outcome reward.
- **The reward is the task's own.** Code and Cyber are graded by `mimoagent`'s own dataset environments, which apply the hidden test patch or read the grading server's verdict. General writes the agent's final reply as `workspace/answer.md`, uploads the verifier materials (only then, so the agent never sees the answer key) and runs the rubric in the sandbox.
- **Mixing domains needs nothing from the algorithm.** GRPO's advantages are relative to the other samples of the same prompt, so a domain whose rewards sit lower contributes the same unit-variance signal as one whose rewards sit higher.

## Running

```sh
# once, from a login node: the General task directories are ~50 small files each
HF_HUB_CACHE=/fsx/$USER/hf-hub python -c "import general_domain; general_domain.load_tasks(1000, 0, language='en')"

# measure the base policy on one domain, no training
python probe.py --domain code --n-tasks 64 --samples-per-task 4 --output probe.jsonl

# train, one node: vLLM on the first GPUs, the trainer on the rest
./run_slurm.sh                                   # general, code, and the three mixed
DOMAIN_SETS=general,code,cyber ./run_slurm.sh    # just the mixed run
```

`probe.py` prints the reward distribution, the per-task spread that GRPO trains on, and an estimate of what tree packing would fold. Its `trainable_instances.json` is the task list where the base policy scores above zero, which is what a short run should train on.

## What this example fixes in the environment

Grading is untouched in all three cases; the fixes are applied to the copies uploaded into each sandbox, or to the setup that precedes the agent.

- **The MCP bridge imported a library per tool call** (General). Every tool call spawned a Python process whose `import mcp.client.streamable_http` alone measures ~12 s inside a sandbox, against 0.12 s for the interpreter. `mcp_bridge_fast.py` speaks the same streamable-http JSON-RPC with the standard library: identical tool lists, results and errors, 2.9× faster per call.
- **The verifier judged rubric items one at a time** (General). A task has a median of five LLM-judged items, each a separate call to the judge, and the verifier runs inside the sandbox while the rollout still holds its slot. `PARALLEL_JUDGE` appends a concurrent override of that loop to the uploaded `verify.py`; weights, gates and votes are untouched.
- **The published images are not history-truncated** (Code). Upstream builds its images with the history cut at the task's base commit and asserts that property in setup; the images published with the dataset still carry the branches and release tags the fix was made on, so an agent could read the answer out of `git log`. `PublicOpenSourceCodeEnvironment` truncates in the sandbox instead, which is the fallback the assert itself names.

The first two roughly halve General's rollout wall time, which matters because the trainer is otherwise starved: a rollout spends about 2% of its life inside a model call.

## Notes

- **Packing is `sequence`, not `tree`.** Three of every four layers of Qwen3.5 are linear-attention layers, whose recurrent state cannot branch, so a packed prefix forest has no mask to express. They do take `cu_seqlens`, which is what sequence packing needs. On a dense policy (`MODEL=Qwen/Qwen3-8B TOOL_PARSER=hermes REASONING_PARSER=qwen3`) `--packing tree` folds the shared prefixes of a group into one row.
- The General judge is any OpenAI-compatible endpoint reachable from the sandbox, by default Qwen3-235B on the Hugging Face router with GLM as a fallback chain. Use a **non-reasoning** judge: the verifier caps a verdict at 4000 tokens and cannot disable thinking, so a reasoning model loses about 13% of verdicts.
- The public General image ships MCP 2.x while the task scripts are written for 1.x, so each sandbox builds a small `mcp<2` venv at the path the entrypoint hardcodes.
- `SandboxPool` does not work for these tasks: a pooled sandbox is an unprivileged home on a shared host, so it cannot write the paths a task installs itself into.
- Upstream runs Cyber's agent as an unprivileged user so it cannot reach the grading server's verdict file; a sandbox has one user, so the anti-hack guard stands in for that wall, as it does for General's sidecar.
- A full fine-tune of a 9B on 45k-token rows needs seven FSDP ranks; on fewer, use `LORA_RANK`.
