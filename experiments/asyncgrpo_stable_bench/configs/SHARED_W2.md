# W2 (agentic terminal tasks) shared config

Every library maps these values onto its own knobs in `configs/<lib>/EQUIVALENCE.md` (W2 section). Where a library can't express a value, use its closest setting and write down the gap. Revisions to this file apply to every library and are logged at the bottom. The SHARED_W1 rules carry over unless this file says otherwise: reward call safety (rev. 7), engine at CLI defaults (rev. 3), packing correctness on Qwen3.5 (rev. 3), definitions of weight sync, clip reference, staleness, checkpoints and startup (rev. 4-6).

## Environment

- **Tasks:** TermGrade (`ai-and/termgrade-environments`, 1,004 Harbor terminal tasks, Apache-2.0), split `unseen` only (the 802 tasks ai& never trained on). Each task is a container, an `instruction.md` and a pytest verifier (`tests/test.sh` -> `/logs/verifier/reward.txt`, 0 or 1). Reward = the verifier's score; no shaping, no format reward.
- **Sandbox backend:** HF Sandboxes (`hf-sandbox`, flavor `cpu-basic`), through Harbor 0.25.0 with the `tar --no-same-owner` fix (`workloads/agentic/harbor-0.25.0-hf-sandbox-no-same-owner.patch`).
- **Image:** TermGrade images are local builds (`FROM dsbase-venv312:local` + per-task `setup.sh` + tmux/asciinema), never pushed to a registry, and HF Sandboxes only pull prebuilt images. So the shared part is built once as a Docker Space, `hf.co/spaces/aminediroHF/termgrade-base` (`workloads/agentic/base_image/Dockerfile` = TermGrade's base image verbatim + the tmux/asciinema step of every task Dockerfile), and the per-task part runs at sandbox start: `workloads/agentic/prepare.py` writes Harbor task dirs whose `environment/` (setup.sh + prepare.sh) is uploaded to `/app` and whose `[environment.healthcheck]` runs prepare.sh once before the agent starts. prepare.sh does what the rest of the task Dockerfile does: `setup.sh` with errors ignored, then deletes the tests, the solution and itself.
- **Network:** TermGrade tasks are `no-network`. HF Sandboxes can't disable internet, and setup.sh installs packages at start, so the prepared tasks are `public`. The agent can reach the internet; the verifier tests are never in the sandbox during the agent phase (uploaded at scoring time, as in TermGrade).
- **Server:** each library runs Harbor's `Trial` its own way (TRL: OpenEnv's Harbor server `openenv harbor serve` at OpenEnv `86a180ede`, SkyRL / miles: Harbor in-process). The prepared task directories are the shared input: `/fsx/amine_dirhoussi/stable-bench-src/termgrade/harbor/{train,eval}/tasks/<task>/`, built by `workloads/agentic/prepare.py` from the task lists in `workloads/agentic/{train,eval}_tasks.txt`.

## Harness

- **terminus-2** (Harbor's default agent: the model answers in JSON with keystrokes for a tmux session in the sandbox; no tool-call parser). It runs host-side (in the Harbor process), so model calls never leave the node and no tunnel is needed. Every library that runs Harbor can run it; TermGrade's published pass rates were measured with it.
- **Summarisation off** (`enable_summarize=False`, `proactive_summarization_threshold=0`): a summariser's turns would be trained with the episode's reward, and compaction breaks the prefix the trainer stitches turns on. A context overflow ends the episode, which is then verified as-is.
- **Budgets:** agent wall clock 600 s per episode (Harbor `agent_timeout_sec`), 30 model calls per episode (`agent_step_limit`), 8192 output tokens per call (OpenEnv `--max-output-tokens`, terminus-2 `max_output_tokens`), max model length 65,536 (prompt + completion of a single call). Sandbox start, setup and verification have their own timeouts (setup 600 s, verifier 300 s) and don't count against the agent budget.

## Model and task set

- **Model: `Qwen/Qwen3.5-9B`** (size M), thinking off, full fine-tune on one node (4 inference + 4 training GPUs; TRL fits it under FSDP2 with fp32 master weights). The 2B part of the probe was cancelled to keep sandbox load down; 9B already has nonzero success on most of the pool.
- **Candidate pool:** 128 `unseen` tasks whose setup.sh has no apt/curl/wget/systemctl (fast, reliable sandbox start), 64 that Qwen3.6-27B always solves and gemma-4-31B (no thinking) solves >= 75%, 64 that Qwen3.6-27B solves 50-88% (`workloads/agentic/select_probe.py`). Every candidate passes the oracle check (128/128); **15 pass with the nop agent** (setup.sh already produces what the tests check) and are excluded (`nop_pass_tasks.txt`), leaving 113.
- **Base pass rates** (probe 364979, 113 tasks x 4 episodes, the training harness, budgets and sampling): Qwen3.5-9B pass@1 **0.386** (448 scored of 452). Per task: 43 never solved, 18 always, 52 in [0.2, 0.8]. Half the episodes hit the 600 s agent budget (verified as-is). Episodes: 24 model calls mean, last-call context 34k tokens mean (p90 54k), 15k sampled tokens, 537 s wall.
- **Split** (`select_split.py`, on 3 episodes per task so the run could start): tasks with base pass rate in [0.2, 0.8]: **28 train** (mean base pass 0.53), **14 eval** (0.48), `workloads/agentic/{train,eval}_tasks.txt`. Small; a later revision can widen the pool (another 300 fast `unseen` tasks exist).

## Training (size M; S reuses it)

| knob | value |
|---|---|
| model | `Qwen/Qwen3.5-9B`, full fine-tune, bf16 compute; master weight dtype = the library's default (record it) |
| thinking | off: `enable_thinking=False` in the chat template (vLLM `--default-chat-template-kwargs`) |
| tasks per optimizer step | 8 |
| episodes per task (group size) | 8 (64 episodes per step) |
| sampling | temperature 1.0, top_p 1.0, no top_k, no min_p, repetition_penalty 1.0 |
| optimizer | AdamW, lr 1e-6 constant, no warmup, weight decay 0.0, betas (0.9, 0.999), grad clip 1.0 |
| optimizer steps per batch | 1 |
| loss | as W1: GRPO, group-mean advantage (no std normalization), clip eps 0.2 / 0.28 against the sampler's logprobs, token-level aggregation, no KL, no entropy bonus |
| which tokens train | the model's own sampled tokens (loss mask 1); the system prompt, the instruction and every terminal observation are context only (mask 0) |
| failed episodes | an episode the verifier never scored (sandbox or harness error, capture rejected) has no reward and is excluded from training and from its group's baseline. An episode that hits the agent time or call budget, or overflows the context, is verified as-is and trained with that score. Each library records how it handles them (e.g. SkyRL zeroes the whole group), and every run reports the counts |
| batch composition | the step trains on the next 8 complete groups as they arrive; no refill or oversampling of zero-advantage groups |
| off-policy | the library's recommended async setting (record max staleness) |
| concurrency | at most 64 episodes in flight = at most 64 live sandboxes (the HF Sandbox service dropped episodes in bursts above ~64-100 per user in the agent-swarm runs) |
| total steps | 60 |
| checkpoints | HF format at steps 20, 40, 60 |
| hardware | 1 node, 8 B200, `--exclusive`: 4 GPUs for inference, 4 for training |
| inference engine | the library's default; record the engine and version |
| seed | 42 |

## Eval

`workloads/agentic/probe.py` on the held-out split (`eval_tasks.txt`), 4 episodes per task, same harness, budgets and sampling as training, on the base model and on the step-20/40/60 checkpoints served by vLLM. Reported: mean verifier reward (pass@1 averaged over the 4), and the counts of unscored episodes.

## Metrics (on top of collect/SCHEMA.md)

Throughput is counted per rollout, without deduplication, as in W1:

- `trained_tokens`: the sum over the step's training samples of each sample's full token sequence (prompt + completion), no deduplication. With terminus-2 on Qwen3.5 every model call is its own sample, because the harness re-renders the chat each turn and the template renders a past assistant turn differently from what was sampled (OpenEnv flags it `per_turn_capture_only`). So an episode counts as ~24 samples whose prompts repeat its history: ~515k tokens per episode vs ~34k of final context. This is what a library with no prefix sharing forwards for the same episodes.
- `loss_tokens`: tokens with loss mask 1 (the model's own sampled tokens).
- `forwarded_tokens`: the tokens the library actually ran through the model for the step (after its own prefix sharing or tree packing; a library that trains every turn as its own sample forwards each turn's whole prompt again).
- `prefix_share_ratio` = `trained_tokens / forwarded_tokens` (> 1: shared prefixes forwarded once; < 1: histories forwarded more than once).

Per episode (`episodes.jsonl`, one row per finished episode): `start`, `end`, `wall_s`, `task`, `reward` (null if unscored), `status` (`scored`, `budget_scored`, `failed`), `error`, `turns`, `completion_tokens`, `seq_tokens` (the last call's prompt + completion), Harbor's phase times (`environment_setup` = sandbox start + task setup, `agent_setup`, `agent_execution`, `verifier`), and `model_s` (time inside model calls) / `sandbox_s` (everything else: sandbox start, setup, command execution, verifier) when the library exposes per-call timing. Per step (`steps.jsonl` additions): `episodes_finished`, `episode_wall_mean`, `turns_mean`, `tokens_per_episode_mean`, `failed_episodes`, `budget_episodes`, `forwarded_tokens`, `loss_tokens`, `prefix_share_ratio`.

## What each library needs to run W2

Every library starts from the same prepared task dirs and the same base image, and must apply the Harbor tar patch to whatever Harbor it installs (`workloads/agentic/harbor-0.25.0-hf-sandbox-no-same-owner.patch`, written against the PyPI 0.25.0 wheel; check the line it edits on other versions). Sandbox concurrency, cleanup and the orphan sweep (`workloads/agentic/sweep_sandboxes.py`) apply to everyone.

- **TRL:** done (`configs/trl/w2-M.py`, `launch/trl/w2.sbatch`). OpenEnv Harbor server + `HarnessAsyncGRPOTrainer`, agent-owned loop, captured tokens.
- **SkyRL:** runs Harbor in-process with terminus-2 already (`examples/train_integrations/harbor`, `collect_rollout_details: true` for token capture). Needs its Harbor pin (git `3de07a0e`, v0.13.1+25) overridden to >= 0.21 for `hf-sandbox` (the names it imports still exist on 0.25), the tar patch, `environment.type: hf-sandbox` in the trial config, and the trial config's terminus-2 kwargs set to the budgets above (summarisation off). Its failure handling differs: a failed trajectory zeroes its whole group (record it).
- **miles:** Harbor connector (`examples/experimental/harbor`, in-process `Trial.run()`) needs the hf-sandbox environment backported onto its `harbor-miles-v0.20.0` branch (that branch carries the terminus-2 truncation policy miles needs for token-in/token-out), plus an entry in `miles/rollout/agentic/credentials.py`. Alternative: point its OpenEnv connector at an OpenEnv Harbor server we run (the TRL setup), which needs no Harbor change on the miles side.
- **prime-rl:** verifiers' `HarborTaskset` parses Harbor tasks and has a `terminus_2` harness, but runs each task in its own box runtime (Prime Sandboxes / docker / subprocess). Needs a new verifiers `RuntimeBackend` for HF Sandboxes (`start`, `exec` with streaming, `write`, `cleanup`; prime.py is 572 lines) and a tunnel for its interception server, and its taskset must honour the task's healthcheck (or run prepare.sh itself) since that is where TermGrade's setup happens here.
- **verl:** no Harbor integration. Either a `HarborAgentLoop(AgentLoopBase)` (~250-400 LOC) that drives an OpenEnv Harbor server (reset/step) with its own tokenization, or verl-recipe's `swe_agent` model-proxy pattern to let terminus-2 call verl's vLLM; both are new code, and the first loses terminus-2 (it becomes verl's own loop over the sandbox).

## Revisions

- rev. 0 (2026-11-10 ~05:30 UTC): first version, from the TRL W2 setup (envs/trl/NOTES.md, W2 section).
