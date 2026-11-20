# TRL (AsyncGRPOTrainer) mapping of SHARED_W1

TRL main @5623e8b63, `trl/experimental/async_grpo/`. Config: [`w1-S.py`](w1-S.py) + [`w1-S.accelerate.yaml`](w1-S.accelerate.yaml), launched by `launch/trl/w1.sbatch`. Line references are to that commit.

| SHARED_W1 knob | value | TRL knob and value | semantics / gap |
|---|---|---|---|
| model | Qwen/Qwen3.5-2B, bf16 compute | `model="Qwen/Qwen3.5-2B"`, `dtype="float32"`, `bf16=True`, FSDP2 (`mixed_precision: bf16`) | Loaded as the text-only causal LM with TRL's tree attention (`attn_implementation="tree_fa3"`, FA4 hub kernel on B200). The vision tower is not loaded. Master weights and AdamW state are **fp32** (accelerate upcasts trainable params before FSDP sharding); forward/backward in bf16. Checkpoints are written in bf16. vLLM serves bf16. |
| prompts per optimizer step | 32 | `per_device_train_batch_size * gradient_accumulation_steps * num_processes = 256` rollouts | **Gap.** TRL's optimizer step is not a fixed number of rollouts. It is `gradient_accumulation_steps` packed micro-batches, each made of one row per rank holding up to `token_budget` unique tokens (`TokenBudgetBatcher`, `async_grpo_trainer.py` ~L355). The 256 only sizes the in-flight pool (`max_concurrent_rollouts = max_staleness * 256 = 1024`). Rollouts per step therefore float with completion length. The shared run uses `GRAD_ACCUM=16`, `TOKEN_BUDGET=49152`: 16 x 4 rows x ~41k filled tokens (smoke fill 0.82-0.88) is ~2.6M tokens, about 236 rollouts at the probe's ~11k mean length, and `batch/samples_per_step` (logged every step, `num_rollouts` in steps.jsonl) records the real number. Groups are not kept whole across steps either: a group's samples can straddle two steps. |
| generations per prompt | 8 | `num_generations=8` | same |
| max prompt length | 2048, never truncated (rev. 1) | none; vLLM `--max-model-len` = 2048 + `MAX_COMPLETION` (18432) | TRL has no prompt-length knob and never truncates prompts; a prompt only has to fit `max_model_len` with its completion. Longest train prompt under the Qwen3.5 template: see NOTES.md (prompt-length check). |
| max completion length | 16384 (rev. 2) | `max_completion_length=16384` (vLLM `max_tokens`), `MAX_COMPLETION` env | same. Truncated = last token is not EOS/pad (`completions/clipped_ratio`). Truncated completions are trained on with their reward (0 unless a box was closed before the cut). Probe at the base policy: 26% truncated at 16384 (NOTES.md item 20). |
| sampling / chat template | T 1.0, top_p 1.0, no top_k/min_p; `enable_thinking=False` (rev. 2) | `temperature=1.0, top_p=1.0, top_k=0, min_p=None`, `chat_template_kwargs={"enable_thinking": False}`; vLLM `--generation-config vllm` | same. The rollout worker renders prompts with the tokenizer's own template plus `chat_template_kwargs` (`async_rollout_worker.py:1173-1180`) and sends token ids to `/v1/completions`, so vLLM's chat template handling is not involved. `--generation-config vllm` stops vLLM from applying the model card's sampling defaults. Sampler logprobs are `processed_logprobs`. |
| optimizer | AdamW, lr 1e-6 const, no warmup, wd 0, betas (0.9, 0.999), clip 1.0 | `learning_rate=1e-6, lr_scheduler_type="constant", warmup_steps=0, weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, max_grad_norm=1.0`; optim = transformers default `adamw_torch_fused` | same |
| optimizer steps per batch | 1 | always 1 | TRL has no PPO epochs or mini-batch split: every consumed sample is trained on once. |
| loss | GRPO, clip 0.2 / 0.28, token-level over the batch, no KL, no entropy | `score_centering=False`, `epsilon=0.2`, `epsilon_high=0.28` | The clipped surrogate `-min(r*A, clip(r, 0.8, 1.28)*A)` per token. **Differences:** (1) `r = exp(logp_trainer - logp_vllm)`: the "old" logprobs are the vLLM sampler's, there's no recompute with the trainer's previous weights, so the ratio carries both staleness and trainer/sampler mismatch. (2) On top of the clip, tokens whose ratio is outside `[vllm_importance_sampling_clip_min, vllm_importance_sampling_clip_max] = [0.5, 5.0]` are dropped from the loss (masked IS, trainer default, kept). (3) Aggregation (`compute_loss`, ~L1455): each micro-batch's summed token loss is divided by that micro-batch's trained-token count across all ranks, and then by `gradient_accumulation_steps`. That is a mean over micro-batches of per-micro-batch token means, which equals the token mean over the step when the micro-batches carry equal token counts (they are budget-filled, so close). No KL term exists in this trainer; no entropy bonus. TRL's default is `score_centering=True` (a different objective); the shared run turns it off to match SHARED_W1, and the ladder treats it as an arm. |
| advantage | group mean, no std (unless only option) | `W1RolloutLoop.compute_advantages`: `r - mean(r)` over the group | TRL's default is `(r - mean) / (std + 1e-8)` with population std (`async_rollout_worker.py` ~L1372). Overridden through the public `loop_cls` hook, no source patch. NaN rewards (none here) get 0. |
| batch composition / zero-advantage groups | next 32 complete groups, no refill of zero-adv groups (rev. 1) | not expressible: zero-std groups always dropped, step filled by tokens | **Gap.** A group whose 8 rewards are all equal is dropped before training (`_score_group`, ~L1399), not switchable by config. With mean-centred advantages those samples would carry zero gradient anyway, but they would still count in the token denominator of a library that keeps them, so TRL's effective per-token step is larger by `1 / (1 - frac_zero_adv)` in token terms. Combined with the token-sized step, this **is** refill: the next step keeps pulling scored groups until its micro-batches are full, so a TRL step trains on the next ~256 *informative* rollouts, not on the next 32 groups. At step 1 of the smoke 53% of groups had zero reward std, so a TRL step consumes roughly twice as many prompts as a lib that keeps them. Loss denominator: zero-adv tokens never enter it (they are not in the batch). `frac_reward_zero_std` and `batch/groups_per_step` are logged per step. Matching rev. 1 exactly would need overriding the private `_score_group` (to keep the group) plus a custom group-count batcher in place of `TokenBudgetBatcher`; both are source-level changes, so the shared arm keeps TRL's behaviour and records it. |
| reward | `workloads/math/reward.py` | `math_reward` in `w1-S.py` calls `compute_score` on `tokenizer.batch_decode(completion_ids, skip_special_tokens=True)` | Raw completion text, thinking included, so a `\boxed{}` inside an unfinished `<think>` still counts. Called in a worker thread, so math_verify timeouts are off (SIGALRM only works on the main thread). |
| off-policy | lib's recommended | `max_staleness=4`, `weight_sync_steps=1`, `max_concurrent_rollouts=None` (-> 1024, adaptively capped from vLLM KV usage) | Trainer defaults. Samples more than 4 policy versions old are dropped (`sample/dropped_stale_total`). |
| total steps | 200 | `max_steps=200` | same (optimizer steps) |
| checkpoints | HF at 50/100/150/200 | `save_strategy="steps", save_steps=50`, FSDP `FULL_STATE_DICT` | `checkpoint-<step>/model.safetensors` (bf16) + tokenizer + optimizer/scheduler state; eval.py reads the dir directly. |
| hardware | 1 node, 8 B200, 4 inference + 4 train | vLLM `--data-parallel-size 4` on GPUs 0-3, `accelerate launch --num_processes 4` (FSDP2) on GPUs 4-7 | Weight sync over NCCL (`--weight-transfer-config '{"backend":"nccl"}'`), every step. |
| inference engine | lib default | vLLM (version in `envs/trl/setup.json`), `--api-server-count 8`, `--gpu-memory-utilization 0.9` | |
| seed | 42 | `seed=42`, vLLM `--seed 42`; data shuffle is the shared parquet order | The rollout worker walks the dataset in parquet order (already seed-42 shuffled). |

## Recommended arm (`ARM=recommended`, [`w1-S-recommended.env`](w1-S-recommended.env))

Everything above stays, except the trainer defaults on main come back for loss, advantage and batching:

| knob | shared arm | recommended arm (AsyncGRPO default) |
|---|---|---|
| objective | clipped GRPO, `score_centering=False`, eps 0.2 / 0.28 | `score_centering=True` (importance-weighted score minus its expectation over the sampler's top-8 logprobs, masked IS [0.5, 5.0]); `epsilon_high` unset |
| advantage | `r - mean(r)` (`W1RolloutLoop.compute_advantages`) | `(r - mean) / (std + 1e-8)` |
| token budget | 49152 per row | 49152 (same as shared since 2026-11-10; run 364705 used the default `max_model_len` 18432) |
| micro-batches per step | 16 | 16 (same as shared since 2026-11-10; run 364705 used 43). Batching is a user choice, not an algorithm default |
| sampler logprobs | none requested | `num_sampler_logprobs`: 8 (default) in 364705; 2 in the rerun (PAUSE_STALL.md) |
| async | `max_staleness=4`, adaptive concurrency, sync every step | same (these are the defaults) |
| zero-adv groups | dropped | dropped (default; no switch) |

## M (Qwen3.5-9B)

Config: [`w1-M.py`](w1-M.py) (a copy of `w1-S.py` with only `MODEL` changed), [`w1-M.accelerate.yaml`](w1-M.accelerate.yaml) (identical to S), [`w1-M-recommended.env`](w1-M-recommended.env). Launched with `SIZE=M launch/trl/w1.sbatch`. Everything in the tables above holds for M; only these change:

| knob | S | M | why |
|---|---|---|---|
| model | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3.5-9B` | SHARED_W1 size M |
| sharding | FSDP2 full shard, 4 ranks | same | 9B with fp32 master + AdamW is ~145 GB, ~36 GB per rank sharded. No change needed |
| activation checkpointing | on | on | already on at S |
| token budget x micro-batches | 49152 x 16 | 49152 x 16 | tokens per step equal to S. W2-M ran the same 9B at 65536 per row with 89 GB peak allocated / 124 GB reserved per rank, so 49152 fits without a smaller row |
| vLLM | DP 4, TP 1, `--gpu-memory-utilization 0.9` | same | 9B bf16 is ~19 GB per replica, leaving plenty of KV cache per GPU |
| recommended arm sampler logprobs | 8 (364705), 2 (rerun 365172) | `NUM_SAMPLER_LOGPROBS=2` in `w1-M-recommended.env` | top-8 saturates the vLLM API servers and stalls weight-sync pauses (PAUSE_STALL.md); k=2 is the 1.18 default candidate |
| checkpoint size | 30 GB per checkpoint | ~120 GB per checkpoint (model.safetensors 18.8 GB bf16, pytorch_model_fsdp.bin 37.6 GB fp32, optimizer.bin 71.6 GB), ~480 GB for the 4 saves | FULL_STATE_DICT gathers on rank 0. the W1-M smoke (365362) wrote one in ~2.8 min (W2-M: ~4.4 min), so the 4 saves add ~12-18 min to a 200-step run. Smoke 365362: ~79 s/step, peak reserved 107 GB per trainer rank (NOTES W1-M) |

# TRL mapping of SHARED_W2 (agentic)

Config: [`w2-M.py`](w2-M.py) (+ `w1-S.accelerate.yaml`, FSDP2), launched by `launch/trl/w2.sbatch`, venv `bench-trl-w2` (`envs/trl/install_w2.sh`). Rollouts go through `HarnessAsyncGRPOTrainer` (`trl/experimental/async_grpo/openenv_harness.py`) with no `harness_adapter`: the agent owns its loop, and TRL trains on the `TrainingTrace` OpenEnv's capture proxy recorded. Only knobs that differ from the W1 mapping above are listed.

| SHARED_W2 knob | value | TRL knob and value | semantics / gap |
|---|---|---|---|
| environment | TermGrade on HF Sandboxes via Harbor | OpenEnv Harbor server (`openenv harbor serve --expose direct`, OpenEnv 86a180ede) + `workloads/agentic/termgrade_env.py` `TermGradeSessionFactory` | One `run_rollout` websocket call per episode: the server starts the sandbox, runs setup (healthcheck), terminus-2, then the verifier, and returns the captured turns and the reward. Client keepalive 60/600 s, message timeout 3600 s (NOTES W2 #6). |
| harness | terminus-2, summarisation off | `harness="terminus-2"` | Host-side in the server process; the seam turns summarisation off and declares 65,536 input / 8,192 output tokens to litellm. |
| sampling | T 1.0, full vocabulary | `temperature=1.0, top_p=1.0, top_k=0, min_p=None`; the trainer builds `training_sampling(...)` with `repetition_penalty=1.0` and hands it to the factory | Applied by the capture proxy to every request the agent makes. Agent-owned loops require full-vocabulary sampling (TRL raises otherwise). |
| thinking | off | vLLM `--default-chat-template-kwargs '{"enable_thinking": false}'`; `chat_template_kwargs={"enable_thinking": False}` | The agent's requests are rendered by vLLM's chat endpoint, so the server flag is what applies. |
| per-call output cap | 8192 | Harbor server `--max-output-tokens 8192`; `max_completion_length=8192` is unused for agent-owned loops | |
| budgets | 600 s agent, 30 calls | `agent_timeout_sec=600`, `agent_step_limit=30` on the factory | An agent timeout keeps the captured turns and the verifier's score; a session failure is unscorable. |
| tasks per step x group | 8 x 8 = 64 episodes | `num_generations=8`; `gradient_accumulation_steps=GRAD_ACCUM`, `token_budget=65536` (shared-arm batching, as W1) | **Gap, as in W1:** the step is GRAD_ACCUM packed micro-batches of one <= 65,536-token row per rank, not 64 episodes. GRAD_ACCUM is sized from the probe's mean episode length so a step holds ~64 episodes; `batch/samples_per_step` records the real count. Zero-std groups are dropped, so a step keeps pulling scored groups (refill). `token_budget` = vLLM `max_model_len` = 65,536, so no episode is dropped as oversize. |
| failed episodes | excluded, not scored 0 | default: a session failure returns an empty rollout with reward `None`, left out of the group mean; after 10 x `max_concurrent_rollouts` failures in a row training stops | Same rule as the spec. |
| which tokens train | sampled tokens only | OpenEnv's `loss_mask` mapped to TRL's completion mask (`_turns_from_training_trace`) | Turns are stitched into one sample when each prompt extends the previous turn's prompt + completion; a rewritten history forks a new sample (`rollout/fork_frac`). |
| concurrency | <= 64 episodes in flight | `max_concurrent_rollouts=64` | Hard cap: the harness session thread pool has that many workers; the adaptive controller only moves below it. |
| async | library default | `max_staleness=4` | |
| reward safety (W1 rev. 7) | nothing scores on the event loop | `rollout_reward_fn=None`: the reward is the verifier's float, read in the session's pool thread | |
| trained tokens | per rollout, no dedup | `batch/forwarded_tokens_per_step x batch/packing_ratio` (sum of per-sample lengths) | `batch/trained_tokens_per_step` is the loss-mask count (`loss_tokens`); `batch/forwarded_tokens_per_step` is after tree packing (`forwarded_tokens`); `prefix_share_ratio = batch/packing_ratio`. |

## W2 on local containers (`SANDBOX=local`)

Same run with the sandboxes on the training node instead of HF Sandboxes, so TRL and prime-rl can be compared with the sandbox backend held on our own nodes (prime-rl's W2 runs on local podman). Launch: `SANDBOX=local SIZE=M b200-queue -J w2local-trl --nodes=1 --gres=gpu:8 --exclusive launch/trl/w2.sbatch`, run id `trl-w2-M-local-<jobid>`. Everything in the table above is unchanged (model, thinking off, tasks and split, terminus-2 and its budgets, 8 x 8 episodes, 64 in flight, 60 steps, checkpoints 20/40/60, trainer config, shared-arm batching, 4 vLLM + 4 trainer GPUs). What differs:

| | HF run (`trl-w2-M-shared-364987`) | local variant |
|---|---|---|
| Harbor environment type | `hf-sandbox` | `singularity` (`W2_SANDBOX=singularity`, read by `workloads/agentic/termgrade_env.py`) |
| runtime | HF Sandboxes (HF Jobs, `cpu-basic`: 2 vCPU / 16 GB per box) | unprivileged Apptainer 1.4.5 from the official release .deb (`container-tools/apptainer-env.sh`), one `singularity exec --fakeroot --containall --pid --writable-tmpfs` per episode on the training node; no per-box CPU or memory cap (Harbor's singularity env sets none) |
| image | `hf.co/spaces/aminediroHF/termgrade-base` | `container-tools/images/termgrade-base.sif`, built from the same Dockerfile |
| tasks | `termgrade/harbor/train` | `termgrade/harbor-local-sif/train`: same 28 tasks, only `docker_image` changed |
| box start | sandbox job start + upload + healthcheck | Apptainer start + Harbor's in-box FastAPI server bootstrap (apt-get update + a pip venv, from the internet, every start) + upload + healthcheck |
| env kwargs | none | none: OpenEnv's server passes no Harbor environment kwargs, so Harbor's default `--no-mount home,tmp,bind-paths` applies; DNS works with it on the GPU nodes (job 365442), and the launcher also sets `APPTAINER_BIND=/etc/resolv.conf` |
| cleanup | `workloads/agentic/sweep.sh` (HF API) | kill the Harbor server's process tree, remove `/scratch/$USER/w2-boxes-<jobid>` |

Box CPU and memory are sampled every 5 s into `boxes.csv` (`workloads/agentic/box_usage.py`). Per-episode records, the collector (`collect/trl.py`, `collect/agentic.py`) and the W2 schema are the same.
