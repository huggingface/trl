# NeMo-RL: notes

NeMo-RL = NVIDIA's RL library (Ray; Megatron-Core or Automodel/DTensor training; vLLM, SGLang or TRT-LLM generation). Repo https://github.com/NVIDIA-NeMo/RL, read at release `v0.8.0` (`5e012b45e`, 2026-11-06; `main` was `defdaef48`, 2026-11-12). Source checkout `stable-bench-src/nemo-rl`. Goal: text-to-SQL on Qwen/Qwen3.5-35B-A3B, one 8xB200 node, same env, data and eval as TRL run 373132 (workloads/text2sql/README.md). Times are `date -u`, 2026-11-12.

## Feasibility (read from v0.8.0)

- Install: recommended path is the NGC container (`docs/about/installation.md`: "The fastest way to get started is with the pre-built NGC container"; `docker pull nvcr.io/nvidia/nemo-rl:latest`). We use `nvcr.io/nvidia/nemo-rl:v0.8.0` via Enroot/Pyxis (`envs/nemo-rl/install.sbatch`); anonymous pull from nvcr.io works, no NGC key. Pins (`pyproject.toml`): torch 2.13.0, vLLM 0.29.0, TransformerEngine v2.14.1, flash-attn 2.8.1 (FA2, no FA4), cuDNN 9.20.
- Qwen3.5-35B-A3B: supported on both backends (`docs/guides/models/qwen/qwen3-5.md`: Megatron TP+EP+PP+CP, Automodel EP+CP; vLLM inference). Shipped recipes `examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-*` (Megatron ep16tp2cp2, ep16tp2-fp8, 6n4g async 1-off; Automodel ep16, dapo-4n8g ep32). The page marks the shipped recipes "Long-Run Convergence Validated" and shows a 100-step curve for `grpo-qwen3.5-35ba3b-dapo-4n8g-automodel-ep32` (DAPO math, val 0.33 -> 0.69). GDN: Megatron uses MCore kernels; Automodel dispatches to fla / causal-conv1d (installed in the v0.7.0+ container, `automodel` extra).
- MoE EP: yes, both backends (`automodel_cfg.expert_parallel_size`, `megatron_cfg.expert_model_parallel_size`); vLLM `expert_parallel_size`.
- LoRA: yes on both backends (`docs/guides/lora.md`; `automodel_cfg.lora_cfg` with HF module names in `target_modules`, `megatron_cfg.peft`). Shipped LoRA recipes are Nemotron-3-Nano only (`grpo-nanov3-30BA3B-2n8g-{automodel,megatron}-lora.yaml`); no Qwen3.5 LoRA recipe.
- Multi-turn tool use: two shipped routes. (1) `EnvironmentInterface` Ray actor + `grpo.max_rollout_turns` (`docs/guides/environments.md`, `examples/run_grpo_sliding_puzzle.py`): NeMo-RL generates each turn, calls `env.step(message_log, metadata)`, appends the returned observation tokenized with `tokenizer(text, add_special_tokens=False)`; per-turn tokens are clamped to `min(max_new_tokens, max_model_len - input)` (`vllm_worker_async.py:1708`). (2) NeMo Gym (agent + resource servers over HTTP, `examples/nemo_gym/`). We use (1): it maps one-to-one onto t2sql_episode's turn loop.
- B200: the 6n4g async recipe pins `moe_backend: flashinfer_trtllm` for vLLM (so NVIDIA runs it on Blackwell). No routing replay unless `router-replay` is turned on (`docs/guides/router-replay.md`), so the miles-022 failure mode (flashinfer_trtllm records no routing) doesn't apply by default. Training attention: the validated Automodel recipe uses `attn: sdpa` (torch SDPA, not TE), so the TE -> FA4 window bug that broke miles on B200 (miles-006/016) is not on this path.

## Setup choice

Coordinator (20:3x): use NeMo-RL's own recommended config for this model; keep our task data, reward and eval. So the config (`launch/nemo-rl/t2sql/t2sql.yaml`) inherits the long-run-validated `grpo-qwen3.5-35ba3b-dapo-4n8g-automodel-ep32.yaml` and changes only: one node (EP 32 -> 8, vLLM TP8 colocated as in the recipe), the post-trained `Qwen/Qwen3.5-35B-A3B` (TRL's model, not -Base), the workload (29,696-token sequences, 3,072 per turn, 6 turns, 64 x 16 per step, 100 steps), the recipe's DAPO-specific reward scaling / overlong shaping off, no validation, checkpoints every 20 steps. That means **full fine-tune** (the recipe; TRL is LoRA r32), lr 1e-6 constant after 10 warmup steps, FusedAdam with bf16 moments, no KL, TIS (ratio 2), clip 0.2/0.28, reward std normalization on (NeMo-RL default), synchronous colocated GRPO.

## Env glue

`launch/nemo-rl/t2sql/nemo_t2sql.py`: `Text2SQLEnv` (async Ray actor) + dataset + a driver modelled on `run_grpo_sliding_puzzle.py` that calls NeMo-RL's `setup` / `grpo_train`. Every episode helper is imported from `workloads/text2sql/t2sql_episode.py` (prompt ids, SQL tool + truncation, observation framing, reward incl. process penalties, gold cache, `SqlWork(16)`); t2sql_episode's TRL imports are stubbed so the module imports unchanged. One forced difference: NeMo-RL ends a rollout after any length-truncated turn, so a 3,072-token turn that still contains a tool call is scored as unanswered (status `length_cut` in episodes.jsonl).

## Run log (`date -u`, 2026-11-12/13)

1. **Install** (job 373576, hopper-cpu): `enroot import docker://nvcr.io#nvidia/nemo-rl:v0.8.0`, anonymous, 61 min, 82 GB squashfs at `stable-bench-src/images/nemo-rl-v0.8.0.sqsh`. The container's NeMo-RL is `a09162e1e` (2026-11-05), driver venv `/opt/nemo_rl_venv` (Python 3.13, torch 2.13.0+cu130, transformers 5.12.1), worker venvs prebuilt under `/opt/ray_venvs` (vLLM 0.29.0; Automodel from `/opt/nemo-rl/3rdparty/Automodel-workspace`).
2. **Our plumbing, not NeMo-RL:** with `--export=ALL` our login PATH/`UV_*` leak into the Pyxis container, so `uv run` picked the wrong environment (`No module named 'vllm'`); scripts call `/opt/nemo_rl_venv/bin/python` directly. Ray's AF_UNIX socket path limit (107 bytes) rules out `/scratch/<user>/<job>/...` as `RAY_TMPDIR`; it is `/tmp/n$SLURM_JOB_ID`. The Ray env actor needs the `trl_stub/` dir on PYTHONPATH (t2sql_episode imports TRL classes it doesn't use here).
3. **Glue bug (ours), job 373717, 23:19:** `calculate_rewards` passes the env only `role`/`content` of each message (`rollouts.py:841`), so `token_ids` aren't available; the env now re-derives each turn's token count by encoding the content (vLLM's decoded `text`, skip_special_tokens) and tracks the running total in metadata. Finish reason = length when the turn used its whole budget. CPU check 373831: tool turn continues, gold answer 0.8-0.9, no answer -1.1.
4. **Full fine-tune OOM at step 1 (nemo-rl-001), job 373838, 23:52-00:16.** Step-1 rollout fine: 1,024 episodes, reward 0.695, correct 0.793, 2.33 turns, 1.33 tool calls, mean 3.8k / max 15.2k tokens per episode (TRL step 1: reward 0.742). Logprob recompute fine; the first train step OOMed in `get_next_token_logprobs_from_logits -> log_softmax` (27.35 GiB request; 102 GiB allocated + 43.6 GiB reserved-unallocated on a 178 GiB GPU). Dynamic batching's `train_mb_tokens` defaults to 29,696 tokens, the full-vocab fp32 logits of such a micro-batch are ~29 GiB, and `logprob_chunk_size` does not reach that log_softmax on this path (no TP/CP: the plain-tensor branch at `model_utils.py:1683` has no chunking). The recipe was validated at 9,216 tokens on 4 nodes.
5. **Retry, built-in options only (coordinator):** `automodel_cfg.lora_cfg` with TRL's setup (r32 / alpha 64, `match_all_linear: false`, 12 wildcard targets = TRL's 310 modules: 10 x q/k/v/o, 30 x GDN in_proj_qkv/z/b/a + out_proj, 40 x shared expert gate/up/down; routed experts and router frozen; `use_triton: false`), lr 5e-5, betas (0.9, 0.95), 10 warmup steps then cosine to 5e-6 over 90 (TRL run 373132), and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `automodel_cfg.env_vars` (shipped in the ep16 Automodel recipe). Recipe settings kept otherwise (no KL, TIS 2, clip 0.2/0.28, std-normalized rewards, sync colocated, weight decay 0.1, FusedAdam). Deviations from the recipe are therefore: LoRA instead of full FT, lr/betas/schedule, expandable_segments, plus the earlier workload/1-node changes. Checkpoints are PEFT adapters; `evals.sh` folds them into the base with the streaming merger (`envs/prime-rl/merge_lora.py`) before the shared eval.
6. **LoRA retry fails at setup (nemo-rl-002), job 373868, 01:10.** CPU check 373854 passed (config validates; the 12 wildcard targets hit exactly TRL's 310 modules). On the GPUs every Automodel policy worker died in `parallelize_model -> apply_fsdp -> fully_shard_by_dtype`: "FSDP could not isolate parameters with a distinct dtype from siblings in the same module: linear_attn.out_proj.weight, linear_attn.in_proj_qkv.weight, ..., mlp.shared_experts.*". No shipped option changes that; NeMo-RL ships LoRA recipes only for Nemotron-3-Nano.
7. **Last try (coordinator): NeMo-RL's own 2-node recipe**, `grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml` as shipped (full FT, EP16, TE attention/linear/rms_norm, hybridep, activation checkpointing, expandable_segments, FusedAdam bf16 moments, colocated vLLM TP8 eager; grpo_math_1B loss/optimizer defaults: KL 0.01 k3, clip 0.2, lr 5e-6 with 50 warmup steps, leave-one-out std-normalized GRPO), with only our model, workload (29,696 / 3,072 / 6 turns / 64 x 16 / 100 steps), no validation and checkpoints every 20 steps (`launch/nemo-rl/t2sql/t2sql-2n.yaml`). The LoRA/lr/schedule changes of item 5 are not in it. Launcher `launch/nemo-rl/t2sql-2n.sbatch` runs NeMo-RL's own `ray.sub` (v0.8.0 tree, unchanged) across the two-node allocation; the driver attaches to that Ray cluster (no private RAY_TMPDIR) and passes its environment to every actor (init_ray). If step 1 also OOMs, stop: ledger row, no more retries.

## W1 (math, Qwen3.5-2B, W1-S shared arm)

Files: `configs/nemo-rl/w1-S.yaml`, `launch/nemo-rl/w1/` (`w1.sbatch`, `inner.sh`, `nemo_w1.py`, `nemo_w1_env.py`, `w1_prompt.txt`, `watch.py`, `check.sbatch` + `check_w1.py`). Same NGC v0.8.0 container and plumbing as the t2sql runs (container venv first on PATH, short `TMPDIR`/`RAY_TMPDIR`, env module on PYTHONPATH).

Recipe source: NeMo-RL's only shipped Qwen3.5 dense GRPO recipe, `examples/configs/recipes/llm/grpo-qwen3.5-9b-1n8g-megatron.yaml` (Megatron backend, TP4 + sequence parallel, activation checkpointing, `defer_fp32_logits`, no packing, micro-batch 1; inherits `grpo_math_1B.yaml`), with async GRPO set up as `docs/guides/async-grpo.md` says (non-colocated vLLM, async engine, IS correction on). Driver = NeMo-RL's own `examples/run_grpo.py`, run unchanged through `runpy` after registering one env.

Harness glue: `nemo_w1_env.W1MathEnv`, a Ray actor shaped like NeMo-RL's `MathEnvironment` that scores the assistant text with `workloads/math/reward_safe.py::compute_score` instead of NeMo-RL's own math_verify wrapper. Data goes through NeMo-RL's built-in `ResponseDataset` (our `train.parquet`, `problem`/`answer`) and `math_hf_data_processor` with a prompt file that formats to `reward.PROMPT`.

Knob mapping (shared value -> NeMo-RL):
- 32 prompts x 8: `grpo.num_prompts_per_step=32`, `num_generations_per_prompt=8`, `policy.train_global_batch_size=256` (one optimizer step per batch), micro-batch 1.
- lengths: `max_total_sequence_length=18432`, `generation.max_new_tokens=16384`, vLLM `max_model_len=18432`, `data.max_input_seq_length=2048` (longest prompt 1,517 tokens, none masked).
- thinking off: `policy.tokenizer.chat_template_kwargs.enable_thinking=false`; NeMo-RL tokenizes the rendered prompt and sends token ids to vLLM.
- sampling T=1, top_p=1, no top_k.
- optimizer: Megatron Adam (distributed, precision-aware, fp32 master params, bf16 compute), lr 1e-6 constant (`min_lr=1e-6`, `lr_warmup_iters=0`), weight decay 0, betas (0.9, 0.999), grad clip 1.0. Recipe default was lr 5e-6 with 13 warmup iters, wd 0.01.
- advantage: `normalize_rewards=false`, `use_leave_one_out_baseline=false` = reward minus group mean (recipe default: leave-one-out + std normalization).
- loss: `ratio_clip_min/max=0.2/0.28`, `token_level_loss=true` (global token mean), `reference_policy_kl_penalty=0` (recipe 0.01), no entropy term. Zero-advantage groups stay in the batch and their tokens count in the token-mean denominator (library default, no dynamic sampling).
- clip reference (rev. 5 gap): NeMo-RL clips the ratio against `prev_logprobs`, the trainer's own recomputed logprobs before the update, not the sampler's. With one optimizer step per batch that ratio is 1, so the clip never bites; the off-policy gap (sampler vs trainer, incl. 1-step staleness) is carried by the async-required IS correction `use_importance_sampling_correction=true`, an unclipped per-token weight pi_old/pi_vllm (no TIS cap: `truncated_importance_sampling_type=null`, the default). One-sided PPO clip, recorded as such.
- off-policy: `async_grpo.enabled=true`, `max_trajectory_age_steps=1` (default and the guide's recommended start), `in_flight_weight_updates=false` (default). Weights refit to vLLM after every step.
- batch composition: async GRPO trains on the next 32 complete groups from the replay buffer that are within the age limit; no refill or oversampling (dynamic sampling off, and async refuses it anyway).
- hardware: 4 vLLM GPUs (`colocated.enabled=false`, `resources.gpus_per_node=4`, vLLM TP1 = 4 engines, CUDA graphs on, `gpu_memory_utilization=0.85`) + 4 Megatron GPUs. TP4 is the recipe's value kept as is, so the 2B trainer runs DP1 x TP4 (a smaller TP was not tried; it is what the validated recipe uses).
- steps/checkpoints: `max_num_steps=200`, `checkpointing.save_period=50`, all kept, with optimizer (also the auto-resume path: NeMo-RL resumes from the run's checkpoint dir, and the sbatch is `--requeue` with a job-id run dir). No validation.
- seed 42 (`grpo.seed`); the dataloader reshuffles the already-shuffled file, so prompt order differs from other libraries.
- training attention: Megatron-Core via TE for full-attention layers, MCore GDN kernels for linear attention (recipe defaults).

Deviations from the recipe: model (2B post-trained, not 9B-Base), the shared workload above, non-colocated async GRPO, vLLM `gpu_memory_utilization` 0.7 -> 0.85 (0.7 was for colocation), no wandb.

CPU check (job 397929, hopper-cpu, 2026-11-16): config loads and validates; all 17,255 rendered prompts are byte-identical to `apply_chat_template(PROMPT, enable_thinking=False)` and end with the empty think block, ground truths match; the env gives the same rewards as `reward.compute_score` on six scripted completions (correct, wrong, two boxes, unclosed box, no box, trailing text). Container has math-verify 0.9.0.

## W2 (TermGrade, terminus-2, Qwen3.5-9B)

Route: NeMo Gym's `harbor_agent` (Gym `1f8afc71`, the container's pin), which is how NeMo-RL trains agents that own their loop. NeMo-RL's vLLM exposes its OpenAI server (`expose_http_server`), Gym's `Terminus2NemoGym` (terminus-2 with a Gym LLM client) calls it, NeMo-RL's on-policy token correction keeps each episode one token sequence, and the Harbor verifier's score is the reward. Entry point: NeMo-RL's own `examples/nemo_gym/run_grpo_nemo_gym.py`. The EnvironmentInterface route was not used: NeMo-RL owns the turn loop there, so terminus-2 could not run unchanged.

Config `configs/nemo-rl/w2-M.yaml`, inheriting `grpo-qwen3.5-9b-1n8g-megatron.yaml` (NeMo-RL's only Qwen3.5 dense recipe: Megatron TP4 + SP, activation checkpointing, `defer_fp32_logits`, no packing). Mapped to SHARED_W2:
- model `Qwen/Qwen3.5-9B` (recipe: -Base), thinking off (`chat_template_kwargs.enable_thinking=false` on NeMo-RL's tokenizer and on Gym's vllm_model server, `uses_reasoning_parser=false`);
- 8 tasks x 8 episodes, `train_global_batch_size=64` (one optimizer step per batch), 60 steps, seed 42, no dynamic sampling;
- advantage: `normalize_rewards=false`, `use_leave_one_out_baseline=false` (reward minus group mean);
- loss: KL 0 (reference logprobs skipped), clip 0.2 / 0.28, `token_level_loss=true`, `use_importance_sampling_correction=true` (required by async GRPO). NeMo-RL clips against `prev_logprobs` (the trainer's recomputed logprobs at the current weights), and the sampler gap is corrected by an untruncated token IS weight, so with one optimizer step per batch the clip band is not against the sampler (SHARED rev. 5). Same as the W1 arm;
- optimizer: Megatron Adam, lr 1e-6 constant (min_lr 1e-6, 0 warmup), weight decay 0, betas (0.9, 0.999), clip 1.0, fp32 master (`params_dtype=float32`), distributed optimizer;
- async: `async_grpo.enabled`, `max_trajectory_age_steps=1` (the guide's default and suggested start), no in-flight weight updates; non-colocated 4 vLLM GPUs (4 engines, TP1; recipe: one colocated TP4 engine) + 4 Megatron GPUs (TP4, DP1);
- lengths: `max_total_sequence_length` = vLLM `max_model_len` = 65,536, `max_new_tokens` 8192, and terminus-2 `model_info` 65,536 in / 8,192 out;
- sampling: T 1.0, top_p 1.0, no top_k. Set in each data row's `responses_create_params` (Gym's LLM sends those, not terminus-2's own 0.7 default); the CPU check saw `temperature=1.0, top_p=1.0, max_tokens=8192` on the wire;
- harness: `max_turns=30`, `enable_summarize=false`, `proactive_summarization_threshold=0`, `collect_rollout_details=true`, agent timeout 600 s, verifier 300 s (overrides; same values as the tasks), concurrency 64 (Gym's semaphore = live boxes), `harbor_ray_task_num_cpus=0.5`;
- data: 28 train tasks x 18 = 504 rows (`launch/nemo-rl/w2/prepare_tasks.py`), shuffled by NeMo-RL's dataloader (task order differs from TRL);
- checkpoints every 20 steps (Megatron, with optimizer, all kept), converted to HF for the 12-sample local eval;
- failed episodes: Gym returns reward 0 with an empty output for a Harbor exception (README "Known failure cases"); NeMo-RL has no drop for those (Gym's harbor agent sets no `mask_sample`), so they stay in their group at 0. Deviation from SHARED_W2 (unscored episodes should be excluded); counted from the archived Harbor job dirs.

Harness glue, no NeMo-RL or Gym source change:
- task layout (`prepare_tasks.py`): Gym pins Harbor `9dddd797` (2026-01-26, reports 0.1.42), which has no `[environment.healthcheck]`; Gym's SingularityEnvironment runs `environment/files/setup.sh` instead. Each shared local-box task is copied to `stable-bench-src/termgrade/harbor-nemogym/` with the healthcheck removed and `environment/files/setup.sh` = put the task's setup.sh/prepare.sh in /app and run prepare.sh (exactly what the healthcheck ran). Same .sif image, tests and timeouts. Gym's in-box exec server needs uvicorn/fastapi, which the image lacks; Gym's own `write_min_setup_sh.py` pip-installs them into the box python. Instead they are copied to /opt/hbdeps with a `python3` wrapper at /opt/conda/bin (a path Gym's bootstrap probes), so the task's python and site-packages are unchanged;
- `UV_OVERRIDE=launch/nemo-rl/w2/uv-override.txt` (pathspec): Gym's pyproject drops pathspec from every server venv (copyleft exclusion, "servers that need it re-enable it via their own overrides.txt"); harbor_agent ships no overrides.txt, and Harbor imports pathspec via dirhash/scantree, so as shipped the Harbor server cannot import Harbor. The env var re-enables it without touching the tree;
- `APPTAINER_IGNORE_FAKEROOT_COMMAND=1`: Gym always runs boxes with `--fakeroot`; the container's Apptainer 1.4.5 injects its libfakeroot (needs GLIBC 2.38) into the box, whose libc is older, so every box died at start (check 397973). Apptainer's own flag keeps the root-mapped namespace without the fakeroot library;
- `APPTAINER_BIND=/etc/resolv.conf`, Apptainer cache/tmp on /scratch, as for TRL's local boxes.
- Box memory: Gym's SingularityEnvironment runs a userspace watchdog that kills a box at 95% of the task's `memory_mb` (Harbor default 2048 MB; TermGrade tasks set none). TRL's local boxes had no cap. Kept as shipped; counted in the results.

CPU check (hopper-cpu, 2026-11-16): config loads and validates with NeMo-RL's loader (397939). Harbor venv built from harbor_agent/requirements.txt with Python 3.13 (Gym needs >= 3.13.14). On 2 train tasks through Gym's own `run_harbor_job` (397981): oracle 1.0 / 1.0, nop 0.0 / 0.0, scripted terminus-2 episode (stub server answering like NeMo-RL's HTTP server) reaches the verifier (0.0) and Gym's conversion yields output items carrying the stub's token ids. Earlier attempts: 397937 (Python 3.12, unresolvable), 397949 (pathspec), 397973 (fakeroot GLIBC).

Run log (`date -u`):
- 2026-11-16T12:48:50Z: GPU run 398019 started on ip-10-53-7-197 (run dir stable-bench-runs/nemo-rl/nemo-rl-w2-M-local-398019), 30 h limit.
- 2026-11-16T12:51:07Z: 398019 died in data setup (ours): the base config's data.train.dataset_name (OpenMathInstruct-2) overrode default.dataset_name; set train.dataset_name=NemoGymDataset. Resubmitted.
- 2026-11-16T12:53:40Z: 398023 refused by the Gym runner: grpo.max_val_samples must be null (base config sets 256); set max_val_samples/val_batch_size null. Resubmitted.
- 2026-11-16T13:01:40Z: 398026 got through setup (vLLM, Megatron, Gym servers up) but every box died at start on the GPU node: Apptainer inside the Pyxis container cannot mount a fresh /proc for Gym's --containall --pid boxes ('can't mount proc filesystem to /proc: operation not permitted'); on hopper-cpu the same boxes start. NeMo-RL then aborted on the first empty Gym result. Probing flags (box_probe.sbatch).
- 2026-11-16T13:04:02Z: probe 398036 (GPU node, in the container): Gym's flags fail; the same flags plus --no-mount proc and -B /proc start the box with its own pid namespace. Set Gym's built-in harbor_environment_kwargs.singularity_no_mount=home,tmp,bind-paths,proc and APPTAINER_BIND=/etc/resolv.conf,/proc. The box sees the node's /proc listing (pids belong to the node namespace, so kill by pid from inside reaches nothing outside the box). Resubmitted.
- 2026-11-16T13:20:47Z: 398039 step 1 at 13:19 (16 min after start): reward 0.453, token_mult_prob_error 1.006, gen_kl_error 0.0002, step time 351 s.

### W2 failed attempts (all, in order)

| job | stage | error line | root cause | attribution |
|---|---|---|---|---|
| 397930 | container probe (hopper-cpu) | none (exploratory) | found: Gym checkout in container, Apptainer 1.4.5 in container, box image has no uvicorn | - |
| 397937 | CPU check | `No solution found ... nemo-gym[dev] ... Python>=3.13.14` and `TypeError: 'GRPOConfig' object is not subscriptable` | our check built the Harbor venv with Python 3.12 (Gym needs >= 3.13.14); our print code indexed a pydantic config (the config itself validated) | ours |
| 397939 | CPU check | `ModuleNotFoundError: No module named 'pathspec'` (all 6 episodes) | Gym's pyproject drops pathspec from every server venv; harbor_agent ships no overrides.txt; Harbor imports it via dirhash -> scantree | NeMo-RL as shipped (nemo-rl-003) |
| 397949 | CPU check | same pathspec error | adding `pathspec` to the install line does not help: Gym's override list still drops it; needs `UV_OVERRIDE` | NeMo-RL as shipped (same row) |
| 397973 | CPU check | `/usr/bin/bash: ... version 'GLIBC_2.38' not found (required by /.singularity.d/libs/libfakeroot.so)`, `Server process died` | Gym always runs boxes with `--fakeroot`; the container's Apptainer injects the container's libfakeroot into the box, whose libc is older | NeMo-RL as shipped with an older-libc task image (nemo-rl-004) |
| 398019 | GPU | `ConnectionError: Couldn't reach 'nvidia/OpenMathInstruct-2' on the Hub (OfflineModeIsEnabled)` | our config inherits the math recipe, whose `data.train.dataset_name` wins over `data.default.dataset_name: NemoGymDataset` | our config; a trap for anyone stacking a Gym config on a shipped recipe (nemo-rl-006) |
| 398023 | GPU | `ValueError: A non-null grpo.max_val_samples parameter is not supported.` | the Gym runner requires `max_val_samples: null`; the inherited base sets 256 | our config; same trap (nemo-rl-006) |
| 398026 | GPU | `FATAL: container creation failed: mount hook function failure: mount proc->/proc error: ... operation not permitted`, then `ValueError: NeMo Gym returned a result with no generation data` and `RuntimeError: Rollout batch failed to buffer prompt groups [0, 1, 2, 3, 4, 5, 6, 7]` | Gym's boxes use `--containall --pid`, which mounts a fresh /proc; inside the Pyxis container on the GPU nodes /proc has masked submounts, so the kernel refuses a new proc mount (the same boxes start on hopper-cpu). Every box failed, so every Harbor job returned reward 0 with an empty output; NeMo-RL's collector refuses a result with no generation data, so all 8 prompt groups of the first batch failed to buffer, and with `max_generation_failures: 0` (default) the run aborted | environment (Pyxis on GPU nodes) for the box start (nemo-rl-005); NeMo-RL as shipped for "one failed episode aborts the run" (nemo-rl-007) |

What changed in 398039: Gym's built-in `harbor_environment_kwargs.singularity_no_mount` gained `proc` and `APPTAINER_BIND` gained `/proc` (box_probe 398036 on the GPU node: Gym's flags fail, the same flags with `--no-mount proc -B /proc` start the box). Nothing else changed from 398026. Step 1 followed 16 min after start.

Open risk from nemo-rl-007: any single later Harbor failure (box start, setup timeout, memory watchdog without trajectory) returns an empty output and aborts the run the same way; the in-job watcher sees it as the driver exiting.

### W1 run log and failed attempts

Training run: job 397935 (waiter 397934), started 2026-11-16 11:17. Step 1: reward 0.477, gen_kl 3.4e-4, token_mult_prob_error 1.010, 271 s (includes vLLM warmup); steady state ~107 s per step (both async halves in use); step 26 reward 0.625, step 139 reward 0.672. Checkpoints `ckpt/step_{50,100,...}` are Megatron dist checkpoints with optimizer (~iter dir `iter_0000000`).

No training failures. Failed attempts, all in the post-run export path:

| job | error line | root cause | attribution |
|---|---|---|---|
| (sbatch refused, no id) | `sbatch: error: CPU count per node can not be satisfied` | convert job asked hopper-cpu for 16 CPUs + 96 GB, more than a spot node offers | ours (job shape) |
| 398138 convert, hopper-cpu | `OSError: libcuda.so.1: cannot open shared object file` | NeMo-RL's `examples/converters/convert_megatron_to_hf.py` imports Megatron-Core, which imports Transformer Engine, which dlopens libcuda at import: the converter can't run on a CPU-only node | environment/NeMo-RL as shipped (documented as "requires mcore", not that it needs a GPU node); config trap, not a NeMo-RL bug |
| 398168 convert, 1 B200 | `AssertionError: 36 shape/dtype mismatches ... linear_attn.A_log ((16,), 'BF16') vs ((16,), 'F32')` | NeMo-RL's Megatron->HF export writes Qwen3.5's GDN `A_log` and `linear_attn.norm.weight` (kept fp32 in the base checkpoint) as bf16. Our completion helper (`t2sql/complete_ckpt.py`) required exact dtype equality | ours (helper too strict); the export's bf16 downcast of fp32-by-design GDN params is noted as a NeMo-RL behaviour (eval loads it fine; values are the trained ones rounded to bf16) |

Fix: `convert.sbatch` runs on one B200 GPU through b200-queue, with the policy worker's Megatron venv (`/opt/ray_venvs/*MegatronPolicyWorker`); `complete_ckpt.py` takes `ALLOW_DTYPE=1` (shapes must still match; the count goes into complete.json as `dtype_differs`). Steps 50 and 100 then converted (398172): 617 exported tensors (vision tower included), 15 MTP tensors filled from the base.
- 2026-11-16T16:38:26Z: HF export: NeMo-RL's convert_megatron_to_hf.py needs the Megatron worker venv and a GPU (libcuda); it writes the language model and vision tower (760 tensors, all bf16, so the 48 fp32 GDN A_log/norm tensors are bf16), MTP (15) taken from the base by complete_ckpt.py (ALLOW_DTYPE=1). checkpoint-20 done; x12 eval queued (TAG x12-nemo9b-20).
- 2026-11-16T17:38:43Z: 398039 aborted after step 34 (16:59): from 16:58 every new box failed with 'script: failed to create pseudo-terminal: No space left on device' -> 'Failed to start tmux session' (1,523 of 1,544 trials in 10 min; before that 6 isolated failures in 3.7 h, 2,617 trials). The first empty Gym result failed all 8 prompt groups (nemo-rl-007) and the run exited 1. Steps 1-34: reward 0.36-0.61, prob_err ~1.006, gen_kl ~2.4e-4, 228-533 s/step. Resume from step_20 (RESUME_DIR), save_period 20 -> 10, PTY counts logged to pty.log.

### W1 result

Run 397935 finished 200 steps with exit 0 at 17:37 (6 h 24 min of training, ~107 s per step after step 2; watcher never fired; gen_kl stayed 3-10e-4, token_mult_prob_error 1.010-1.022). Shared eval (`workloads/math/eval.sbatch`, `results/raw/nemo-rl-w1-S-shared-397935/eval.jsonl`), AIME24 avg@16 / AIME25 avg@16 / MATH-500:

| step | 50 | 100 | 150 | 200 |
|---|---|---|---|---|
| nemo-rl-w1-S-shared-397935 | 0.292 / 0.246 / 0.868 | 0.327 / 0.275 / 0.872 | 0.354 / 0.296 / 0.876 | 0.317 / 0.288 / 0.890 |

Step 200 is at the bottom of the W1-S shared band (TRL .388/.304/.904, prime-rl .352/.302/.900, verl .350/.300/.892, OpenRLHF .404/.365/.908) and above base (.212/.171/.776).
- 2026-11-16T19:21:36Z: resume 398814 (17:40-19:12) trained from step 20 and saved step_30, then OUR watcher killed it (WATCH_KILL 'no new step for 90 min (last 34)'): resumed steps reuse step numbers already logged by 398039, so its last-step marker never moved. Bug ours (watch.py now tracks new train/loss events). pty.log confirmed the leak: tmux processes 830-894 with 51-64 live boxes, PTYs 1,786/4,096 after ~1.5 h. Added a tmux reaper (>30 min old) in inner.sh. Resuming from step_30.
- 2026-11-16T22:39:08Z: resume 399748 (19:23-22:20) trained steps 31-56, saved step_40/step_50, then aborted the same way (one empty Gym result -> all 8 prompt groups fail, nemo-rl-007); PTYs 3,071/4,096 with the reaper (tmux 320-340). Resuming from step_50.
- 2026-11-17T00:10Z: resume 400651 (from step_50) trained steps 51-60 and exited 0. Final steps 58-60: reward 0.66 / 0.73 / 0.72, prob_err ~1.007, gen_kl ~2.6e-4, 377-508 s/step. Training time over four jobs: 398039 13:05-16:59, 398814 17:40-19:12 (steps 21-30, wasted by our watcher), 399748 19:23-22:20, 400651 23:2x-00:09.

More failed attempts (continuing the table):

| job | stage | error line | root cause | attribution |
|---|---|---|---|---|
| 398039 | GPU, after step 34 | `script: failed to create pseudo-terminal: No space left on device` -> `Failed to start tmux session` -> `Rollout batch failed to buffer prompt groups [0..7]` | Gym's Singularity boxes leave tmux servers holding PTYs; PTYs run out after ~2,600 episodes, then one empty result aborts the run | NeMo-RL as shipped (nemo-rl-009, via nemo-rl-007), possibly worsened by the /proc workaround |
| 398814 | GPU resume | `WATCH_KILL no new step for 90 min (last 34)` | our watcher keyed progress on the highest step number, which a resume re-logs below the old maximum | ours (watch.py fixed) |
| 399748 | GPU resume, after step 56 | same PTY error, same abort | same as 398039; tmux reaper slowed it, did not stop it | NeMo-RL as shipped (nemo-rl-009) |
| 398136, 398198, 398480/398482 | HF export | `No module named 'megatron.bridge'`; `libcuda.so.1: cannot open shared object file`; 48 shape/dtype mismatches in complete_ckpt | converter needs the Megatron worker venv and a GPU; export writes fp32 GDN A_log/norm as bf16 (complete_ckpt ALLOW_DTYPE=1); 398480 was a quoting bug in our export script, cancelled by me | ours |

### W2 results (12-sample held-out eval, local boxes, same setup as results/tables/w2_heldout_x12.md)

| step | NeMo-RL | scored | TRL | prime-rl | verl | SkyRL dppo |
|---|---|---|---|---|---|---|
| 20 | 0.395 | 167/168 (eval 398489) | .386 | .337 | .369 | .417 |
| 40 | 0.353 | 167/168 (eval 400655) | .417 | .405 | .370 | .417 |
| 60 | 0.416 | 166/168 (eval 400940) | .443 | .411 | .429 | .491 |

Base 9B 0.301. Checkpoints: stable-bench-runs/nemo-rl/nemo-rl-w2-M-local-398039/hf/checkpoint-{20,40,60}. Training reward on the train tasks went from ~0.45 (steps 1-20) to 0.66-0.73 (steps 58-60).
