# verl: mapping of SHARED_W1 (rev. 2) onto verl v0.9.1

Config: `configs/verl/w1-S.yaml` (Hydra, extends verl's `ppo_trainer`). Entry point: `python -m verl.trainer.main_ppo` with `trainer.use_v1=True`, `trainer.v1.trainer_mode=separate_async`. Code references are to `/fsx/amine_dirhoussi/stable-bench-src/verl` at `v0.9.1`.

| SHARED_W1 knob | verl knob | value | semantics / gap |
|---|---|---|---|
| model | `actor_rollout_ref.model.path` | `Qwen/Qwen3.5-2B` | Loaded as `Qwen3_5ForConditionalGeneration` (VLM class), with verl's Qwen3.5 monkey patch (`verl/models/transformers/qwen3_5.py`). Text-only data. `actor.freeze_vision_tower` is never read by the FSDP path; the vision tower gets no gradient from text-only data (grad `None`, AdamW skips it), so it's frozen in practice. |
| compute / master dtype | `actor.fsdp_config.mixed_precision` (defaults) | bf16 compute, fp32 master | FSDP2 keeps fp32 params and optimizer state, `param_dtype=bf16`, `reduce_dtype=fp32`. vLLM runs bf16. |
| prompts per optimizer step | `actor.ppo_mini_batch_size` | 32 | Counts prompts. The V1 trainer multiplies it by `rollout.n` (`verl/trainer/ppo/v1/trainer_base.py:1845`), so one optimizer step sees 32 x 8 = 256 trajectories. |
| batch composition | V1 `ReplayBufferAsync`, `algorithm.filter_groups.enable=false` (default) | next 32 complete groups | A step trains on the first 32 prompt groups whose 8 rollouts have all finished. Zero-advantage groups (all 0 or all 1) are not filtered, refilled or oversampled. A group is evicted and replaced only if it exceeds `max_off_policy_threshold` (8 versions, `drop`) or its rollout fails; with a weight sync every step and one warmup batch, staleness stays at 0-1, so the drop rule shouldn't fire. Zero-advantage tokens **do count in the loss denominator**: their advantage is 0 but their `response_mask` tokens are in `batch_num_tokens` for `token-mean`. |
| (verl-only) train batch | `data.train_batch_size` | 32 | One "global step" in V1 `separate_async`. Must equal `parameter_sync_step x ppo_mini_batch_size` (asserted in `trainer_separate_async.py:59`). With `parameter_sync_step=1`, one global step = one optimizer step = one weight sync, so verl's step counter is the shared optimizer-step counter. |
| generations per prompt | `rollout.n` | 8 | |
| max prompt length | `data.max_prompt_length` | 2048 | Prompt tokens only, after the chat template. Never truncated: `truncation=error` raises on an overlong prompt, and `filter_overlong_prompts=true` drops any prompt over 2048 at load time (the dataloader logs the filtered count). Checked: the longest rendered prompt is 1517 tokens, 9 are over 1024, none over 2048 (`envs/verl/render_check.py`, cpu job 364378). |
| max completion length | `data.max_response_length` (-> `rollout.response_length`), `rollout.max_model_len` | 16384; vLLM `max_model_len` 18432 | Completion tokens only; separate from the prompt budget (rev. 2). `ppo_max_token_len_per_gpu` / `log_prob_max_token_len_per_gpu` = 36864 so a full-length 18,432-token sequence always fits one micro-batch (dynamic batching only; no effect on the math). |
| sampling | `rollout.temperature`, `top_p`, `top_k` | 1.0, 1.0, -1 | -1 disables top-k in vLLM. No min_p is set (vLLM default 0). |
| optimizer | `actor.optim.*` | AdamW (torch), lr 1e-6, `lr_scheduler_type=constant`, `lr_warmup_steps=0`, `weight_decay=0.0`, betas (0.9, 0.999) | verl's default weight decay is 0.01; we set 0.0. |
| grad clip | `actor.grad_clip` / `actor.optim.clip_grad` | 1.0 | Global norm over all params. |
| optimizer steps per batch | `actor.ppo_epochs`, mini-batch split | 1 epoch, 1 mini-batch per global step | `use_dynamic_bsz=true` with `ppo_max_token_len_per_gpu=32768` only splits the step into micro-batches for gradient accumulation; still one optimizer step. |
| advantage | `algorithm.adv_estimator=grpo`, `norm_adv_by_std_in_grpo=false` | group mean only | Reward minus the mean over the 8 rollouts of the prompt; no std division. |
| clipping | `actor.clip_ratio_low`, `clip_ratio_high` | 0.2 / 0.28 | verl also applies dual-clip with `clip_ratio_c=3.0` (default) to negative-advantage tokens: the loss is capped at 3 x |A|. Not in the shared config; kept at the library default and noted here. |
| IS correction / off-policy ratio | `algorithm.rollout_correction.bypass_mode=true`, `rollout_is=null` | PPO ratio = π_θ / π_vLLM | Old logprobs are not recomputed by the trainer; vLLM's sampling logprobs are the PPO anchor ("bypass PPO-clip", `docs/algo/rollout_corr.md`). This is what the V1 e2e tests and fully-async docs use for async runs. No extra truncated-IS weight. |
| loss aggregation | `actor.loss_agg_mode=token-mean` | token mean over the optimizer step | `agg_loss` divides the masked token sum by the global token count of the mini-batch across all DP ranks (`batch_num_tokens`, `verl/trainer/ppo/core_algos.py:1170-1175`), i.e. every non-pad completion token in the 256 trajectories gets equal weight. |
| KL / entropy | `actor.use_kl_loss=false`, `algorithm.use_kl_in_reward=false`, `actor.entropy_coeff=0` | none | No reference model is created. |
| reward | `reward.custom_reward_function.path=configs/verl/w1_reward.py` | shared `workloads/math/reward.py` | Called through verl's default `naive` reward manager on the decoded completion (special tokens skipped). |
| prompt | `data/w1/train.parquet` built by `configs/verl/w1_data.py` | single user message = shared `PROMPT` | Rendered by verl with the tokenizer's chat template at its defaults (`apply_chat_template_kwargs` empty). |
| chat template kwargs | `data.apply_chat_template_kwargs` | `{enable_thinking: false}` (rev. 2) | The same dict reaches both `tokenizer.apply_chat_template` calls: the dataset's length filter (`verl/utils/dataset/rl_dataset.py:120`) and the agent loop that renders the prompt sent to vLLM (`verl/experimental/agent_loop/agent_loop.py:242-251`). |
| data order | `data.shuffle=false` | file order | The shared parquet is already deduplicated and shuffled with seed 42. |
| off-policy | `trainer.v1.sampler.max_off_policy_threshold=8`, `max_off_policy_strategy=drop`, `separate_async.num_warmup_batches=1` | verl defaults | Max 8 model versions between a prompt group's first generated token and its training step; older groups are dropped and refilled. Partial rollout is always on in V1 async: generations interrupted by a weight sync resume on the new weights, so one trajectory can span versions. |
| weight sync frequency | `trainer.v1.separate_async.parameter_sync_step` | 1 | verl's default is 4 (sync every 4 optimizer steps). We use 1 so each 32-prompt optimizer step is one global step and checkpoints land on exact optimizer steps; this is the more on-policy setting. Recorded as a deviation. |
| total steps | `trainer.total_training_steps` | 200 | Global steps = optimizer steps here. |
| checkpoints | `trainer.save_freq=50`, `actor.checkpoint.save_contents=[model, optimizer, extra, hf_model]` | steps 50/100/150/200 | HF weights land in `<run>/ckpt/global_step_<N>/actor/huggingface`. |
| GPU split | `rollout.n_gpus_per_node=4`, `trainer.n_gpus_per_node=4`, `hybrid_engine=false` | 4 vLLM + 4 FSDP2 | 4 vLLM replicas at TP=1. With `hybrid_engine=false` no vLLM engine lives on the trainer GPUs. Weight sync is NCCL (`checkpoint_engine.backend=nccl`). |
| inference engine | `rollout.name=vllm`, `mode=async` | vLLM 0.24.0 | Pinned by verl's `uv.lock`. `gpu_memory_utilization=0.85`, `max_num_seqs=256` per replica (default). |
| train backend | `actor.strategy=fsdp2` | FSDP2 | Sequence packing on (`use_remove_padding=true`); Qwen3.5 linear-attention layers get `cu_seqlens` / `seq_idx`, so packing doesn't cross samples (see `envs/verl/NOTES.md`). Gradient checkpointing on. |
| seed | actor / rollout `seed` | 42 (verl default) | |
| eval | `trainer.test_freq=-1` | off | Eval runs with the shared `workloads/math/eval.py` on the saved HF checkpoints. |

## Recommended arm (`configs/verl/w1-S-recommended.yaml`)

Same model, data, prompt, reward, hardware split (4 vLLM + 4 FSDP2, `hybrid_engine=false`), 32 prompts x 8 rollouts per optimizer step, 200 optimizer steps, optimizer and sampling as the shared arm. Only these knobs move to verl's own defaults:

| knob | shared arm | recommended arm | source |
|---|---|---|---|
| advantage | group mean only (`norm_adv_by_std_in_grpo=false`) | (r - mean) / std over the group (`true`) | `norm_adv_by_std_in_grpo` default; docs/algo/grpo.md (turning it off is the Dr. GRPO variant) |
| clip | 0.2 / 0.28 (clip-higher) | 0.2 / 0.2 | actor config default; dual-clip `clip_ratio_c=3.0` in both arms |
| KL | none | KL loss, `kl_loss_coef=0.001`, `low_var_kl`, reference model on the trainer GPUs | docs/algo/grpo.md: "Please set it to True for GRPO"; every verl GRPO example uses `low_var_kl` |
| off-policy anchor | bypass: PPO ratio vs vLLM logprobs | decoupled PPO: the trainer recomputes old logprobs and holds pi_old fixed for the 4 mini-batches of a sync cycle | `algorithm.rollout_correction.bypass_mode` default `false`; docs/advance/v1_async_trainer.md |
| weight sync / batching | `parameter_sync_step=1`, `train_batch_size=32` | `parameter_sync_step=4`, `train_batch_size=128` (4 optimizer steps of 32 prompts, then one weight sync) | V1 separate_async default |
| staleness | max 8 model versions, `drop` | same (already the default) | |
| zero-advantage groups | kept, tokens in the denominator | same: `filter_groups.enable=false` is the default (DAPO dynamic sampling is opt-in) | |
| loss aggregation | token-mean | same (default) | |
| step counting | 200 global steps = 200 optimizer steps; checkpoints at 50/100/150/200 | 50 global steps = 200 optimizer steps; checkpoints at global 12/24/36/48/50 = optimizer steps 48/96/144/192/200 (verl saves only on global steps) | |
| logging | one metrics row per optimizer step | one row per global step (4 optimizer steps; mini-batch metrics averaged). `collect/verl.py` writes `step` = optimizer step at the end of the global step | |

## M (`configs/verl/w1-M.yaml`, `configs/verl/w1-M-recommended.yaml`)

Both M configs are the S configs with only the model swapped. Same 4 vLLM + 4 FSDP2 split, same batching (32 prompts x 8 per optimizer step, 256 rollouts), same micro-batch token budgets, so tokens per optimizer step equal S. Launch: `SIZE=M` (shared) or `SIZE=M CONFIG=w1-M-recommended` (recommended).

| knob | S value | M value | why |
|---|---|---|---|
| `actor_rollout_ref.model.path` | `Qwen/Qwen3.5-2B` | `Qwen/Qwen3.5-9B` | the M model (`Qwen3_5ForConditionalGeneration`, 32 layers, hidden 4096, untied embeddings, 19 GB bf16) |
| trainer sharding | FSDP2 over 4 GPUs, no offload | same | 9B fp32 params + grads + AdamW = ~145 GB, ~36 GB per GPU when sharded 4 ways. S peaked at 31.6 GB allocated with ~8 GB of state, so M's estimate is ~36 GB state + ~30-40 GB activations/logits, well under 180 GB. No CPU offload, no Ulysses SP needed |
| activation checkpointing | on | on (unchanged) | |
| `ppo_max_token_len_per_gpu` / `log_prob_max_token_len_per_gpu` | 36864 | 36864 (unchanged) | fits (see above); keeping it leaves micro-batching identical to S |
| vLLM | TP=1 x 4 replicas, `gpu_memory_utilization=0.85` | same | 19 GB of weights leaves ~130 GB of KV cache per replica |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | `expandable_segments:True` (set by `launch/verl/w1.sbatch` when `SIZE=M`, before `ray start`) | S trainer GPUs reserved 170 GB for 32 GB allocated; at 9B that fragmentation could OOM. vLLM already runs with it on (verl enables it when sleep mode is off), the NCCL checkpoint engine supports it |
| checkpoints | `save_contents=[model, optimizer, extra, hf_model]`, every 50 (shared) / 12 global (recommended) | same | needed for resume after a requeue. Size: S was 34 GB per checkpoint; M is ~36 GB fp32 model shards + ~72 GB optimizer + ~19 GB HF = ~127 GB per checkpoint, ~510 GB for the shared run's 4 (~640 GB for the recommended run's 5). Measured in the smoke, see NOTES.md W1-M |
| HF export | `ckpt/global_step_N/actor/huggingface`, linked by `collect/finalize.sh` | same | same path and class as S, which vLLM loaded for every S eval |
