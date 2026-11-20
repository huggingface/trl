# OpenRLHF ↔ SHARED_W1 mapping (W1-S)

OpenRLHF v0.11.2 (SHA 17fd4a79), `openrlhf.cli.train_ppo_ray`. Flags live in `w1-S.sh`; code references are to `/fsx/amine_dirhoussi/stable-bench-src/openrlhf`.

| SHARED_W1 knob | value | OpenRLHF knob | OpenRLHF value | semantics / gap |
|---|---|---|---|---|
| model | `Qwen/Qwen3.5-2B` | `--actor.model_name_or_path` | local snapshot of `Qwen/Qwen3.5-2B` | Has `vision_config`, so it loads as `AutoModelForImageTextToText` (whole VLM). `--actor.freeze_visual_encoder` sets `requires_grad=False` on everything outside `language_model.*` / `lm_head.*`; only trainable params are synced to vLLM. |
| compute / master dtype | bf16 compute, lib default master | `--ds.param_dtype bf16` | bf16 | DeepSpeed bf16 mode: bf16 params and grads, fp32 master weights + Adam states inside the optimizer. master_dtype = fp32. |
| train backend | lib default | `--ds.zero_stage` | 2 | DeepSpeed. ZeRO-2 is the CLI default and the performance guide says to go to 3 only under OOM; the shipped math scripts use 3. Weight sync under ZeRO-2 broadcasts from rank 0 without an all-gather. |
| prompts per optimizer step | 32 | `--rollout.batch_size` | 32 | Counts prompts. |
| generations per prompt | 8 | `--rollout.n_samples_per_prompt` | 8 | |
| rollouts per step | 256 | `--train.batch_size` | 256 | Counts samples (rollouts), global across DP ranks. Optimizer steps per rollout batch = rollout.batch_size × n / train.batch_size = 1 (`replay_buffer.setup_dynamic_batch`). |
| optimizer steps per batch | 1 | `--train.max_epochs` | 1 | One pass over the batch, one optimizer step. `--train.micro_batch_size 1`: each rank runs its 64 samples one at a time and DeepSpeed accumulates gradients over the 64 micro-steps (no packing, no dynamic batch, no padding; see "attention / packing"). |
| max prompt length | 2048, never truncated (rev. 1) | `--data.max_len` − `--rollout.max_new_tokens` | 18432 − 16384 = 2048 | `max_len` is prompt + completion. A prompt over 2048 tokens would be left-truncated with a warning (`utils/agent.py`), but the longest W1 prompt is 1517 tokens with the template, so none is. |
| max completion length | 16384 (rev. 2) | `--rollout.max_new_tokens` | 16384 | vLLM `max_tokens`. Truncated rollouts (finish_reason `length`) keep their reward (no `--reward.stop_properly_penalty_coef`), logged as `rollout/truncated_rate`. |
| sampling | T 1.0, top_p 1.0, no top_k, no min_p | `--rollout.temperature`, `--rollout.top_p` | 1.0, 1.0 | `top_k=-1`, vLLM default `min_p=0`. `min_tokens=1` (OpenRLHF default) forces at least one token. The trainer divides logits by the same temperature (1.0, no-op). |
| prompt rendering | chat template, `enable_thinking=False` (rev. 2) | pre-rendered by `configs/openrlhf/w1_data.py`, no `--data.apply_chat_template` | `THINKING=false` (`w1_train_S_think-false.parquet`) | `--data.apply_chat_template` can't pass template kwargs, so w1_data.py makes the same call OpenRLHF makes (`AutoProcessor.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`) plus `enable_thinking` when set, and the rendered string goes to vLLM unchanged. Qwen3.5-2B's template defaults to thinking off (`<think>\n\n</think>\n\n` after the assistant tag); `THINKING=true` renders `<think>\n`. |
| reward | `workloads/math/reward.py` | `--reward.remote_url` | `configs/openrlhf/w1_reward.py` | Calls `compute_score` unchanged on the completion text (decoded with special tokens kept; `last_boxed` is unaffected). OpenRLHF calls the adapter in a thread of the vLLM rollout actor; the adapter hands `compute_score` to an 8-process spawn pool, so it runs on a process main thread with math_verify's 5 s timeouts on (as the reference does on its main thread) and a 30 s hard kill scores 0. Run in the actor's thread, an answer like `9^{9^{9}}` held the GIL and froze the actor, which hung two W1 runs (envs/openrlhf/NOTES.md item 19). Scores are identical except for answers that time out (0 instead of hanging). |
| optimizer | AdamW | `--actor.optim adam` | DeepSpeed FusedAdam (AdamW mode) | |
| lr | 1e-6 constant, no warmup | `--actor.adam.lr`, `--actor.lr_scheduler`, `--actor.lr_warmup_ratio` | 1e-6, `constant`, 0 | Lib default is `cosine_with_min_lr` with 3% warmup, overridden. |
| weight decay | 0.0 | `--actor.adam.weight_decay` | 0.0 | |
| betas | (0.9, 0.999) | `--actor.adam.betas` | 0.9 0.999 | Lib default is (0.9, 0.95), overridden. |
| grad clip | 1.0 | `--actor.max_norm` | 1.0 | Global grad-norm clip by DeepSpeed. |
| advantage | group mean, no std | `--algo.advantage.estimator dr_grpo` | `dr_grpo` | `experience_maker.compute_advantages_and_returns`: r − mean(group); no std division and no batch whitening (whitening only applies to gae / reinforce / reinforce_baseline). `group_norm` would divide by the group std. |
| clip eps | low 0.2 / high 0.28 | `--actor.eps_clip_low_high` | 0.2 0.28 | Standard PPO min(surr1, surr2), ratio clamped to [0.8, 1.28]. No dual clip. |
| loss aggregation | token-level over the batch | `--actor.loss_agg_mode token-mean` | token-mean | Sum of masked token losses / total response tokens of the whole optimizer step across all DP ranks and micro-batches (`utils/loss_utils.py`, `aggregate_loss`). Matches "token-level over the batch". |
| batch composition / zero-advantage groups | next 32 complete groups, no refill; record denominator (rev. 1) | `--algo.dynamic_filtering_enable` off | off | A step takes the next 32 prompts × 8 samples as generated. Groups with all-equal rewards get advantage 0 (Dr. GRPO), so their tokens contribute zero loss, but **they still count in the token-mean denominator** (`batch_num_tokens` sums every action token of the step). |
| old logprobs | | (always) | trainer recompute | Before training, the actor runs an extra no-grad forward to get π_old on the trainer side; the PPO ratio is π_θ / π_old (≈ 1 up to numerics, since there is one optimizer step per batch, so the PPO clip barely binds). |
| IS correction | lib's recommended | `--algo.advantage.is_correction_level token`, `_mode mask`, `_threshold 0.5 5.0` | token mask (ICEPOP) [0.5, 5] | Per-token weight π_old_trainer / π_vLLM; tokens with weight outside [0.5, 5] get zero loss, the rest are multiplied by the weight. This is what both of OpenRLHF's async and math example scripts turn on, and it makes vLLM return logprobs, which gives `vllm_kl` (the logprob-mismatch metric). |
| KL term | none | `--algo.kl.init_coef 0` | 0 | No reference model is created when init_coef is 0. |
| entropy bonus | none | `--actor.entropy_coef` | unset | Entropy isn't even computed. |
| off-policy | lib's recommended async | `--train.async_enable --train.async_queue_size 1` | queue 1, no partial rollout | Docs: "start `--train.async_queue_size` at 1". A lock keeps a batch's generation and the weight broadcast apart, so each batch comes from one policy version: batch 1 is on-policy, every later batch was generated one version behind (max staleness 1). Partial rollout (vLLM pause/resume, mixed versions inside a sample) is the more aggressive opt-in mode, not used. |
| total steps | 200 | `--data.max_samples` | 6400 prompts (= 200 × 32), 1 episode | No max-steps flag: the run ends when the 6400 prompts are consumed. |
| data order | shuffled seed 42 | (fixed) | first 6400 rows of train.parquet, reshuffled by OpenRLHF's sampler with `--train.seed 42` | OpenRLHF's prompt dataloader always shuffles (`setup_dataloader(..., shuffle=True)`), so the prompt order differs from libs that read train.parquet in order. Same prompt pool. |
| checkpoints | HF at 50/100/150/200 | `--ckpt.save_steps 50 --ckpt.save_hf` | `ckpt/global_step{50,100,150,200}_hf` | Also a DeepSpeed checkpoint (max 1 kept) for resume after a requeue (`--ckpt.load_enable`). |
| hardware | 1 node, 4 inference + 4 train | `--vllm.num_engines 4 --vllm.tensor_parallel_size 1`, `--actor.num_gpus_per_node 4` | 4 + 4, not colocated | |
| inference engine | lib default | vllm | vLLM 0.29.0 (`AsyncLLMEngine` in Ray actors) | CUDA graphs on (the `--vllm.enforce_eager` CLI default is off; the example scripts turn it on for colocated sleep mode, and in our async split it left the engines at ~31% SM util); `--vllm.enable_prefix_caching` per the performance guide when n > 1. `gpu_memory_utilization 0.9` (examples use 0.5-0.95 depending on colocation). |
| attention / packing | | `--ds.attn_implementation flash_attention_2`, packing off | FA2 2.8.3, micro-batch 1 | OpenRLHF's `--ds.packing_samples` (which `--train.dynamic_batch_enable` forces on) passes only `position_ids` for a packed row, and Qwen3.5's gated-DeltaNet layers ignore them, so recurrent and conv state leaks between packed samples (max \|dlogp\| 7.9 measured, envs/openrlhf/NOTES.md item 8). Packing off is correct and matches OpenRLHF's own Qwen3.5 example. Linear-attention layers run fla's `chunk_gated_delta_rule` and the `causal_conv1d` kernel (verified bound). |
| seed | 42 | `--train.seed` | 42 | Also the vLLM engine seed. |
| CUDA allocator | lib default | env `PYTORCH_CUDA_ALLOC_CONF` (set in `launch/openrlhf/common.sh`) | `expandable_segments:True` | Not a training knob. Trainer ranks hit the allocator's OOM-and-retry path on the ~16.5 GB fp32 logits of 16k-token micro-batches; OpenRLHF has no chunked log-prob option and refuses Liger for VLM checkpoints (NOTES.md item 20). |
| run supervision | | `launch/openrlhf/w1.sbatch` watchdog | kill after 20 min without log output | Launcher only: turns a silent hang into a Slurm failure (NOTES.md item 21). |

## Recommended arm (`w1-S-recommended.sh`)

Everything above stays the same (model, data, prompt rendering, reward, lengths, sampling, optimizer, 4 + 4 GPUs, 200 steps, checkpoints, seed, packing off), except these knobs, which follow OpenRLHF's own async RLVR advice: README ("REINFORCE++-baseline (best for RLVR)", execution-mode table) and its async RLVR recipe `examples/scripts/train_reinforce_baseline_ray_agent_async.sh`.

| knob | shared arm | recommended arm | source |
|---|---|---|---|
| advantage | `dr_grpo`: r − group mean | `reinforce_baseline`: r − group mean, then whitening over all action tokens of the batch ((A − mean) / std) | README algorithm table; both RLVR example scripts |
| clip eps | 0.2 / 0.28 | 0.2 / 0.2 (CLI default `--actor.eps_clip 0.2`) | the async recipe sets no clip |
| KL | none, no reference model | k2 KL as a loss, coef 1e-5, frozen reference model colocated with the actor on the 4 training GPUs | async recipe (`--algo.kl.init_coef 1e-5 --algo.kl.use_loss --algo.kl.estimator k2`) |
| zero-advantage groups | kept, count in the denominator | DAPO dynamic filtering: groups whose mean reward is 0 or 1 are dropped and replaced by fresh prompts until 32 groups pass. Uses more prompts per step, so the 6400-prompt pool can run out before 200 steps | async recipe (`--algo.dynamic_filtering_enable --algo.dynamic_filtering_range 0.0 1.0`) |
| async | queue 1, VLLMLock (each batch from one policy version, staleness exactly 1) | queue 1 + `--train.partial_rollout_enable`: weight sync pauses and resumes vLLM, and in-flight samples finish on the new weights (mixed versions, staleness not logged) | async recipe; README pairs it with the token IS correction, which both arms already use |
| batching | micro-batch 1, no packing | same. OpenRLHF recommends `--ds.packing_samples` + dynamic batch "always", but packing is incorrect for Qwen3.5 (see above) | correctness |

## M (`w1-M.sh`, `w1-M-recommended.sh`)

M reuses every S value in both arms. The M configs source the S ones and append the overrides below (argparse keeps the last value). Launch with `SIZE=M` (and `CONFIG_NAME=w1-M-recommended` for the recommended arm); `w1.sbatch` picks the 9B snapshot and the M data file from `SIZE`.

| knob | S value | M value | why |
|---|---|---|---|
| model | `Qwen/Qwen3.5-2B` snapshot | `Qwen/Qwen3.5-9B` snapshot (`w1.sbatch`, `SIZE=M`) | size M. Same architecture (`Qwen3_5ForConditionalGeneration`, 32 layers, hidden 4096, vocab 248320), vision tower frozen as at S |
| prompt data | `w1_train_S_think-false.parquet` | `w1_train_M_think-false.parquet`, rendered by `w1_data.py` from the 9B processor with `enable_thinking=False` | the 9B `chat_template.jinja` differs from the 2B one (tokenizer is identical), so the 2B-rendered prompts aren't reused |
| `--ds.zero_stage` | 2 | 2 (written out explicitly) | fits: bf16 params replicated (~19 GB) + fp32 master and Adam moments sharded over 4 ranks (~28 GB/rank) + grads, leaving ~100 GB of 180 GB for a 16k-token micro-batch. ZeRO-3 would add a param all-gather to each of the 64 micro-steps and the old-logprob pass, and change the weight-sync path. The recommended arm's reference model is a full bf16 replica per training GPU under ZeRO-2 (OpenRLHF shards it only under ZeRO-3), ~19 GB more |
| micro-batch / packing | 1 sample, packing off | same | packing is wrong for Qwen3.5 (SHARED_W1 rev. 3); micro-batch 1 is already the minimum, so tokens per optimizer step are unchanged (256 rollouts) |
| gradient checkpointing | on | on | |
| vLLM | 4 engines, TP 1, `gpu_memory_utilization 0.9` | same | 9B bf16 weights are ~19 GB, the rest is KV cache |
| `--ckpt.save_steps` | 50 | `${SAVE_STEPS:-50}` | env knob for smokes only (time a 9B save); runs keep 50 |

Smoke 367077 (shared arm, 4 steps, COMPLETED) confirmed the sizing: trainer GPUs peak at 76-109 GiB with 16k-token micro-batches present, vLLM GPUs at 163 GiB, `vllm_kl` 0.0002. Steady step ~100-115 s, generation-bound (S ~71 s). A 9B save takes ~2 min: a 119 GB DeepSpeed checkpoint (1 kept) plus an 18 GB HF dir. Details: envs/openrlhf/NOTES.md "W1-M" item 5.
