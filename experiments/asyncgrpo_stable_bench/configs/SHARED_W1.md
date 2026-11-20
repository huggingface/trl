# W1 (math) shared config

Every library maps these values onto its own knobs in `configs/<lib>/EQUIVALENCE.md`. Where a library can't express a value, use its closest setting and write down the gap. Revisions to this file apply to every library and are marked inline (rev. N).

## Data

- Train: `BytedTsinghua-SIA/DAPO-Math-17k`, deduplicated on the problem text, shuffled with seed 42. Prepared once into `workloads/math/data/train.parquet` with columns `problem` (str) and `answer` (str).
- Prompt: a single user message, `"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{}."`, rendered with the model's own chat template with **`enable_thinking=False` passed explicitly** (rev. 2). For Qwen3.5 the rendered prompt must end with `<|im_start|>assistant\n<think>\n\n</think>\n\n`, and every library checks this.
- Reward: 1.0 if `math_verify` judges the last `\boxed{}` in the completion equal to `answer`, else 0.0. No format reward, no length penalty. The reference implementation is `workloads/math/reward.py`, and each library calls it or matches it exactly.
- Eval: `workloads/math/eval.py --enable-thinking false --max-tokens 16384`, the same script for every library, run on saved HF checkpoints with vLLM. AIME 2024 (`HuggingFaceH4/aime_2024`) and AIME 2025 (`math-ai/aime25`) report avg@16, and MATH-500 (`HuggingFaceH4/MATH-500`) reports avg@1. Sampling and max length are the same as in training.

## Training (size S; M and L reuse it unless noted)

| knob | value |
|---|---|
| model | `Qwen/Qwen3.5-2B`, bf16 compute; master weight dtype = the library's default (record it) |
| prompts per optimizer step | 32 |
| generations per prompt | 8 (256 rollouts per step) |
| max prompt length | 2048. No prompt is ever truncated (9 DAPO prompts exceed 1024 tokens with the Qwen3.5 template; rev. 1) |
| max completion length | 16384 (rev. 2). Max model length = 2048 + 16384 = 18432. A truncated completion scores whatever reward.py gives it, which in practice is 0 (no final `\\boxed{}`): no overlong penalty and no masking of truncated samples |
| sampling | temperature 1.0, top_p 1.0, no top_k, no min_p |
| optimizer | AdamW, lr 1e-6 constant, no warmup, weight decay 0.0, betas (0.9, 0.999), grad clip 1.0 |
| optimizer steps per batch | 1 (no PPO epochs, no mini-batch split of a step's rollouts) |
| batch composition | a step trains on the next 32 complete groups as they arrive. No refill or oversampling to replace zero-advantage groups (dynamic sampling off). Whether zero-advantage tokens count in the loss denominator stays at the library default and is recorded (rev. 1) |
| loss | GRPO, group-mean advantage (no std normalization unless that's the library's only option; record it), clip eps low 0.2 / high 0.28, token-level aggregation over the batch, no KL term, no entropy bonus |
| off-policy | the library's recommended async setting (record max staleness / off-policy steps) |
| total steps | 200 |
| checkpoints | HF format at steps 50, 100, 150 and 200 |
| hardware | 1 node, 8 B200, `--exclusive`: 4 GPUs for inference, 4 for training |
| inference engine | the library's default; record the engine and version |
| seed | 42 |

## Two arms per library (rev. 1)

- **shared:** the table above. This is the like-for-like comparison.
- **recommended:** same model, data, prompt, reward, hardware split, step count, checkpoints and eval, but loss, advantage, async/off-policy, batching and zero-advantage handling at the library's own recommended defaults (e.g. prime-rl's IPO loss). This answers "what you get using each library as its authors intend". Each lib writes `configs/<lib>/w1-S-recommended.*` and lists the differences from the shared arm in EQUIVALENCE.md. Run order: shared first, recommended second.

## Correctness and engine rules (rev. 3)

- **Packing must keep linear-attention state per sample.** Any library that packs samples into one row must pass the packed-vs-alone log-prob test on Qwen3.5 (max |dlogp| about bf16 noise, < 0.1). If it fails, the library runs with packing off. That's how its authors run Qwen3.5 too (OpenRLHF), and the throughput cost is reported as is.
- **Inference engine at its CLI defaults** (CUDA graphs on, etc.), not settings copied from colocated or sleep-mode example scripts, unless the library's docs recommend them for a disaggregated setup.

## Definitions (rev. 4)

- **Weight sync:** in the shared arm, inference gets new weights after every optimizer step (verl `parameter_sync_step=1`). The recommended arm keeps the library default.
- **Dual-clip** (PPO clip on ratio > c for negative advantages): left at the library default and recorded. It only bites when the ratio is above c (verl c=3), which is rare at staleness <= 8.
- **Checkpoints when a library step spans several optimizer steps:** save at the library steps closest to optimizer steps 50/100/150/200, record the actual optimizer step in eval.jsonl, and plot eval against optimizer steps and wall-clock, never against library steps.
- **`logprob_mismatch_kl`:** only the inference-vs-trainer gap at the *same* weights. A metric that mixes in policy lag (e.g. verl `actor/ppo_kl`) is written as `null`.
- **Clip reference (rev. 5):** the 0.2/0.28 clip is measured against the sampler's (inference) logprobs, since a recomputed old policy makes the clip a no-op with one optimizer step per batch. Each library records whether it clips one side (PPO) or masks both sides outside the band (e.g. SkyRL `rollout_is`).
- **Staleness (rev. 6):** no shared cap, since each library keeps its recommended async setting. But `staleness_max` is plotted next to reward for every run, so differences are visible.
- **meta.json records the training attention backend** (e.g. FA4 via TE, fla, SDPA), so kernel bugs are visible across libraries.
- **Checkpoints for eval:** every library produces complete HF dirs `checkpoint-<step>` that vLLM can load as `Qwen3_5ForConditionalGeneration`, including the frozen vision (and MTP) weights copied from the base checkpoint if the library doesn't save them.
- **Reward call safety (rev. 7):** `reward.compute_score` must never run with math_verify's timeouts off. Libraries that call the reward from a thread or event loop use `workloads/math/reward_safe.py` (same interface; spawn process pool, 30 s hard kill scores 0). TRL's harness uses an `async def` reward instead, since its rollout child is daemonic and can't start a pool. Scores only differ for answers that would otherwise freeze the process.
- **Startup** is reported separately from training time (cold and warm), and every venv lives on /fsx.

## Revisions

- rev. 8 (2026-11-10 12:35 UTC, user): **built-in options only.** Each library maps the shared spec to its closest built-in option; no custom losses, advantage overrides or patches through plugin hooks (prime-rl custom loss, TRL loop_cls advantages, SkyRL loss registry). Gaps are recorded in EQUIVALENCE.md. Shared runs made with custom code (TRL 364895, prime-rl 364425) are superseded by built-in reruns; TRL shared now uses TRL's default std-normalized advantages, prime-rl shared its default IPO loss. Measurement-only hooks (span logging) are allowed. Our own reward wrapper is part of the task, not a library.

- rev. 7 (2026-11-10 00:14 UTC): reward call safety (the math_verify GIL freeze hung OpenRLHF twice and TRL once). Applied to future runs: verl and SkyRL adapters import reward_safe, OpenRLHF has its own pool adapter, TRL's reward is async. Runs already in flight keep the old import.

- rev. 6 (2026-11-09 20:20 UTC): staleness plotted rather than capped, attention backend in meta.json, complete loadable HF checkpoints (from the miles arm).

- rev. 5 (2026-11-09 20:00 UTC): clip reference = sampler logprobs; record one-sided clip vs two-sided mask (from the SkyRL arm).

- rev. 4 (2026-11-09 19:55 UTC): definitions for weight sync, dual-clip, multi-step checkpoints, logprob_mismatch_kl and startup (from the verl arm).

- rev. 3 (2026-11-09 19:45 UTC): added the packing correctness and engine-default rules (from the OpenRLHF arm: packing leaks GDN state, max |dlogp| 7.9; `--vllm.enforce_eager` from example scripts left inference at ~31% SM).

- rev. 2 (2026-11-09 18:48 UTC): `enable_thinking=False` passed explicitly; max completion 8192 -> 16384; eval uses the same. TRL probe 364283 (base Qwen3.5-2B, 256 train prompts x 8, T=1.0, top_p=1.0, step-0 policy): thinking off truncates 72.7% @8192 (reward 0.261, 49.6% zero-std groups), 26.1% @16384 (reward 0.416, 31.3% zero-std), 17.7% @32768. Thinking on truncates 99.8% @8192, 98.9% @16384, 91.5% @32768. Neither branch of the original rule (thinking on <=15% @16k, or thinking off <=10% @8k) passed. Thinking off @16384 is the closest, and keeping the sampling unchanged avoids pushing every library onto top-p/top-k off-policy correction paths. RL is expected to cut truncation, since a truncated answer earns 0.
- rev. 1 (2026-11-09 ~17:58 UTC): max prompt length 1024 -> 2048 (never truncate prompts); batch composition made explicit (no refill of zero-advantage groups); added the recommended arm. Prompted by the prime-rl agent.
