# Text-to-SQL, Qwen3.5-35B-A3B LoRA: TRL vs prime-rl vs SkyRL

Generated 2026-11-18 16:19 UTC by `collect/t2sql_35b.py` (rerun it as checkpoints and evals land). Greedy evals with TRL's `eval.sbatch` (T=0, 1 sample per task, outcome-only score) for every model; SE is the binomial standard error. Hard overlaps the train split, so validation and arcwise are the held-out numbers.

Hours: wall clock from the job's sacct start to the step's log time (includes startup: model load, vLLM, first rollouts). GPU-hours: hours x the job's GPUs. Episodes trained: samples that went into optimizer steps (one per episode in all three). Episodes generated: rollouts finished by then (TRL from episodes.jsonl; prime-rl drops zero-advantage groups, and its refill run regenerates them; SkyRL trains everything it generates).

## Setup

- **TRL** (job 373132, 8 x B200, at step 100): clipped GRPO (eps 0.2 / 0.28), no KL, mean-centred advantage; lr 5e-5, 10 warmup steps, cosine to 10%; LoRA r32 / alpha 64 on attention + GDN + shared expert (310 modules); batch = token budget 32,768 x 10 micro-batches per rank (mean ~57 groups, ~907 samples per step, zero-std groups dropped); 16 samples per question; T = 1.0; async, max staleness 8, 256 episodes in flight; 8 GPUs (trainer EP4 x FSDP4 on 4, vLLM DP4 + EP on 4).
- **TRL (with #8212 fix)** (job 374011, 8 x B200, at step 100): same settings as TRL 373132, plus a per-request vLLM cache_salt of the weight version (the TRL #8212 fix: vLLM was paused with clear_cache=false, so the shared prompt prefix kept base-weight KV all run).
- **prime-rl** (job 389085, 8 x B200, at step 100): same settings as prime-rl 373642 except the batch: 960 samples per step (60 x 16), zero-advantage groups dropped and not refilled (constant_trainer_batch_size = false), the same rule as TRL.
- **prime-rl, refill** (job 373642, 8 x B200, at step 40): prime-rl's IPO loss (masked where |pi - mu| > 0.3, no ratio clip), no KL, mean-centred advantage; lr 5e-5, 10 warmup steps, cosine to 10%; LoRA r32 / alpha 64 on the same 310 modules; batch 912 samples (57 x 16), zero-advantage samples pruned and refilled; 16 samples per question; T = 1.0; async, max staleness 8, 256 episodes in flight; 8 GPUs (trainer EP4 x FSDP4 on 4, vLLM DP4 on 4).
- **SkyRL** (job 373788, 4 x B200, at step 42): SkyRL-SQL Megatron LoRA recipe: dppo loss (main's default), no KL, mean-centred advantage; lr 3e-5 constant; LoRA r32 / alpha 64; batch 256 prompts x 5 samples = 1,280 episodes, one optimizer step; 5 samples per question; T = 0.6, top_p 0.95; synchronous (staleness 0), colocated; 4 GPUs (Megatron TP2 EP4 DP2 + one vLLM TP4; the recipe assumes 1 x 8, 4 was what was free).

## Per checkpoint

| library | step | hours | GPU-hours | episodes trained | episodes generated | validation (398) | arcwise (498) | hard (658, in train) |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0.00 | 0.0 | 0 | 0 | 0.827 ± 0.019 | 0.751 ± 0.019 | 0.524 ± 0.019 |
| TRL | 20 | 1.35 | 10.8 | 18,935 | 20,294 | 0.844 ± 0.018 | 0.811 ± 0.018 | 0.582 ± 0.019 |
| TRL | 40 | 2.69 | 21.5 | 37,660 | 39,758 | 0.862 ± 0.017 | 0.801 ± 0.018 | 0.599 ± 0.019 |
| TRL | 60 | 3.74 | 29.9 | 55,564 | 58,540 | 0.862 ± 0.017 | 0.823 ± 0.017 | 0.634 ± 0.019 |
| TRL | 80 | 4.83 | 38.6 | 73,051 | 76,820 | 0.857 ± 0.018 | 0.819 ± 0.017 | 0.626 ± 0.019 |
| TRL | 100 | 5.96 | 47.7 | 90,747 | 95,351 | 0.844 ± 0.018 | 0.805 ± 0.018 | 0.634 ± 0.019 |
| TRL (with #8212 fix) | 20 | 1.16 | 9.3 | 17,053 | 19,016 | 0.852 ± 0.018 | 0.809 ± 0.018 | 0.600 ± 0.019 |
| TRL (with #8212 fix) | 40 | 2.74 | 22.0 | 29,554 | 41,539 | 0.867 ± 0.017 | 0.839 ± 0.016 | 0.666 ± 0.018 |
| TRL (with #8212 fix) | 60 | 4.52 | 36.2 | 40,510 | 68,981 | 0.872 ± 0.017 | 0.847 ± 0.016 | 0.681 ± 0.018 |
| TRL (with #8212 fix) | 80 | 6.58 | 52.7 | 50,650 | 96,368 | 0.887 ± 0.016 | 0.859 ± 0.016 | 0.707 ± 0.018 |
| TRL (with #8212 fix) | 100 | 8.28 | 66.2 | 60,606 | 122,171 | 0.877 ± 0.016 | 0.871 ± 0.015 | 0.704 ± 0.018 |
| prime-rl | 20 | 1.92 | 15.4 | 14,337 | 19,200 | 0.849 ± 0.018 | 0.819 ± 0.017 | 0.585 ± 0.019 |
| prime-rl | 40 | 3.99 | 31.9 | 19,731 | 38,400 | 0.779 ± 0.021 | 0.757 ± 0.019 | 0.617 ± 0.019 |
| prime-rl | 60 | 6.34 | 50.8 | 25,719 | 57,600 | 0.829 ± 0.019 | 0.817 ± 0.017 | 0.676 ± 0.018 |
| prime-rl | 80 | 8.60 | 68.8 | 30,585 | 76,800 | 0.887 ± 0.016 | 0.859 ± 0.016 | 0.729 ± 0.017 |
| prime-rl | 100 | 10.94 | 87.5 | 34,793 | 96,000 | 0.877 ± 0.016 | 0.857 ± 0.016 | 0.720 ± 0.018 |
| prime-rl, refill | 20 | 3.09 | 24.8 | 18,240 | 31,600 | 0.862 ± 0.017 | 0.819 ± 0.017 | 0.632 ± 0.019 |
| prime-rl, refill | 40 | 11.87 | 94.9 | 36,480 | 100,544 | 0.857 ± 0.018 | 0.857 ± 0.016 | 0.717 ± 0.018 |
| SkyRL | 20 | 5.29 | 21.2 | 25,600 | 25,600 | 0.879 ± 0.016 | 0.823 ± 0.017 | 0.652 ± 0.019 |
| SkyRL | 40 | 9.89 | 39.6 | 51,200 | 51,200 | 0.887 ± 0.016 | 0.851 ± 0.016 | 0.701 ± 0.018 |

## Per training step, 10-step bins

Step time and rollout share leave out step 1 (startup). Rollout share: time the trainer waited for rollouts / step time (TRL, prime-rl: async); for SkyRL (synchronous) it is generation time / step time. Output tokens: sampled tokens per episode (TRL `completions/mean_length`, prime-rl `out_tokens_mean`, SkyRL `generate/avg_assistant_tokens`). Train reward is each library's rollout reward with its own batch (16 samples per question in TRL and prime-rl, 5 in SkyRL).

**TRL**

| steps | step time (s) | rollout share | output tokens | turns | train reward |
|---|---|---|---|---|---|
| 1-10 | 192 | 68% | 819 | 2.28 | 0.679 |
| 11-20 | 175 | 65% | 838 | 2.33 | 0.726 |
| 21-30 | 232 | 74% | 876 | 2.43 | 0.714 |
| 31-40 | 250 | 76% | 843 | 2.25 | 0.741 |
| 41-50 | 144 | 57% | 875 | 2.36 | 0.736 |
| 51-60 | 236 | 74% | 870 | 2.39 | 0.736 |
| 61-70 | 199 | 69% | 908 | 2.40 | 0.770 |
| 71-80 | 191 | 68% | 928 | 2.35 | 0.754 |
| 81-90 | 159 | 61% | 898 | 2.39 | 0.776 |
| 91-100 | 249 | 76% | 942 | 2.36 | 0.766 |

**TRL (with #8212 fix)**

| steps | step time (s) | rollout share | output tokens | turns | train reward |
|---|---|---|---|---|---|
| 1-10 | 170 | 64% | 870 | 2.33 | 0.670 |
| 11-20 | 148 | 60% | 1048 | 2.58 | 0.781 |
| 21-30 | 280 | 79% | 1151 | 2.76 | 0.768 |
| 31-40 | 289 | 80% | 1215 | 2.77 | 0.720 |
| 41-50 | 277 | 78% | 1226 | 2.91 | 0.746 |
| 51-60 | 363 | 84% | 1269 | 2.93 | 0.764 |
| 61-70 | 349 | 83% | 1267 | 3.00 | 0.762 |
| 71-80 | 393 | 85% | 1268 | 2.95 | 0.761 |
| 81-90 | 312 | 81% | 1337 | 3.06 | 0.735 |
| 91-100 | 298 | 80% | 1291 | 3.00 | 0.772 |

**prime-rl**

| steps | step time (s) | rollout share | output tokens | turns | train reward |
|---|---|---|---|---|---|
| 1-10 | 204 | 29% | 898 | 2.37 | 0.722 |
| 11-20 | 331 | 72% | 1460 | 3.07 | 0.828 |
| 21-30 | 366 | 85% | 1589 | 3.62 | 0.859 |
| 31-40 | 377 | 87% | 1488 | 3.99 | 0.887 |
| 41-50 | 415 | 86% | 1644 | 4.41 | 0.890 |
| 51-60 | 433 | 85% | 1704 | 4.50 | 0.880 |
| 61-70 | 410 | 87% | 1602 | 4.40 | 0.904 |
| 71-80 | 401 | 89% | 1582 | 4.24 | 0.912 |
| 81-90 | 424 | 89% | 1685 | 4.30 | 0.890 |
| 91-100 | 416 | 89% | 1670 | 4.22 | 0.916 |

**prime-rl, refill**

| steps | step time (s) | rollout share | output tokens | turns | train reward |
|---|---|---|---|---|---|
| 1-10 | 215 | 36% | 904 | 2.42 | 0.729 |
| 11-20 | 747 | 78% | 1438 | 3.07 | 0.839 |
| 21-30 | 1300 | 86% | 1670 | 3.64 | 0.880 |
| 31-40 | 1858 | 89% | 2142 | 4.39 | 0.894 |

**SkyRL**

| steps | step time (s) | rollout share | output tokens | turns | train reward |
|---|---|---|---|---|---|
| 1-10 | 880 | 33% | 1075 | 2.58 | 0.807 |
| 11-20 | 850 | 29% | 1231 | 2.70 | 0.864 |
| 21-30 | 810 | 28% | 1211 | 2.64 | 0.881 |
| 31-40 | 845 | 31% | 1185 | 2.74 | 0.887 |
| 41-42 | 814 | 26% | 1085 | 2.79 | 0.893 |
