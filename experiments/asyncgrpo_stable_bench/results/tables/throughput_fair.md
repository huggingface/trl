# Fair W1-S throughput (throughput_fair.csv)

Built by `collect/throughput_fair.py` (hopper-cpu job 365307) from `results/raw/*-w1-S-{shared,recommended}-*`. Steady state is optimizer steps > 5: tokens of those steps divided by the wall clock from the end of the last step <= 5 to the end of the last step (same window as `e_throughput.csv`). All rates are for the whole 8-GPU job.

| run | step time (s) | all trained tok/s | useful trained tok/s | generated tok/s | zero-adv frac | zero-adv in trained batch |
|---|---|---|---|---|---|---|
| prime-rl shared | 26.0 | 60.2k | **60.2k** (exact) | 102.3k | 0.44 | no |
| TRL shared | 52.3 | 52.0k | **52.0k** (exact) | 97.9k | 0.50 | no |
| miles shared | 39.7 | 64.3k | 32.4k (~35k with bias) | 63.5k | 0.50 | yes |
| verl shared | 50.3 | 59.1k | 30.3k est. (~33k) | 58.4k | 0.49 est. | yes |
| SkyRL shared (collapsed) | 37.2 | 39.3k | 21.2k est. (~23k) | 38.5k | 0.49 partly est. | yes |
| OpenRLHF shared | 71.0 | 34.9k | 17.4k est. (~19k) | 34.4k | 0.50 est. | yes |
| prime-rl recommended | 50.0 | 53.8k | **53.8k** (exact) | 100.2k | 0.52 | no (refilled) |
| TRL recommended | 65.6 | 36.3k | **36.3k** (exact) | 69.3k | 0.51 | no |
| verl recommended | 40.2 | 66.5k | 34.7k est. (~38k) | 65.7k | 0.48 est. | yes |
| miles recommended | 46.8 | 62.6k | 29.9k (~33k) | 62.0k | 0.52 | yes |

## Column definitions

- **step time**: wall clock per optimizer step over the window. verl recommended logs one row per global step (4 optimizer steps), so its window starts at optimizer step 4 and covers 196 steps.
- **all trained tok/s**: prompt + completion tokens of every sample that went through a backward, padding excluded, counted once per sample (no prefix dedup).
- **useful trained tok/s**: the same, restricted to samples whose group had nonzero reward spread, i.e. nonzero advantage. Rewards are 0/1 and advantages are group-mean-centred (or std-normalised), so a sample has zero advantage exactly when its whole group does.
- **generated tok/s**: completion tokens produced by the inference engine. Only TRL and prime-rl count every generated rollout; the other four count completions of the step's trained batch (see below).
- **zero-adv frac**: share of generated groups whose 8 rewards are all equal, mean over the window's steps.
- **zero-adv in trained batch**: whether those groups go through the backward and count in the token-mean loss denominator.

## Per library

- **TRL**: zero-std groups are dropped in `_score_group` before batching (never in the backward or the denominator); the step is token-sized and keeps pulling scored groups. Native `batch/trained_tokens_per_step` is completion (loss-mask) tokens only, including the few IS-masked tokens; it is what the old table used (51.5k / 35.8k). Here all = `batch/forwarded_tokens_per_step x batch/packing_ratio` (sum of per-sample prompt + completion; the tree forwards each group's shared prompt once, ~0.7% fewer tokens). Inter-rank padding is stripped before the forward. Useful = all. Generated = completion tokens of vLLM requests finished in the step window (`spans-generate`), which includes the dropped zero-std groups and stale-dropped samples. Zero-adv frac = `frac_reward_zero_std`, groups scored in the window.
- **prime-rl**: `_prune_zero_advantages` (orchestrator/train_sink.py) zeroes the RL weight of zero-advantage tokens and drops samples left with no loss before the trainer, in **both** arms (shared: from the next 32 groups; recommended: before batching, then refill to 256 samples with signal). So prime-rl's trainer count, `perf/throughput x time/forward_backward`, is already useful tokens (prompt + completion, packed, cp=1 so no pad). It would also count zero-loss dummy micro-batches added to make the micro-batch count divisible by 4 ranks; checked against `progress/input_tokens + progress/output_tokens`: 0 here. Generated = `progress/rollouts x train/agg/all/num_output_tokens/mean` (every rollout that arrived in the window, including the recommended arm's discarded no-signal ones). Zero-adv frac = `solved_none + solved_all` over arrived rollouts.
- **verl** (v0.9.1 V1 separate_async): `perf/total_num_tokens` = sum of prompt + response lengths of every sample of the step's batch (`trainer_base.py:1911`, nested-tensor offsets, no pad), zero-adv groups included; they are in the `token-mean` denominator. verl logs no zero-std share (only reward/advantage min/mean/max), so the frac and useful are **estimated**. Generated = `response_length/mean x 256` (x1024 per global step in recommended): completions of the trained batch only.
- **miles**: `perf/actor_train_tok_per_s x perf/actor_train_time` = `sum(Timer().seq_lens)` = `total_lengths` (prompt + response) of the whole rollout batch, zero-adv groups included; they count in the `--calculate-per-token-loss` denominator. Packed thd, no pad. Zero-adv frac is logged (`zero_std/all_zero_percentage + all_one_percentage`), but per-group token counts aren't, so useful = all x (1 - frac) per step (count share, see bias below). Generated = `response_len/mean x 256`, trained batch only.
- **SkyRL**: every sequence, zero-adv groups included, goes through the backward and into the `token_mean` denominator (`zero_variance_filter=false`). The collector's `trained_tokens = generate/avg_num_tokens x num_seq` was labelled prompt + response but `avg_num_tokens` is `len(response)` (generators/utils.py), so it is completion only (that's why it equalled `gen_tokens`). SkyRL logs no prompt length for W1, so all adds 122.7 prompt tokens per sample (prime-rl shared's mean over the same parquet and template, ~2% of the total). Zero-adv frac: all-wrong groups = 1 - `reward/avg_pass_at_8` (exact), all-correct groups **estimated** from the step reward mean. **Padding:** W1 runs with `remove_microbatch_padding=false`, and `TensorBatch.chunk` never trims, so every 1-sequence micro-batch is forwarded at the step's global padded width (max prompt + max response). That is `padded_forward_tok_s` = 104.1k forwarded tok/s against 39.3k real: ~62% of SkyRL's forwarded tokens are padding. `configs/skyrl/EQUIVALENCE.md`'s "so nothing is padded" for W1 is wrong.
- **OpenRLHF**: `total_length` is the sample-weighted mean of `attention_mask.sum()` (prompt + response) over the trained samples, x `rollout/num_samples`. Per-sample tensors with micro-batch 1, so no pad. Zero-adv groups are kept and in the `token-mean` denominator. No zero-std share is logged (only `group_reward_std`, a mean), so frac and useful are **estimated**. Generated = `rollout/response_length_mean x n`, trained batch only.

## Approximations

- **Zero-adv frac estimate** (verl, OpenRLHF, SkyRL's all-correct part): quadratic fit of the per-step zero-std share against the step reward mean, pooled over 1,170 steps of prime-rl, miles and TRL (both arms): `f = 0.404 r^2 - 0.023 r + 0.377`. Leave-one-library-out, the predicted steady-state mean is within 0.02 of the logged one (miles 0.492 vs 0.510, prime-rl 0.501 vs 0.481, TRL 0.507 vs 0.505). SkyRL's all-correct share uses the same fit on prime-rl's `solved_all` alone. The SkyRL run collapsed (reward ~0.05 late), so its fit point sits outside most of the data.
- **Count share as token share** (miles, verl, SkyRL, OpenRLHF): useful = all x (1 - zero-adv group share). prime-rl logs both, and its shipped token share is on average 1.09x the count share (p10 0.98, p90 1.21): nonzero-advantage samples run longer than all-wrong/all-right groups on average. The "~" figures in parentheses in the table apply that 1.09 factor; the CSV keeps the uncorrected value.
- **Generated tokens for miles, verl, SkyRL, OpenRLHF** count completions of the samples trained in the step, not the engine's output: groups dropped for staleness or failure and partial generations in flight are missing. Since these libs train every generated group, it tracks the engine closely in steady state. TRL's and prime-rl's engines produce ~100k tok/s and train on half of it; miles' and verl's run at 58-66k.
- `lib_reported_trained_tok_s` in the CSV is each collector's `trained_tokens` (what `e_throughput.csv` and Finding 4 used). It equals the new all-trained column except TRL (completion-only, 1% lower) and SkyRL (completion-only, 2% lower).

## Collector fixes (job 365307 re-collected these runs)

- `collect/prime-rl.py`: `gen_tokens` was `progress/output_tokens`, which counts only the shipped (nonzero-advantage) samples' completions; now every arrived rollout (`progress/rollouts x all/num_output_tokens/mean`). `trained_tokens` unchanged, comment and meta note corrected.
- `collect/skyrl.py`: `trained_tokens` value unchanged but relabelled completion only; new W1 field `forwarded_tokens_padded = batch_padded_seq_len x num_seq`.
