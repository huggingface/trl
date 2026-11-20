# SkyRL W1-S shared run 364970: why it collapsed

Run `skyrl-w1-S-shared-364970` (Qwen3.5-2B, DAPO math, 16k completions, fully async, staleness 4, `policy_loss_type=rollout_is` with `eps_clip_low/high=0.2/0.28`, token mean, group-mean advantage without std, lr 1e-6 constant, no KL). Analysis scripts and outputs are in `/fsx/amine_dirhoussi/stable-bench-runs/skyrl/collapse-364970/` (all CPU work ran on hopper-cpu).

## Verdict

- **Most likely ours, not SkyRL.** The collapse is a self-reinforcing drift between the trainer and the sampler that starts around step 33. The most likely driver is our mapping of the shared clip onto `rollout_is`: PPO's 0.2 / 0.28 epsilons used as a *two-sided mask* band. That drops the restoring gradient on tokens that drift the wrong way, and SkyRL never runs `rollout_is` with a band this tight. Every pipeline-side suspect we could check (exports, staleness, step-1 numerics, untied embeddings) is ruled out or too small. Not yet proven: the DPPO control crashed at step 36, before the window where the drift took off. An A/B from the step-50 export is queued (last section).
- **Separate SkyRL bug found:** the FSDP init unties tied embeddings (`lm_head` vs `embed_tokens`), so the trainer and vLLM run slightly different models. It's real and upstream-reportable, but it moves logprobs by only 0.004-0.014 nats here and didn't cause the collapse.
- **Redo the shared run:** yes, with a one-sided PPO-clip loss against the sampler (`ppo_rollout`, registered through SkyRL's public loss registry, no source patch). Ideally also re-tie the embeddings.
- Don't publish "SkyRL is unstable" from run 364970.

## What the collapse looks like

Eval samples (`eval-split/batch-20261110T053644/eval-samples-<step>.jsonl`, 1460 completions per checkpoint):

| step | score (all) | median chars | starts with `<think>` | has `\boxed` | >5% non-ASCII | top prefix |
|---|---|---|---|---|---|---|
| 50 | 0.354 | 10,905 | 0% | 98% | 0.1% | "To find the ..." (normal) |
| 100 | 0.057 | 23 | 65% | 12% | 15% | `<think>\n\n</think>\n\n`, then garbage ("filePath", "commiss") or empty |
| 150 | 0.355 | 14,282 | 79% | 76% | 5% | `<think>\n\n</think>: We are ...` (long, coherent) |
| 200 | 0.095 | 1,098 | 75% | 14% | 2% | `<think>\n\n</think>: Here's a thinking process ...`, often cut off by an early EOS ("Here's a") |

The prompt already ends with `<think>\n\n</think>\n\n` (thinking off). From step 100 on, the policy re-opens and closes an empty think block at the start of most answers, often followed by a colon (`</think>:`), which the base model never does. At step 100 most answers are a few tokens of junk with some full-width punctuation and Chinese tokens. At step 200 the model emits a fixed "Here's a thinking process" template and stops early, mid-sentence. Step 150 sits between the two dips and scores near base, which is why its eval looked fine.

## When it starts

Per-step metrics from `results/raw/skyrl-w1-S-shared-364970/steps.jsonl` (`native`). "Out of band" is `policy/loss_metrics/clip_ratio`, which for `rollout_is` is the fraction of loss tokens whose ratio to the sampler is outside (0.8, 1.28) and therefore masked. "Trainer vs sampler" is `policy/minibatch_rollout_logprobs_abs_diff_mean`, the mean |trainer logprob - vLLM logprob| on the trained tokens (trainer weights are 0-4 steps newer than the sampler's).

| step | reward | entropy | mean len | out of band | trainer vs sampler (nats) | min len |
|---|---|---|---|---|---|---|
| 1 | 0.63 | 0.38 | 6.2k | 0.03% | 0.009 | 444 |
| 10 | 0.50 | 0.37 | 10.2k | 0.10% | 0.010 | 746 |
| 20 | 0.49 | 0.40 | 10.4k | 0.11% | 0.012 | 664 |
| 30 | 0.61 | 0.41 | 8.9k | 0.17% | 0.013 | 909 |
| 40 | 0.28 | 0.46 | 9.1k | 0.46% | 0.017 | 891 |
| 45 | 0.57 | 0.47 | 7.2k | 1.6% | 0.026 | 937 |
| 50 | 0.48 | 0.57 | 5.8k | 6.1% | 0.055 | 854 |
| 60 | 0.39 | 1.05 | 5.3k | 6.8% | 0.066 | 522 |
| 70 | 0.32 | 1.36 | 3.2k | 12.9% | 0.091 | 480 |
| 80 | 0.26 | 1.41 | 3.0k | 19.4% | 0.15 | 241 |
| 90 | 0.01 | 0.81 | 0.3k | 48.9% | 1.15 | 2 |
| 93 | 0.03 | 1.17 | 0.7k | 60.3% | 2.94 | 2 |
| 100 | 0.04 | 0.62 | 2.7k | 36.7% | 1.44 | 2 |
| 120 | 0.07 | 0.17 | 7.4k | 7.3% | 0.14 | 2 |
| 130 | 0.23 | 0.07 | 6.4k | 2.7% | 0.027 | 2 |
| 150 | 0.35 | 0.20 | 5.9k | 6.1% | 0.070 | 2 |
| 170 | 0.38 | 0.08 | 5.4k | 3.0% | 0.031 | 2 |
| 190 | 0.26 | 0.17 | 1.8k | 6.9% | 0.077 | 9 |
| 200 | 0.05 | 0.21 | 0.9k | 12.1% | 0.21 | 2 |

Sequence of events:

1. Steps 1-30 look healthy and match the recommended arm step for step (see below): trainer-vs-sampler gap 0.009 -> 0.013 nats, under 0.2% of tokens out of band, entropy 0.38-0.42.
2. From step ~33 the out-of-band fraction starts compounding: 0.25% (33), 0.46% (40), 1.6% (45), 6% (50). The trainer-vs-sampler gap grows with it (0.013 -> 0.055) while staleness stays flat at 4 and grad norm stays at 0.08-0.12. Mean completion length is already falling (9-10k -> 5.8k).
3. Entropy follows from step ~45 (0.47 -> 1.05 at 60 -> 1.87 at 71) and reward drops (0.5 -> 0.2 by step 71). By step 85-95 the policy emits near-empty answers (minimum response length 2-7 tokens), reward 0.01-0.05, 49-60% of tokens out of band and a 1-3 nat trainer-vs-sampler gap.
4. Steps 105-170 partly recover, but into a low-entropy mode (0.07-0.2, below the base model's 0.4) with the `<think>\n\n</think>:` prefix. Train reward climbs back to 0.35-0.45.
5. Steps 175-200 collapse a second time: length 5k -> 0.85k, reward 0.05.

None of the other libraries' shared arms show anything like this on the same config: their reward at step 200 is 0.47-0.72, completion length stays 7.6-13.9k, entropy (where logged) stays 0.39-0.53 (TRL shared) and 0.42-0.47 (prime-rl shared), and their `logprob_mismatch_kl` stays at 3-9e-4 throughout. Prime-rl's shared arm runs at a comparable staleness (mean 4.1-4.7) without drifting.

## What it is not

- **HF exports.** All four exports are `Qwen3_5ForCausalLM`, fp32, 321 tensors, no vision or MTP weights (expected: `language_model_only=true` never builds them), `tie_word_embeddings: true` in the config, with a separate `lm_head.weight` tensor. Step 50 and 150 evaluate near base, so loading works.
- **Staleness.** `async/staleness_mean` is 3.8-4.2 for the whole run, flat across the healthy and collapsed phases. The recommended arm has the same staleness.
- **The trainer forward at step 1.** The step-1 trainer-vs-sampler gap (0.0087 nats at staleness 0) is ordinary bf16 noise and matches the recommended arm's 0.0086.
- **Untied embeddings (a real SkyRL bug, but not this collapse).** See the next section: it makes the trainer and vLLM disagree by only 0.004-0.014 nats on these checkpoints, with no token leaving the band, and the `<think>` re-emission shows up equally in both views.

## A SkyRL bug found on the way: FSDP init unties tied embeddings

`FSDPStrategy._fsdp_init_model` (`skyrl/backends/skyrl_train/distributed/fsdp_strategy.py`) calls `module.to(torch.device("meta"))` before `apply_fsdp2` and then loads the full state dict. Moving a module to meta replaces each module's parameter separately, so `lm_head.weight` and `embed_tokens.weight` stop being the same tensor; SkyRL never calls `model.tie_weights()` afterwards (accelerate's `fsdp2_prepare_model`, which the comment cites, does). Checked on CPU with SkyRL's venv (torch 2.13): Qwen3.5-2B is tied after `from_pretrained`, untied after `.to("meta")`, tied again after `tie_weights()`. The exports confirm it: `lm_head.weight != embed_tokens.weight` at every checkpoint (relative difference 2.4e-4 at step 50, 5.5e-4 at step 200), both moved away from the base embedding.

Consequence: the trainer trains two separate matrices, while vLLM (`tie_word_embeddings=true`) keeps one shared matrix and its weight loader skips `lm_head.weight` as an alias of `embed_tokens.weight`. So the sampler (and the eval) never see the trainer's output-head updates and use the trainer's input embedding as their head. On 3.4k tokens of step-50 eval text (fp32, `collapse-364970/cpu_check.py`), the sampler view `(E, E)` differs from the trainer view `(E, H)` by mean |dlogp| 0.0039 / 0.0044 / 0.0052 / 0.0137 at steps 50 / 100 / 150 / 200 (p99 0.03-0.11, max 0.16), with zero tokens outside (0.8, 1.28). The first-token probability of `<think>` after the prompt is the same in both views (0.32 vs 0.32 at step 100, 0.57 vs 0.57 at step 150). So it is a silent-correctness bug that affects every tied-embedding model on SkyRL's FSDP backend, but it is far too small to explain 1-3 nat gaps and is not what drove this run. Fix: `module.tie_weights()` right after `module.to(torch.device("meta"))` (upstream-reportable). Logged as a separate ledger row.

## Cause

The likely cause is our loss mapping: PPO's clip epsilons (0.2 / 0.28) plugged into `rollout_is`, which masks both sides of the band instead of clipping one side.

What `rollout_is` does (`ppo_utils.py::rollout_is_policy_loss`): `loss = -sg(f(r)) * A * log pi`, with `r = pi_trainer / pi_sampler` and `f(r) = r` inside (0.8, 1.28), else 0. PPO-clip against the same sampler gives the same gradient inside the band, and outside it zeroes only the tokens that already moved in the advantage's direction (A > 0 and r > 1.28, A < 0 and r < 0.8). The two-sided mask also zeroes the tokens that moved the wrong way: a bad token (A < 0) whose probability rose above 1.28x, or a good token (A > 0) that fell below 0.8x. Those are exactly the tokens PPO keeps pushing back. Without that restoring gradient, anything that drifts the wrong way, through generalization from other updates or through the 0-4 steps of lag, stays where it drifted, and since the band is tight, a growing share of tokens ends up outside it with no gradient at all. The run shows that loop: the out-of-band share compounds from 0.2% (step 30) to 6% (50), 19% (80) and 60% (93), the trainer-vs-sampler gap grows with it at constant staleness, and entropy and length go after.

Why we think it's this and not SkyRL's pipeline:

- Same pipeline, one-sided trust region, no drift: the recommended arm (DPPO, which for A > 0 masks only tokens pushed more than 0.2 above the sampler, for A < 0 only those pushed 0.2 below, i.e. one-sided like PPO) tracks the shared arm step for step through step 35 (gap 0.009 -> 0.011-0.013, entropy 0.37-0.44) but with no sign of the compounding the shared arm starts at step ~33. It crashed at step 36 (below), so it covers only the start of the window.
- Every other library's shared arm uses a one-sided clip against the sampler on the same config (SHARED_W1 rev. 5) and stays stable, including prime-rl at a higher mean staleness (4.1-4.7).
- SkyRL never pairs `rollout_is` with a tight band. Its fully async DAPO recipe (`examples/train/algorithms/dapo/run_dapo_qwen3_1.7b_aime_fully_async.sh`) uses `eps_clip_low=0.5`, `eps_clip_high=4.0`, i.e. a (0.5, 5.0) mask that only drops extreme mismatches, plus a geometric sequence mask (0.99, 1.01) as the actual trust region, and 160 warmup steps. Our band is 10x tighter on the high side with no sequence mask, which is a usage SkyRL doesn't recommend.
- The pipeline-side suspects don't hold up: staleness is flat, the step-1 numerics are normal, the exports are fine, and the untied-embedding bug moves logprobs by only 0.004-0.014 nats.

What we can't rule out from the logs alone: `token_mean` vs `prompt_mean` also differs between the two SkyRL arms (the shared arm's length starts falling earlier, 9-10k at step 30 vs 12-13k), and the recommended arm died before the window where the shared arm diverged. Hence the A/B below.

## Comparison with the recommended arm (job 365013, DPPO + prompt_mean)

| step | shared: gap / out of band / entropy / len | recommended: gap / masked / entropy / len |
|---|---|---|
| 1 | 0.0087 / 0.03% / 0.38 / 6.2k | 0.0086 / 0% / 0.37 / 6.2k |
| 10 | 0.0104 / 0.10% / 0.37 / 10.2k | 0.0120 / 0.002% / 0.43 / 11.1k |
| 20 | 0.0116 / 0.11% / 0.40 / 10.4k | 0.0130 / 0.001% / 0.41 / 10.7k |
| 30 | 0.0128 / 0.17% / 0.41 / 8.9k | 0.0128 / 0.00002% / 0.44 / 12.5k |
| 35 | 0.0148 / 0.48% / 0.40 / 9.3k | 0.0115 / 0.0004% / 0.39 / 12.6k |

The recommended run then failed at step 36 (06:31 UTC, `EXIT=1`): after the step-35 weight sync, every generate request got HTTP 503 "No available workers" from SkyRL's vllm-router (the router log goes quiet at 06:28:45), and after 30 retries `orjson.loads` on the 503 body raised `JSONDecodeError` in a generation worker, which killed training. Unrelated to the collapse; it needs its own look and a resubmit.

## Fix

Smallest change that keeps the shared-arm semantics (PPO-clip 0.2 / 0.28 against the sampler, one-sided, token mean, no extra masks): register a one-sided PPO loss against the rollout logprobs through SkyRL's public `PolicyLossRegistry` (the mechanism `examples/train/algorithms/custom_policy_loss/` shows) and select it with `trainer.algorithm.policy_loss_type=ppo_rollout`. It's SkyRL's `regular` loss with `rollout_logprobs` in place of the recomputed `old_log_probs`. The name isn't in `LOSSES_WITH_OLD_LOGPROBS`, so the fully async assert accepts it; the only side effect is that SkyRL then runs the old-logprob forward it would skip for `rollout_is` (unused, costs one extra forward per step). No library source is patched. Code: `/fsx/amine_dirhoussi/stable-bench-runs/skyrl/collapse-364970/collapse_main.py` (a wrapper around `launch/skyrl/skyrl_w1_main.py`); configs validated on CPU and the gradient checked on a toy case (A > 0, r = 0.61 keeps its gradient; A < 0, r = 1.65 keeps its gradient; A < 0, r = 0.61 is clipped).

```python
def ppo_rollout_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs):
    ratio = safe_exp_delta(log_probs - rollout_logprobs, clip=20.0, out_dtype=log_probs.dtype)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    return reduce_loss(loss, loss_mask), {"clip_ratio": clip_ratio}

PolicyLossRegistry.register("ppo_rollout", ppo_rollout_policy_loss)
```

Alternatives, both config-only but further from the shared spec: `policy_loss_type=dppo` with `loss_reduction=token_mean` (one-sided, but the band is a 0.2 probability difference, not a ratio), or SkyRL's recipe setting for `rollout_is` (band 0.5 / 4.0 plus the 0.99-1.01 geometric sequence mask, which adds a sequence mask the other libraries don't have).

Separately, the untied-embedding bug should be fixed for both SkyRL arms before the numbers are final: one line, `module.tie_weights()` after `module.to(torch.device("meta"))` in `FSDPStrategy._fsdp_init_model`. That is a library source patch (`patched_lib_source: true`), so it should be recorded as such, or the runs kept as-is with the measured 0.004-0.014 nat effect noted.

The W2 SkyRL setup reuses the `rollout_is` 0.2 / 0.28 mapping, so it carries the same risk.

## A/B check from the step-50 export (queued, not run yet)

`/fsx/amine_dirhoussi/stable-bench-runs/skyrl/collapse-364970/ab.sbatch`, queued through `b200-queue --nodes=1 --gres=gpu:8 --exclusive` (waiter job 365060, `stable-bench-runs/queue/q-collapse-364970-ab-*.out`). As of 06:45 UTC it's waiting on the 4-node cap. Both arms start from `global_step_50/policy` (onset of the drift: 6% out of band, entropy 0.57), run 40 steps each in parallel on one node (2 FSDP GPUs + 2 vLLM engines per arm) with w1-S.overrides unchanged except the loss: A = `rollout_is` 0.2 / 0.28 (shared mapping), B = `ppo_rollout` 0.2 / 0.28. Optimizer state restarts (the export has none), the same for both. Outputs: `collapse-364970/ab-A-rollout_is-<job>/` and `ab-B-ppo_rollout-<job>/` (tb/, train.log); read them with `collapse-364970/rec.py`-style tb parsing.

How to read it: if A's out-of-band share and trainer-vs-sampler gap keep compounding (out of band past ~15%, entropy past ~0.8 within 40 steps, as in steps 50-90 of the original run) while B's gap stays near 0.01-0.02 with steady entropy, the mapping is confirmed as the cause (`ours`). If B drifts the same way, the cause is in SkyRL's pipeline, not the loss, and the ledger row should move to `lib` or `unknown`.

Note that the recommended arm's "masked" column above is DPPO's probability-difference mask, not the ratio band, so only the gap, entropy and length columns compare directly.
