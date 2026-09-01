# GPG

In the paper [GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](https://huggingface.co/papers/2504.02546), the authors propose a minimalist GRPO variant that drops the critic, the reference model and the KL constraint, and optimizes the policy-gradient objective directly instead of a surrogate. What remains beyond those simplifications is a correction for the gradient bias introduced by groups whose completions all receive the same reward.

To use GPG, you can use the [`GPGTrainer`] class in `trl.experimental.gpg`.

## Usage

```python
from trl.experimental.gpg import GPGConfig, GPGTrainer

training_args = GPGConfig(
    beta=0.0,  # no KL constraint and no reference model; the GPG default
    scale_rewards="none",  # mean-centered advantage, no std scaling; the GPG default
)
trainer = GPGTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

## The gradient-bias correction

A group whose completions all receive the same reward has a zero advantage, so it contributes nothing to the gradient. It nevertheless counts toward the loss denominator, which scales the update down by the fraction of such degenerate groups. GPG divides the loss by the fraction of groups that are *not* degenerate, restoring the magnitude the informative groups should have produced.

This is paper eq. 7, whose multiplier is \\( \alpha = B / (B - M) \\), with \\( M \\) counting the samples that belong to groups whose responses are all right or all wrong. The trainer stores the reciprocal \\( (B - M) / B \\), the fraction of groups whose advantages carry some spread, and divides by it; that is the same operation written the other way around, and it is how the authors' implementation stores it too. Invalidity is a property of the *group*, not of a single completion, so the fraction is read once per generation batch by counting groups. A whole group's advantages collapse to a single shared value in the two cases the correction must discount: the group-mean subtraction leaves every member of a group whose rewards are identical on the same number, and a group no reward function could score (every reward function returned `None` for every member) is zeroed explicitly. That shared value is zero in exact arithmetic but not in float, because every member subtracts the same mean from the same reward and so lands on the same rounding residual rather than on zero. Judging the spread rather than the distance from zero is what makes the count right: over 20000 fully degenerate groups at `num_generations=8` with continuous rewards, a zero test counts 44.1% of them as informative on residuals as large as `9.5e-07`. Entries that are exactly zero are left out of the spread, because such an entry is a completion no reward function could score and carries no signal by construction. Counting it would misread the mixed case: rewards `[2.9, 2.9, 2.9, None]` leave advantages `[-2.4e-07, -2.4e-07, -2.4e-07, 0.0]`, a degenerate group that the lone exact zero would otherwise make look informative, and under `scale_rewards="group"` that residual grows to `-2.4e-03`. Scoring each completion on its own would over-correct instead: with a continuous reward, rewards `[0, 1, 2]` center to `[-1, 0, 1]`, which a per-completion rule reads as 2 valid out of 3 and corrects by `3/2`, where the paper leaves the group intact and corrects by `1`. `frac_reward_zero_std`, which [`GRPOTrainer`] logs, is close but not equal, because it is derived from the reward standard deviation rather than from the advantages. A group no reward function could score has a NaN standard deviation, which is not close to zero, so the metric counts that group valid although its advantages carry no spread: on rewards `[1, 2, 3]` followed by an unscorable group, `1 - frac_reward_zero_std` is `1.0` where the correct factor is `0.5`. The metric also turns into a batch-wide statistic that collapses to 0 or 1 when `scale_rewards="batch"`. Reading the advantages is exact in both cases. Set `bias_correction=False` to recover the uncorrected GRPO gradient magnitude.

With `bias_correction=True` the trainer rejects the settings under which the multiplier would be applied to something it cannot correct:

| Rejected setting | Why |
| --- | --- |
| `loss_type="luspo"` | it sums each completion's token losses rather than averaging them, so the loss scales with completion length and the factor cannot cancel it at any length: on two completions of length 3 with a unit per-token loss it returns `3.0` where `"grpo"` returns `1.0`. Every other reduction is accepted; see the paragraph below for which ones keep the correction exact |
| `mask_truncated_completions=True`, `top_entropy_quantile<1.0`, `off_policy_mask_threshold`, `use_vllm=True` under a masking `vllm_importance_sampling_mode` | each silences a completion's token losses while leaving its advantage intact, so the completion contributes no gradient yet keeps its slot in the loss denominator. That is the dilution the correction exists to undo, and it cannot see it, because it reads only the advantages. The vLLM one is the easiest to hit by accident: `vllm_importance_sampling_correction` defaults to `True` and its mode defaults to `"sequence_mask"`, so `use_vllm=True` alone enables it |
| `beta != 0.0` | the KL term is added to the loss before the correction divides it, and KL does not vanish for a group whose rewards are identical, so its effective coefficient would become `beta / fraction` |
| `multi_objective_aggregation="normalize_then_sum"` | it normalizes per group and then subtracts a batch mean that is only zero up to floating-point error, so a degenerate group keeps a residual advantage instead of a clean zero and is counted as informative. Over 2000 randomized batches the residual was non-zero in 1745 of them, peaking at `2.5e-07` |
| `use_liger_kernel=True` | `compute_loss` routes to `compute_liger_loss` and never reaches the override that applies the factor |
| entropy bonus or MoE router auxiliary loss | both are added to the loss before the correction divides it, so their effective coefficients would move with the reward spread. [`GPGConfig`] defaults `router_aux_loss_coef` to `0.0` for that reason, so a Mixture-of-Experts model trains under GPG without further configuration |

Eq. 5 normalizes by the total completion-token count, which is what `"bnpo"` and `"dapo"` do and what the authors' implementation uses. Against that denominator, a multiplier counting completions is exact only when every completion has the same length, so the paper's own eq. 5 and eq. 7 do not compose exactly. Against `"grpo"` and `"sapo"`, which average one token-mean per completion, the same multiplier is exact at any lengths. [`GPGConfig`] therefore defaults to `"grpo"`, trading literal fidelity to eq. 5 for a correction that does what it claims. Set `loss_type="bnpo"` to reproduce the published objective instead; the correction still applies, it is just approximate on ragged batches.

Because the correction only rescales the loss, it interacts with the learning rate: a run where most groups are degenerate takes proportionally larger steps than plain GRPO would on the same batch.

One numeric case cannot be decided from the advantages at all. If a group's float32 rewards are large enough for their mean to overflow, `nan_to_num` saturates every advantage in the row to the same value, and a degenerate group becomes bit-for-bit identical to an informative one: rewards `[3.4e38, 3.3e38]` and rewards `[3.4e38, 3.4e38]` both produce `[-3.4028235e+38, -3.4028235e+38]`. No rule reading the advantages can separate them, because the information that distinguished them is already gone. This is recorded rather than guarded, since a guard would have nothing to test.

One published component is deliberately left out. Alongside the correction the paper thresholds the valid-group proportion at \\( \beta_{th} \\) and accumulates valid samples into the next resampled batch whenever the proportion falls below it, which curbs exactly the variance the paragraph above describes. This trainer applies the correction alone, so a batch where nearly every group is degenerate still takes one large, high-variance step.

## Relation to GRPO

GPG's other properties are already expressible with [`GRPOConfig`], which is why [`GPGConfig`] changes only defaults:

| GPG property | How it is expressed |
| --- | --- |
| no KL constraint, no reference model | `beta=0.0` |
| mean-centered advantage, no std scaling | `scale_rewards="none"` |
| no surrogate, no clipping | `num_iterations=1` (the default) |
| per-completion loss normalizer | `loss_type="grpo"` (a deliberate departure from eq. 5, see above) |
| gradient-bias correction | `bias_correction=True` |

With `num_iterations=1` the GRPO surrogate is gradient-identical to the plain policy gradient GPG writes down: the importance ratio is exactly one at the point of evaluation and clipping around one is inert, so both reduce to  \\( \hat{A} \nabla_\theta \log \pi_\theta \\). Raising `num_iterations` above 1 makes the objective genuinely off-policy and departs from the method as published.

## GPGTrainer

[[autodoc]] experimental.gpg.GPGTrainer
    - train
    - save_model
    - push_to_hub

## GPGConfig

[[autodoc]] experimental.gpg.GPGConfig
