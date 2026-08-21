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

The factor is the fraction of completions whose advantage is non-zero, read once per generation batch. A completion's advantage is zeroed in the two cases the correction must discount: every member of a group whose rewards are identical loses it to the group-mean subtraction, and an unscorable completion (one where every reward function returned `None`) is zeroed explicitly. `frac_reward_zero_std`, which [`GRPOTrainer`] logs, sees only the first of those and collapses to 0 or 1 under a batch-wide standard deviation, so it is not the source. Set `bias_correction=False` to recover the uncorrected GRPO gradient magnitude.

The factor counts completions, so it cancels the denominator only when the denominator is also a completion count. [`GPGConfig`] therefore defaults `loss_type` to `"grpo"`, which averages one token-mean per completion, instead of inheriting GRPO's `"dapo"`, which divides by a completion-token total. With `bias_correction=True` the trainer rejects the settings that break that identity rather than applying a factor that would be silently wrong:

| Rejected setting | Why |
| --- | --- |
| `loss_type` outside `"grpo"`, `"sapo"`, `"dr_grpo"`, `"luspo"` | those four divide by something proportional to a completion count, which the factor cancels exactly; `"bnpo"`, `"dapo"`, `"cispo"` and `"vespo"` divide by a completion-token total, which it cancels only when every group emits the same number of tokens |
| `beta != 0.0` | the KL term is added to the loss before the correction divides it, and KL does not vanish for a group whose rewards are identical, so its effective coefficient would become `beta / fraction` |
| `multi_objective_aggregation="normalize_then_sum"` | it centers on the batch mean rather than each group's, so a group whose rewards are identical keeps a non-zero advantage and cannot be identified |
| `use_liger_kernel=True` | `compute_loss` routes to `compute_liger_loss` and never reaches the override that applies the factor |
| entropy bonus or MoE router auxiliary loss | both are added to the loss before the correction divides it, so their effective coefficients would move with the reward spread |

Because the correction only rescales the loss, it interacts with the learning rate: a run where most groups are degenerate takes proportionally larger steps than plain GRPO would on the same batch.

## Relation to GRPO

GPG's other properties are already expressible with [`GRPOConfig`], which is why [`GPGConfig`] changes only defaults:

| GPG property | How it is expressed |
| --- | --- |
| no KL constraint, no reference model | `beta=0.0` |
| mean-centered advantage, no std scaling | `scale_rewards="none"` |
| no surrogate, no clipping | `num_iterations=1` (the default) |
| per-completion loss normalizer | `loss_type="grpo"` |
| gradient-bias correction | `bias_correction=True` |

With `num_iterations=1` the GRPO surrogate is gradient-identical to the plain policy gradient GPG writes down: the importance ratio is exactly one at the point of evaluation and clipping around one is inert, so both reduce to  \\( \hat{A} \nabla_\theta \log \pi_\theta \\). Raising `num_iterations` above 1 makes the objective genuinely off-policy and departs from the method as published.

## GPGTrainer

[[autodoc]] experimental.gpg.GPGTrainer
    - train
    - save_model
    - push_to_hub

## GPGConfig

[[autodoc]] experimental.gpg.GPGConfig
