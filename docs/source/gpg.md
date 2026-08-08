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

A group whose completions all receive the same reward has a zero advantage, so it contributes nothing to the gradient. Its tokens nevertheless count toward the loss denominator, which scales the update down by the fraction of such degenerate groups. GPG divides the loss by the fraction of groups that are *not* degenerate, restoring the magnitude the informative groups should have produced.

[`GRPOTrainer`] already logs that quantity as `frac_reward_zero_std`, the fraction of completions whose group reward standard deviation is zero, so the correction factor is its complement. Set `bias_correction=False` to recover the uncorrected GRPO gradient magnitude.

Because the correction only rescales the loss, it interacts with the learning rate: a run where most groups are degenerate takes proportionally larger steps than plain GRPO would on the same batch.

## Relation to GRPO

GPG's other properties are already expressible with [`GRPOConfig`], which is why [`GPGConfig`] changes only defaults:

| GPG property | How it is expressed |
| --- | --- |
| no KL constraint, no reference model | `beta=0.0` |
| mean-centered advantage, no std scaling | `scale_rewards="none"` |
| no surrogate, no clipping | `num_iterations=1` (the default) |
| gradient-bias correction | `bias_correction=True` |

With `num_iterations=1` the GRPO surrogate is gradient-identical to the plain policy gradient GPG writes down: the importance ratio is exactly one at the point of evaluation and clipping around one is inert, so both reduce to  \\( \hat{A} \nabla_\theta \log \pi_\theta \\). Raising `num_iterations` above 1 makes the objective genuinely off-policy and departs from the method as published.

## GPGTrainer

[[autodoc]] experimental.gpg.GPGTrainer
    - train
    - save_model
    - push_to_hub

## GPGConfig

[[autodoc]] experimental.gpg.GPGConfig
