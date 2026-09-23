# KLPO

In the report [KL-Regularized Policy Optimization for Critic-Free Agentic Reinforcement Learning](https://yifanzhang-pro.github.io/KLPO/), the authors propose KLPO, a critic-free, single-rollout method for off-policy reinforcement learning. One complete response per prompt is sufficient: terminal rewards provide the feedback, and independent auxiliary token draws estimate the sampler-conditioned score correction. It needs no same-prompt response group or learned value/normalizer model, and there is no importance-ratio multiplier, reward centering, or ratio clipping.

This trainer implements the report's default route: **token regression + Monte Carlo KL (MC-KL)**. Let \\( p \\) be the current policy, \\( q \\) the sampler (behavior) policy, and \\( R \\) the terminal reward. At each visited prefix, \\( M \\) auxiliary tokens \\( v_j \sim q \\) are drawn IID with replacement, independently of the rollout. The loss is

$$
\ell_u = \log p(a_u) - \log q(a_u), \qquad
z_u = \log p(a_u) - \frac{1}{M}\sum_{j=1}^{M} \log p(v_j),
$$

$$
\mathcal{L} = -\underset{\text{responses}}{\text{mean}} \sum_{\text{tokens } u} \operatorname{sg}\!\left[R - \beta\,\ell_u\right] z_u,
$$

where tokens are summed **without length normalization** and averaged over complete responses. The KL regularization is toward the sampler policy (coefficient `klpo_beta`), not a reference model, so no reference model is loaded.

To use KLPO, you can use the [`experimental.klpo.KLPOTrainer`] class in `trl.experimental.klpo`.

## Usage

```python
from trl.experimental.klpo import KLPOConfig, KLPOTrainer

training_args = KLPOConfig(
    num_generations=1,  # single rollout per prompt, the KLPO default
    klpo_beta=0.1,  # KL coefficient toward the sampler policy
    mc_samples=128,  # M auxiliary token draws per prefix for MC-KL
)
trainer = KLPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

In this implementation, the sampler \\( q \\) is the training policy snapshot at generation time: right after generation, the trainer scores the completions with the current model, draws the \\( M \\) auxiliary tokens per prefix from that (temperature-scaled) distribution, and keeps those records fixed while the batch is reused (`num_iterations > 1` or gradient accumulation). The group-relative advantages computed by [`GRPOTrainer`] are ignored in favor of the raw terminal reward, so `num_generations=1` is allowed (and is the default).

> [!WARNING]
> KLPO expects each row to be a complete trajectory carrying a terminal reward, so `mask_truncated_completions` defaults to `True`. If your model rarely terminates within `max_completion_length`, most completions will be masked out; either increase `max_completion_length` or set `mask_truncated_completions=False`.

KLPOTrainer does not yet support vision-language models or the Liger kernel.

## KLPOTrainer

[[autodoc]] experimental.klpo.KLPOTrainer
    - train
    - save_model
    - push_to_hub

## KLPOConfig

[[autodoc]] experimental.klpo.KLPOConfig
