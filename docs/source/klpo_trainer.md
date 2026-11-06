# KLPO

In the report [KL-Regularized Policy Optimization for Critic-Free Agentic Reinforcement Learning](https://yifanzhang-pro.github.io/KLPO/), the authors propose KLPO, a critic-free, single-rollout method for off-policy reinforcement learning. One complete response per prompt is sufficient: terminal rewards provide the feedback, and independent auxiliary token draws estimate the sampler-conditioned score correction. It needs no same-prompt response group or learned value/normalizer model, and there is no importance-ratio multiplier, reward centering, or ratio clipping.

The default configuration is the report's **token regression + Monte Carlo KL (MC-KL)** route; all eight route/estimator combinations from the report are selectable (see below). Let \\( p \\) be the current policy, \\( q \\) the sampler (behavior) policy, and \\( R \\) the terminal reward. At each visited prefix, \\( M \\) auxiliary tokens \\( v_j \sim q \\) are drawn IID with replacement, independently of the rollout. The default loss is

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

In this implementation, the sampler \\( q \\) is the training policy snapshot at generation time: right after generation, the trainer scores the completions with the current model, extracts the estimator's records from that (temperature-scaled) distribution, and keeps those records fixed while the batch is reused (`num_iterations > 1` or gradient accumulation). The group-relative advantages computed by [`GRPOTrainer`] are ignored in favor of the raw terminal reward, so `num_generations=1` is allowed (and is the default).

## Regression routes and KL estimators

All eight combinations from the report are implemented. The regression route (`klpo_route`) selects the feedback coefficient; the KL estimator (`kl_estimator`) selects the sampler-conditioned score correction.

| Route                | KL estimator     | Configuration                                             | Feedback                                        |
| -------------------- | ---------------- | --------------------------------------------------------- | ----------------------------------------------- |
| Token (default)      | MC-KL (default)  | no flags; `mc_samples=128` by default, \\( M \geq 1 \\)   | per-token \\( R - \beta\,\ell_u \\)             |
| Token                | TopK-KL          | `kl_estimator="topk"`, `kl_top_k=128`                     | per-token \\( R - \beta\,\ell_u \\)             |
| Token                | Binary KL        | `kl_estimator="binary"`                                   | per-token \\( R - \beta\,\ell_u \\)             |
| Token                | Full KL          | `kl_estimator="full"`                                     | per-token \\( R - \beta\,\ell_u \\)             |
| Sequence             | MC-KL            | `klpo_route="sequence"`, \\( M \geq 2 \\)                 | leave-one-out trajectory residuals              |
| Sequence             | TopK-KL          | `klpo_route="sequence"`, `kl_estimator="topk"`            | trajectory residual \\( D_K \\)                 |
| Sequence             | Binary KL        | `klpo_route="sequence"`, `kl_estimator="binary"`          | trajectory residual \\( D \\)                   |
| Sequence             | Full KL          | `klpo_route="sequence"`, `kl_estimator="full"`            | trajectory residual \\( D \\)                   |

- **MC-KL** averages the log-ratios of \\( M \\) independent sampler draws with replacement, without a head, tail bucket, or probability reweighting. Sequence regression requires \\( M \geq 2 \\) (leave-one-out residuals).
- **TopK-KL** keeps the sampler's \\( K \\) highest-probability tokens per prefix (`kl_top_k`, default 128) and aggregates all remaining tokens into one tail bucket (both tail masses floored at `1e-6`).
- **Binary KL** groups the sampled action against its complement. It needs only sampled-action log-probabilities (no extra records) and requires strictly negative active log-probabilities.
- **Full KL** uses the entire vocabulary. It stores the sampler's full `(B, T, vocab_size)` conditionals per generation batch, which is memory-intensive for real vocabularies; prefer `"topk"` with a large `kl_top_k` in that case.

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
