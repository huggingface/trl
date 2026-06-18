# GMPO

In the paper [Geometric-Mean Policy Optimization](https://huggingface.co/papers/2507.20673), the authors propose a GRPO variant that maximizes the *geometric* mean of the token-level importance ratios instead of the arithmetic mean. Because the geometric mean is far less sensitive to outlier ratios, the policy update is more stable and tolerates a much wider clipping range. Clipping is applied per token, in log space, and one-sided per the advantage sign (the standard PPO trust region) — crucially, *before* the geometric mean is taken.

To use GMPO, you can use the [`GMPOTrainer`] class in `trl.experimental.gmpo`.

## Usage

```python
from trl.experimental.gmpo import GMPOConfig, GMPOTrainer

training_args = GMPOConfig(
    epsilon=0.4,  # log-space clip range -> ratios clipped to (exp(-0.4), exp(0.4)); paper, Sec. 4
    beta=0.0,
)
trainer = GMPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

In GMPO, clipping is applied to the per-token *log*-importance ratios (i.e. in log space) before the geometric mean is taken, so `epsilon` and `epsilon_high` are expressed in log space: the effective ratio clipping range is `(exp(-epsilon), exp(epsilon_high))`. The paper recommends a markedly wider range than GRPO/DAPO, `(exp(-0.4), exp(0.4))`, to encourage exploration.

## GMPOTrainer

[[autodoc]] experimental.gmpo.GMPOTrainer
    - train
    - save_model
    - push_to_hub

## GMPOConfig

[[autodoc]] experimental.gmpo.GMPOConfig
