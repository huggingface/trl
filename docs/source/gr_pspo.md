# GR-PSPO

In the paper [It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL](https://arxiv.org/abs/2509.21282), the authors propose using probability smoothing to the behaviour policy instead of clipping (Probability Smoothing Policy Optimisation; PSPO), and apply this to GRPO. To use PSPO with GRPO, you can use the `GRPOTrainer` and set the `trust_region_method` to `pspo`, and set the smoothing strength using `smoothing_alpha` in the `GRPOConfig`.

## Usage

```python
from trl import GRPOTrainer, GRPOConfig

training_args = GRPOConfig(
    trust_region_method="pspo",
    smoothing_alpha=0.1,
    ...
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=training_args,
    ...
)
trainer.train()
```

