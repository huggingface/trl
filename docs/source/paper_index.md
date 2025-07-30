# Paper Index

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Group Sequence Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2507.18071

GSPO is a GRPO variant that computes importance sampling weights at the sequence level instead of per-token. To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    steps_per_generation=4,  # "each batch of rollout data is partitioned into four mini- batches for gradient updates"
    beta=0.04,  # not explicitly specified in the paper, but they likely used the same value as in the GRPO paper
    epsilon=3e-4,  # "we set the left and right clipping ranges in Equation (5) to 3e-4 and 4e-4, respectively"
    epsilon_high=4e-4,  
)
```
