# TPO Trainer

## Overview

[`TPOTrainer`] implements Target Policy Optimization (TPO), an online post-training algorithm from [Target Policy Optimization](https://huggingface.co/papers/2604.06159).

TPO reuses the [`GRPOTrainer`] rollout and reward stack, but trains with a sequence-level cross-entropy target:

$$
q_i = \frac{p_i^{\text{old}} \exp(u_i / \eta)}{\sum_j p_j^{\text{old}} \exp(u_j / \eta)}
$$

Here \\(p_i^{\text{old}}\\) is the rollout policy probability of completion \\(i\\) in the prompt group, \\(u_i\\) is the normalized reward, and \\(\eta\\) is `tpo_target_temperature`.

## Quick Start

```python
from datasets import load_dataset
from trl import TPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = TPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

## Configuration

[`TPOConfig`] inherits the online rollout, reward, vLLM, tool-calling, and logging arguments from [`GRPOConfig`]. Because TPO uses a sequence-level softmax over every completion in a prompt group, [`TPOConfig`] sets `steps_per_generation=1` by default.

## TPOConfig

[[autodoc]] TPOConfig

## TPOTrainer

[[autodoc]] TPOTrainer
