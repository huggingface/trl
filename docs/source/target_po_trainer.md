# TargetPO Trainer

## Overview

[`TargetPOTrainer`] implements Target Policy Optimization (TPO), an online post-training algorithm from [Target Policy Optimization](https://huggingface.co/papers/2604.06159).

[`TargetPOTrainer`] keeps an aligned copy of the online rollout and reward flow used by [`GRPOTrainer`], but trains
with a sequence-level cross-entropy target:

$$
q_i = \frac{p_i^{\text{old}} \exp(u_i / \eta)}{\sum_j p_j^{\text{old}} \exp(u_j / \eta)}
$$

Here \\(p_i^{\text{old}}\\) is a *length-normalized* proxy for the rollout policy probability of completion \\(i\\) in
the prompt group (per-token mean log-probability by default, controlled by `tpo_length_normalize_logps`), \\(u_i\\) is
the population-whitened group reward, and \\(\eta\\) is `tpo_target_temperature`. Length-normalization prevents the
old-policy term from dominating the target when completions in a group have different lengths; set
`tpo_length_normalize_logps=False` to recover the paper's literal sequence-probability formulation.

## Quick Start

```python
from datasets import load_dataset
from trl import TargetPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

trainer = TargetPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
```

## Configuration

[`TargetPOConfig`] inherits the online rollout, reward, vLLM, tool-calling, and logging arguments from [`GRPOConfig`]. Because TargetPO uses a sequence-level softmax over every completion in a prompt group, [`TargetPOConfig`] defaults `steps_per_generation` to `1` when the user does not specify a generation schedule. Larger values are supported as long as each optimization step still contains whole prompt groups, i.e. `(generation_batch_size // steps_per_generation) % num_generations == 0`.

## TargetPOConfig

[[autodoc]] TargetPOConfig

## TargetPOTrainer

[[autodoc]] TargetPOTrainer
