# GFPO

This feature implements the GFPO algorithm to enforce concise reasoning in the model's output generation, as proposed in the paper [Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning](https://huggingface.co/papers/2508.09726).

## Usage

To activate GFPO in [`GFPOTrainer`]:

- set `num_remains_in_group` in [`GFPOConfig`]
- define a group filter function and set it to `group_filter_func` in [`GFPOTrainer`]. `group_filter_func` will score the `num_generations` completions and The GFPOTrainer filters groups according to their scores to get top `num_remains_in_group` completions as a new group. Model will be trained on the filtered group.

```python
# train_gfpo.py
from trl.experimental.gfpo import GFPOConfig, GFPOTrainer

# dummy group filter to scores the completions based on its indice in group
class GroupFilter:
    def __call__(self, group_completions, group_rewards, **kwargs):
        group_scores = []
        for completions, rewards in zip(group_completions, group_rewards):
            scores = [float(i) for i in range(len(completions))]
            group_scores.append(scores)
        return group_scores

training_args = GFPOConfig(
    output_dir="Qwen3-0.6B-GFPO",
    per_device_train_batch_size=4,
    num_remains_in_group=2,
    bf16=True,
)
trainer = GFPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=...,
    train_dataset=...,
    args=training_args,
    group_filter_func=GroupFilter(),
)
trainer.train()
```

## GFPOTrainer

[[autodoc]] experimental.gfpo.GFPOTrainer
    - train
    - save_model
    - push_to_hub

## GFPOConfig

[[autodoc]] experimental.gfpo.GFPOConfig
