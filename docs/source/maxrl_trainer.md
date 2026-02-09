# MaxRL Trainer

[![](https://img.shields.io/badge/arxiv-2602.02710-b31b1b.svg)](https://arxiv.org/abs/2602.02710)

TRL supports the Maximum Likelihood Reinforcement Learning (MaxRL) training method with the [`MaxRLTrainer`]. MaxRL is a policy optimization algorithm similar to GRPO but uses a different advantage normalization strategy called **p-normalization**, which helps prevent bias towards questions with different difficulty levels.

For a detailed description of the algorithm, check out the paper [Maximum Likelihood Reinforcement Learning](https://huggingface.co/papers/2602.02710) by Tajwar et al. (2026).

## Quick start

This example demonstrates how to train a model using the MaxRL method. We use the GSM8K math dataset as an example.

```python
from datasets import load_dataset
from trl import MaxRLConfig, MaxRLTrainer
from trl.rewards import accuracy_reward

# Load the dataset
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# Configure the training
training_args = MaxRLConfig(
    output_dir="maxrl_model",
    num_generations=8,
    learning_rate=1e-6,
    max_completion_length=256,
)

# Initialize the trainer
trainer = MaxRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model(training_args.output_dir)
```

Executing the command above will start training a model using MaxRL. The model will generate multiple completions for each prompt and use the rewards to update the policy with the MaxRL objective.

## How MaxRL works

MaxRL follows a similar training procedure to GRPO:

1. **Generation**: For each prompt in the training batch, generate multiple completions (controlled by `num_generations`)
2. **Reward Calculation**: Compute rewards for each completion using the provided reward function(s)
3. **Advantage Calculation**: Calculate advantages using MaxRL's p-normalization:
   ```
   advantage = (reward - mean_reward) / (mean_reward + epsilon)
   ```
   Unlike GRPO which divides by the standard deviation, MaxRL divides by the mean reward. This is called p-normalization.
4. **Policy Update**: Update the model using a clipped policy gradient objective with the computed advantages

### Key difference from GRPO

The main algorithmic difference between MaxRL and GRPO is in the advantage calculation:

**GRPO** (standard normalization):
```
advantage = (reward - mean_reward) / (std_reward + epsilon)
```

**MaxRL** (p-normalization):
```
advantage = (reward - mean_reward) / (mean_reward + epsilon)
```

This normalization by mean reward helps prevent bias towards questions with different difficulty levels. For example, if easy questions have rewards around 0.9 and hard questions have rewards around 0.1, standard normalization would give both similar advantage magnitudes. With p-normalization, the advantages are scaled by the question difficulty, providing more balanced learning signals.

## Expected dataset format

The MaxRL trainer expects datasets in the same format as GRPO. The dataset should contain a column `"prompt"` with the prompts. Any additional columns in the dataset are forwarded to the reward functions and can be used to compute custom rewards.

### Example dataset

Here is an example of a sample from a dataset formatted for MaxRL training:

```python
{
    "prompt": "What is 2 + 2?",
    "solution": "4"  # Used by accuracy_reward to verify the answer
}
```

## Using custom reward functions

You can use custom reward functions with MaxRL. The reward function should:
- Accept `prompts` and `completions` as arguments
- Return a list of reward values (one per completion)
- Optionally return `None` for samples where the reward is not applicable

Here's an example:

```python
def custom_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Compute your custom reward logic here
        reward = compute_reward(prompt, completion)
        rewards.append(reward)
    return rewards

trainer = MaxRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=custom_reward,  # Use your custom reward function
    args=training_args,
    train_dataset=dataset,
)
```

You can also combine multiple reward functions:

```python
trainer = MaxRLTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[accuracy_reward, length_penalty, custom_reward],
    args=training_args,
    train_dataset=dataset,
)
```

## MaxRLConfig

[[autodoc]] MaxRLConfig

## MaxRLTrainer

[[autodoc]] MaxRLTrainer

## Logged metrics

During training, MaxRL logs the following metrics:

| Metric | Description |
|--------|-------------|
| `reward` | Mean reward across all completions |
| `reward_std` | Standard deviation of rewards |
| `frac_reward_zero_mean` | Fraction of samples with near-zero mean reward (potential numerical instability indicator) |
| `rewards/{reward_func_name}/mean` | Mean reward for each reward function |
| `rewards/{reward_func_name}/std` | Standard deviation for each reward function |
| `completions/mean_length` | Average completion length |
| `completions/clipped_ratio` | Fraction of completions that were truncated |
| `kl` | KL divergence between current and reference policy (if `beta > 0`) |
| `clip_ratio/region_mean` | Fraction of tokens where the importance sampling ratio was clipped |
| `entropy` | Entropy of the token distribution |

## Tips and best practices

1. **Start with GRPO hyperparameters**: Since MaxRL is similar to GRPO, you can start with GRPO hyperparameters and adjust as needed.

2. **Monitor mean rewards**: Keep an eye on the `frac_reward_zero_mean` metric. If this is high, it indicates many prompts have near-zero mean rewards, which can cause numerical instability with p-normalization.

3. **Use with math/reasoning tasks**: MaxRL was designed for mathematical reasoning tasks where question difficulty varies significantly. It may provide benefits over GRPO in such scenarios.

4. **Generation settings**: The `num_generations` parameter controls how many completions are generated per prompt. The original paper uses 8-128 generations depending on the task.

5. **Learning rate**: The paper recommends a learning rate of 1e-6 for most models.

## Differences from the original implementation

The TRL implementation of MaxRL is based on the paper [Maximum Likelihood Reinforcement Learning](https://arxiv.org/abs/2602.02710) and aims to be as close as possible to the original implementation from the [maxrl repository](https://github.com/tajwarfahim/maxrl). Key implementation details:

- **p-normalization**: Uses `(reward - mean_reward) / (mean_reward + epsilon)` for advantage calculation
- **Integration with GRPO**: Built as a subclass of `GRPOTrainer`, inheriting all GRPO features including vLLM support, multi-GPU training, and tool use
- **Compatible configurations**: All GRPO configuration options work with MaxRL

## Resources

- [Maximum Likelihood Reinforcement Learning paper](https://arxiv.org/abs/2602.02710)
- [Official MaxRL repository](https://github.com/tajwarfahim/maxrl)
- [GRPO documentation](grpo_trainer.md) (MaxRL extends GRPO)
