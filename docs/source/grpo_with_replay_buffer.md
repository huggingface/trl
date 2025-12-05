# GRPO With Replay Buffer

This experimental trainer, trains a model with GRPO but replaces groups (and corresponding completions) that have 0 standard deviation with groups with high rewards and standard deviation that've been used to train a model in prior batches.

## Usage

```python
import torch
from trl.experimental.grpo_with_replay_buffer import GRPOWithReplayBufferConfig, GRPOWithReplayBufferTrainer
from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

# Guarantee that some rewards have 0 std
def custom_reward_func(completions, **kwargs):
    if torch.rand(1).item() < 0.25:
        return [0] * len(completions)  # simulate some None rewards
    else:
        return torch.rand(len(completions)).tolist()

training_args = GRPOWithReplayBufferConfig(
    output_dir="./tmp",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_generations=4,
    max_completion_length=8,
    replay_buffer_size=8,
    report_to="none",
)

trainer = GRPOWithReplayBufferTrainer(
    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    reward_funcs=[custom_reward_func],
    args=training_args,
    train_dataset=dataset,
)

previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()
```

## GRPOWithReplayBufferTrainer

[[autodoc]] experimental.grpo_with_replay_buffer.GRPOWithReplayBufferTrainer
    - train
    - save_model
    - push_to_hub

## GRPOWithReplayBufferConfig

[[autodoc]] experimental.grpo_with_replay_buffer.GRPOWithReplayBufferConfig

## ReplayBuffer

[[autodoc]] experimental.grpo_with_replay_buffer.ReplayBuffer
