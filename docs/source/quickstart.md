# Quickstart

TRL is a comprehensive library for post-training foundation models using techniques like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO),  Direct Preference Optimization (DPO).

## Quick Examples

Get started instantly with TRL's most popular trainers. Each example uses compact models for quick experimentation.

### Supervised Fine-Tuning

```python
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

### Group Relative Policy Optimization

```python
from trl import GRPOTrainer
from datasets import load_dataset

# Define a simple reward function (count unique chars as example)
def reward_function(completions, **kwargs):
    return [len(set(completion.lower())) for completion in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Start from SFT model
    train_dataset=load_dataset("trl-lib/tldr", split="train"),
    reward_funcs=reward_function,
)
trainer.train()
```

### Direct Preference Optimization

```python
from trl import DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use your SFT model
    ref_model="Qwen/Qwen2.5-0.5B-Instruct",  # Original base model
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

## Command Line Interface

Skip the code entirely - train directly from your terminal:

```bash
# SFT: Fine-tune on instructions
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara

# DPO: Align with preferences  
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized
```

## What's Next?

### 📚 Learn More

- [SFT Trainer](sft_trainer) - Complete SFT guide
- [DPO Trainer](dpo_trainer) - Preference alignment
- [GRPO Trainer](grpo_trainer) - Group relative policy optimization
- [Training FAQ](how_to_train) - Common questions

### 🚀 Scale Up

- [Distributed Training](distributing_training) - Multi-GPU setups
- [Memory Optimization](reducing_memory_usage) - Efficient training
- [PEFT Integration](peft_integration) - LoRA and QLoRA

### 💡 Examples

- [Example Scripts](https://github.com/huggingface/trl/tree/main/examples) - Production-ready code
- [Community Tutorials](community_tutorials) - External guides

## Troubleshooting

### Out of Memory?

Reduce batch size and enable optimizations:

<hfoptions id="batch_size">
<hfoption id="SFT">

```python
training_args = SFTConfig(
    per_device_train_batch_size=1,  # Start small
    gradient_accumulation_steps=8,  # Maintain effective batch size
)
```

</hfoption>
<hfoption id="DPO">

```python
training_args = DPOConfig(
    per_device_train_batch_size=1,  # Start small
    gradient_accumulation_steps=8,  # Maintain effective batch size
)
```

</hfoption>
</hfoptions>

### Loss not decreasing?

Try adjusting the learning rate:

```python
training_args = SFTConfig(learning_rate=2e-5)  # Good starting point
```

For more help, see our [Training FAQ](how_to_train) or open an [issue on GitHub](https://github.com/huggingface/trl/issues).
