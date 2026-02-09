# Quickstart

TRL is a comprehensive library for post-training foundation models using techniques like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO).

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
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Start from SFT model
    train_dataset=load_dataset("trl-lib/DeepMath-103K", split="train"),
    reward_funcs=accuracy_reward,
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

### Reward Modeling

```python
from trl import RewardTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = RewardTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
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

# Reward: Train a reward model
trl reward --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized
```

## What's Next?

### 📚 Learn More

- [SFT Trainer](sft_trainer) - Complete SFT guide
- [DPO Trainer](dpo_trainer) - Preference alignment
- [GRPO Trainer](grpo_trainer) - Group relative policy optimization

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

For more help, open an [issue on GitHub](https://github.com/huggingface/trl/issues).
