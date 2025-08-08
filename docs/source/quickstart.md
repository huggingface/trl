# Quickstart

TRL is a comprehensive library for post-training foundation models using techniques like Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Proximal Policy Optimization (PPO).

## Installation

```bash
pip install trl
```

## Quick Examples

Get started instantly with TRL's most popular trainers. Each example runs on a single GPU and uses compact models for quick experimentation.

<div class="trainer-toggle">

### SFT Trainer

**Supervised Fine-Tuning** - Perfect for instruction-following and chat models

```python
from trl import SFTTrainer
from datasets import load_dataset

# Minimal example - just 3 lines!
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train[:1000]"),
)
trainer.train()
```

### DPO Trainer  

**Direct Preference Optimization** - Align models using human preferences

```python
from trl import DPOTrainer
from datasets import load_dataset

# Train on preference data
trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B", 
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train_prefs[:1000]"),
)
trainer.train()
```

### GRPO Trainer

**Group Relative Policy Optimization** - Memory-efficient alternative to PPO

```python
from trl import GRPOTrainer
from datasets import load_dataset

# RLHF without a separate reward model
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=load_dataset("trl-lib/tldr", split="train[:1000]"),
)
trainer.train()
```

</div>

## Command Line Interface

Skip the code entirely - train directly from your terminal:

```bash
# SFT: Fine-tune on instructions
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir ./my-sft-model

# DPO: Align with preferences  
trl dpo --model_name_or_path ./my-sft-model \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir ./my-aligned-model
```

## Configuration Examples

Each trainer can be customized with detailed configurations for production use:

<details>
<summary><strong>Advanced SFT Configuration</strong></summary>

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

config = SFTConfig(
    output_dir="./sft-qwen-chat",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=1000,
    logging_steps=10,
    save_steps=500,
    bf16=True,  # Use bfloat16 for efficiency
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=config,
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```
</details>

<details>
<summary><strong>Advanced DPO Configuration</strong></summary>

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

config = DPOConfig(
    output_dir="./dpo-aligned-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,  # KL penalty strength
    max_steps=500,
    bf16=True,
)

trainer = DPOTrainer(
    model="./sft-qwen-chat",  # Use SFT model as base
    args=config,
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train_prefs"),
)
trainer.train()
```
</details> 

## What's Next?

<div class="grid">
  
**📚 Learn More**
- [SFT Trainer](sft_trainer.md) - Complete SFT guide
- [DPO Trainer](dpo_trainer.md) - Preference alignment 
- [Training FAQ](how_to_train.md) - Common questions

**🚀 Scale Up**
- [Distributed Training](distributing_training.md) - Multi-GPU setups
- [Memory Optimization](reducing_memory_usage.md) - Efficient training
- [PEFT Integration](peft_integration.md) - LoRA and QLoRA

**💡 Examples** 
- [Example Scripts](../examples/) - Production-ready code
- [Research Projects](../examples/research_projects/) - Advanced techniques
- [Community Tutorials](community_tutorials.md) - External guides

</div>

## Troubleshooting

**Out of Memory?** Reduce batch size and enable optimizations:
```python
config = SFTConfig(
    per_device_train_batch_size=1,  # Start small
    gradient_accumulation_steps=8,  # Maintain effective batch size
    bf16=True,                      # Use mixed precision
    gradient_checkpointing=True,    # Save memory
)
```

**Loss not decreasing?** Try adjusting the learning rate:
```python 
config = SFTConfig(learning_rate=2e-5)  # Good starting point
```

For more help, see our [Training FAQ](how_to_train.md) or open an [issue on GitHub](https://github.com/huggingface/trl/issues).
