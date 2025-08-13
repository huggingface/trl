# Quickstart

TRL is a comprehensive library for post-training foundation models using techniques like Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO).

## Quick Examples

Get started instantly with TRL's most popular trainers. Each example uses compact models for quick experimentation.

<div class="trainer-toggle">

### Supervised Fine-Tuning

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

training_args = SFTConfig(
    output_dir="Qwen2.5-0.5B-SFT",
    per_device_train_batch_size=8,
    gradient_checkpointing=True, 
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    logging_steps=10,
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=training_args,
    train_dataset=load_dataset("trl-lib/Capybara", split="train[:1000]"),
)
trainer.train()
trainer.save_model()
```

### Direct Preference Optimization

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# DPO requires an SFT model as base
training_args = DPOConfig(
    output_dir="Qwen2.5-0.5B-DPO",
    per_device_train_batch_size=8,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    beta=0.1,
    logging_steps=10,
)

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Use your SFT model
    ref_model="Qwen/Qwen2.5-0.5B-Instruct",  # Original base model
    args=training_args,
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train_prefs[:500]"),
)
trainer.train()
trainer.save_model()
```

### Group Relative Policy Optimization

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Define a simple reward function (count unique chars as example)
def reward_function(samples):
    return [len(set(sample.lower())) / 10.0 for sample in samples]

training_args = GRPOConfig(
    output_dir="Qwen2.5-0.5B-GRPO",
    per_device_train_batch_size=8,
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    learning_rate=1e-6,
    beta=0.0, 
    epsilon=0.2
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",  # Start from SFT model
    args=training_args,
    train_dataset=load_dataset("trl-lib/tldr", split="train[:500]"),
    reward_function=reward_function,
)
trainer.train()
trainer.save_model()
```

</div>

## Command Line Interface

Skip the code entirely - train directly from your terminal:

```bash
# SFT: Fine-tune on instructions
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-0.5B-SFT

# DPO: Align with preferences  
trl dpo --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --output_dir Qwen2.5-0.5B-DPO
```


## What's Next?

<div class="grid">
  
**📚 Learn More**
- [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer) - Complete SFT guide
- [DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer) - Preference alignment 
- [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer) - Group relative policy optimization
- [Training FAQ](https://huggingface.co/docs/trl/how_to_train) - Common questions

**🚀 Scale Up**
- [Distributed Training](https://huggingface.co/docs/trl/distributing_training) - Multi-GPU setups
- [Memory Optimization](https://huggingface.co/docs/trl/reducing_memory_usage) - Efficient training
- [PEFT Integration](https://huggingface.co/docs/trl/peft_integration) - LoRA and QLoRA

**💡 Examples** 
- [Example Scripts](https://github.com/huggingface/trl/tree/main/examples) - Production-ready code
- [Research Projects](https://github.com/huggingface/trl/tree/main/examples/research_projects) - Advanced techniques
- [Community Tutorials](https://huggingface.co/docs/trl/community_tutorials) - External guides

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
