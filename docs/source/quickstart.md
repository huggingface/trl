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

**Supervised Fine-Tuning** - Transform base models into instruction-following assistants

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Real-world SFT with proper configuration
config = SFTConfig(
    output_dir="./qwen-chat-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    max_steps=500,
    bf16=True,
    logging_steps=10,
    save_steps=250,
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=config,
    train_dataset=load_dataset("trl-lib/Capybara", split="train[:1000]"),
    packing=True,  # Efficient sequence packing
)
trainer.train()
trainer.save_model()
```

### DPO Trainer  

**Direct Preference Optimization** - Align SFT models using human preferences

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# DPO requires an SFT model as base
config = DPOConfig(
    output_dir="./qwen-aligned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-7,  # Much lower LR for stability
    max_steps=300,
    bf16=True,
    beta=0.1,  # KL penalty strength
    logging_steps=5,
)

trainer = DPOTrainer(
    model="./qwen-chat-model",  # Use your SFT model
    ref_model="Qwen/Qwen2.5-0.5B",  # Original base model
    args=config,
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train_prefs[:500]"),
)
trainer.train()
trainer.save_model()
```

### GRPO Trainer

**Group Relative Policy Optimization** - RLHF without separate reward models

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Define a simple reward function (count unique chars as example)
def reward_function(samples):
    return [len(set(sample.lower())) / 10.0 for sample in samples]

config = GRPOConfig(
    output_dir="./qwen-grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=1e-5,
    max_steps=200,
    bf16=True,
    local_rollout_forward_batch_size=4,  # Generation batch size
    response_length=128,  # Max response length
)

trainer = GRPOTrainer(
    model="./qwen-chat-model",  # Start from SFT model
    args=config,
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

<details>
<summary><strong>Advanced GRPO Configuration</strong></summary>

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# Advanced GRPO with vLLM integration and performance optimization
config = GRPOConfig(
    output_dir="./grpo-qwen-advanced",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    max_steps=300,
    bf16=True,
    gradient_checkpointing=True,
    
    # GRPO-specific parameters
    beta=0.0,  # KL penalty coefficient (0.0 disables KL divergence term)
    epsilon=3e-4,  # PPO clipping parameter
    local_rollout_forward_batch_size=8,  # Generation batch size
    response_length=256,  # Max response length
    
    # vLLM integration for faster generation
    use_vllm=True,
    vllm_mode="colocate",  # Run vLLM within same process
    
    # Memory optimization
    torch_dtype="bfloat16",
    dataloader_pin_memory=False,
)

# Enhanced reward function with mathematical reasoning
def advanced_reward_function(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        
        # Reward for appropriate length (not too short/long)
        ideal_length = min(max(len(prompt) // 2, 20), 100)
        length_penalty = abs(len(completion) - ideal_length) / ideal_length
        reward += max(0, 1.0 - length_penalty)
        
        # Reward for diversity (unique characters)
        diversity_score = len(set(completion.lower())) / max(len(completion), 1)
        reward += diversity_score
        
        # Reward for coherence (no repeated phrases)
        words = completion.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio
        
        rewards.append(reward)
    
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=config,
    train_dataset=load_dataset("trl-lib/tldr", split="train[:1000]"),
    reward_funcs=advanced_reward_function,
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
