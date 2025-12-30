# Quickstart

Get started with TRL in minutes. This guide shows you the essentials for training models with SFT, GRPO, and DPO.

> 💡 **Looking for ready-to-run examples?** Check out our [notebooks for Colab](#-ready-to-run-examples) or [production scripts](example_overview#scripts).

## Quick Examples

Copy-paste these minimal examples to start training immediately. Each uses compact models for quick experimentation.

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

## Ready-to-Run Examples

Want to dive deeper? We provide a comprehensive collection of examples for all skill levels:

<div class="grid grid-cols-1 md:grid-cols-2 gap-4 my-4">
  <div class="border dark:border-gray-700 rounded-lg p-4">
    <h4 style="margin-top: 0;">Notebooks (Beginner-friendly)</h4>
    <p class="text-gray-600 text-sm">Self-contained notebooks for interactive learning. <strong>Many run on free Google Colab</strong>, while some require larger GPUs.</p>
    <ul style="margin-bottom: 0.5rem;">
      <li><a href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb">SFT with QLoRA</a> (free Colab ✓)</li>
      <li><a href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_trl_lora_qlora.ipynb">GRPO with QLoRA</a> (free Colab ✓)</li>
      <li><a href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_qwen3_vl.ipynb">GRPO for Vision-Language Models</a> (free Colab ✓)</li>
    </ul>
    <a href="example_overview#notebooks" class="text-sm">→ See all notebooks</a>
  </div>
  <div class="border dark:border-gray-700 rounded-lg p-4">
    <h4 style="margin-top: 0;">Scripts (Production-ready)</h4>
    <p class="text-gray-600 text-sm">Full-featured scripts for <strong>single GPU, multi-GPU, and DeepSpeed</strong> setups. Ready for real-world training.</p>
    <ul style="margin-bottom: 0.5rem;">
      <li><a href="https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py">SFT Script</a></li>
      <li><a href="https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py">GRPO Script</a></li>
      <li><a href="https://github.com/huggingface/trl/blob/main/trl/scripts/dpo.py">DPO Script</a></li>
    </ul>
    <a href="example_overview#scripts" class="text-sm">→ See all scripts</a>
  </div>
</div>

## What's Next?

### 📚 Learn More

- [SFT Trainer](sft_trainer) - Complete SFT guide
- [DPO Trainer](dpo_trainer) - Preference alignment
- [GRPO Trainer](grpo_trainer) - Group relative policy optimization

### 🚀 Scale Up

- [Distributed Training](distributing_training) - Multi-GPU setups
- [Memory Optimization](reducing_memory_usage) - Efficient training
- [PEFT Integration](peft_integration) - LoRA and QLoRA

### 🌐 Community

- [Community Tutorials](community_tutorials) - External guides and resources

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
