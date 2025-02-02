# Unsloth Integration Guide

Unsloth is a lightweight library designed to accelerate and optimize the training of large language models (LLMs) while reducing memory usage. This guide provides step-by-step instructions for integrating Unsloth into your existing workflows.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Benchmarks](#benchmarks)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Installation

1. **Prerequisites**:
   - Python 3.8+
   - PyTorch 2.0+
   - NVIDIA GPU with FP16 support (recommended)

2. **Install Unsloth**:
   ```bash
   pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
   ```

3. **Install Optional Dependencies**:
   ```bash
   pip install flash-attn==2.5.8 trl==0.8.6 accelerate==0.27.2
   ```

---

## Basic Usage

### Step 1: Import and Initialize
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)
```

### Step 2: Apply LoRA or Adapter
```python
model = FastLanguageModel.get_peft_model(
    model,
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
)
```

### Step 3: Training Loop
```python
from transformers import TrainingArguments

trainer = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    learning_rate=2e-5,
    num_train_epochs=1,
)

# Start training
model.train()
```

---

## Advanced Configuration

### Mixed Precision Training
```python
FastLanguageModel.patch_rope_scaling(
    model, 
    fix_ntk=True, 
    scaling_factor=1.23
)
```

### Flash Attention
Enable Flash Attention v2 for 30% speedup:
```python
model = FastLanguageModel.from_pretrained(
    ...,
    use_flash_attention_2=True,
)
```

### Custom Hyperparameters
```python
training_args = TrainingArguments(
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
)
```

---

## Benchmarks

| Model            | Baseline (sec/iter) | Unsloth (sec/iter) | Memory Saved |
|------------------|---------------------|--------------------|--------------|
| Mistral-7B       | 4.2                 | 1.9                | 58%          |
| Llama2-13B       | 7.8                 | 3.1                | 62%          |
| Phi-2            | 1.1                 | 0.4                | 51%          |

---

## Troubleshooting

**Issue**: CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Decrease `max_seq_length`

**Issue**: Slow Training
- Enable `use_flash_attention_2=True`
- Verify `dtype=torch.float16` is set

**Issue**: Installation Errors
```bash
pip install --upgrade "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Resources
- [Official Documentation](https://github.com/unslothai/unsloth)
- [Example Notebooks](https://github.com/unslothai/unsloth/tree/main/examples)
- [Research Paper](https://arxiv.org/abs/2403.19547)
- [Discord Community](https://discord.gg/wwUeB8PXtK)
