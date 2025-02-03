# Unsloth Integration Guide

Unsloth is a lightweight library designed to accelerate and optimize the training of large language models (LLMs) while reducing memory usage. This guide provides step-by-step instructions for integrating Unsloth into your existing workflows.

---

## Installation Instructions

### Stable Releases
For stable releases, use the following command:
```bash
pip install unsloth
```

### Recommended Installation
For most installations, we recommend using:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## System Requirements

- **Operating System**: Works on Linux and Windows via WSL.
- **GPU Support**: Supports NVIDIA GPUs since 2018+. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40, etc.). Check your GPU! GTX 1070, 1080 works, but is slow.
- **Dependencies**: Your device must have `xformers`, `torch`, `BitsandBytes`, and `triton` support.
- **Disk Space**: Ensure you have sufficient disk space to train and save your model.

**Note**: Unsloth only works if you have an NVIDIA GPU.

---

## Fine-tuning VRAM Requirements

How much GPU memory do you need for LLM fine-tuning using Unsloth?  
Check the table below for VRAM requirements sorted by model parameters and fine-tuning method.  
- **QLoRA** uses 4-bit precision.  
- **LoRA** uses 16-bit precision.

| Model Parameters | QLoRA (4-bit) VRAM | LoRA (16-bit) VRAM |
|------------------|-------------------|--------------------|
| 3B               | 2 GB              | 7 GB               |
| 7B               | 4.5 GB            | 16 GB              |
| 8B               | 5 GB              | 19 GB              |
| 9B               | 5.5 GB            | 21 GB              |
| 11B              | 6.5 GB            | 26 GB              |
| 14B              | 8.5 GB            | 33 GB              |
| 27B              | 16 GB             | 64 GB              |
| 32B              | 19 GB             | 76 GB              |
| 40B              | 24 GB             | 96 GB              |
| 70B              | 41 GB             | 164 GB             |
| 81B              | 48 GB             | 192 GB             |
| 405B             | 237 GB            | 950 GB             |

## Training Setup
```python
from transformers import TrainingArguments
from trl import SFTTrainer  # Fixed trainer import 

# Proper initialization sequence
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    learning_rate=2e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)

# Initialize proper trainer 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
)
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
trainer_stats = trainer.train()
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
If you face any issues during installation or updating, please check the official website for guidance:  
[Unsloth Installation Guide](https://docs.unsloth.ai/get-started/installing-+-updating)

---

## Resources
- [Official Documentation](https://github.com/unslothai/unsloth)
- [Example Notebooks](https://github.com/unslothai/unsloth/tree/main/examples)
