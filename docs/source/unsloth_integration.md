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

---

## Basic Usage

### Step 1: Import and Initialize
```python
from unsloth import FastLanguageModel  # FastVisionModel for LLMs
import torch
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
    "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
    "unsloth/Phi-4",  # Phi-4 2x faster!
    "unsloth/Phi-4-unsloth-bnb-4bit",  # Phi-4 Unsloth Dynamic 4-bit Quant
    "unsloth/gemma-2-9b-bnb-4bit",  # Gemma 2x faster!
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # Qwen 2.5 2x faster!
    "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]  # More models at https://docs.unsloth.ai/get-started/all-our-models

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```

### Step 2: Apply LoRA or Adapter
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
```

### Step 3: Training Loop
```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
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
