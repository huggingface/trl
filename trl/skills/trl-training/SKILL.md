---
name: trl-training
description: Train and fine-tune transformer language models using TRL (Transformer Reinforcement Learning). Supports SFT, DPO, GRPO, KTO, RLOO and Reward Model training via CLI commands.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: huggingface
  commands:
    - trl sft
    - trl dpo
    - trl grpo
    - trl kto
    - trl rloo
    - trl reward
  categories:
    - machine-learning
    - llm-training
    - reinforcement-learning
  tags:
    - rlhf
    - supervised-fine-tuning
    - dpo
    - grpo
    - huggingface
    - transformers
  documentation: https://huggingface.co/docs/trl/en/clis
---

# TRL Training Skill

You are an expert at using the TRL (Transformer Reinforcement Learning) library to train and fine-tune large language models.

## Overview

TRL provides CLI commands for post-training foundation models using state-of-the-art techniques:

- **SFT** (Supervised Fine-Tuning): Fine-tune models on instruction-following or conversational datasets
- **DPO** (Direct Preference Optimization): Align models using preference data without a reward model
- **GRPO** (Group Relative Policy Optimization): Train models using LLM-as-a-judge for feedback
- **KTO** (Kahneman-Tversky Optimization): Align models using binary feedback (good/bad)
- **RLOO** (Reinforcement Learning with Language Model Objectives): Online RL training with generation-based rewards
- **Reward Model Training**: Train reward models for RLHF

TRL is built on top of Hugging Face Transformers and Accelerate, providing seamless integration with the Hugging Face ecosystem.

## Core Commands

### trl sft - Supervised Fine-Tuning

Fine-tune language models on instruction-following or conversational datasets.

**Basic usage:**

```bash
trl sft \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/Capybara \
  --output_dir ./sft_output
```

**Key parameters:**

- `--model_name_or_path`: HuggingFace model ID or local path
- `--dataset_name`: Dataset from HuggingFace Hub or local path
- `--dataset_config`: Dataset configuration name (if applicable)
- `--learning_rate`: Learning rate (default: 2.0e-5)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--per_device_train_batch_size`: Batch size per device (default: 8)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--use_peft`: Enable LoRA/QLoRA training
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--packing`: Enable dataset packing for efficiency

**Dataset format:** Expects "messages" column with chat format (system, user, assistant) or "prompt"/"completion" columns.

### trl dpo - Direct Preference Optimization

Align models using preference data (chosen/rejected pairs) without requiring a reward model.

**Basic usage:**

```bash
trl dpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --output_dir ./dpo_output
```

**Key parameters:**

- `--model_name_or_path`: Pre-trained or SFT model to align
- `--dataset_name`: Preference dataset
- `--learning_rate`: Learning rate (default: 5.0e-7)
- `--beta`: DPO temperature parameter (default: 0.1)
- `--loss_type`: Loss function (default: "sigmoid"; options: "sigmoid", "hinge", "ipo", "bco_pair")

**Dataset format:** Requires "prompt", "chosen", and "rejected" columns.

### trl grpo - Group Relative Policy Optimization

Train models using LLM-as-a-judge for evaluating generations and providing rewards.

**Basic usage:**

```bash
trl grpo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/gsm8k \
  --judge_model gpt-4 \
  --output_dir ./grpo_output
```

**Key parameters:**

- `--model_name_or_path`: Model to train
- `--dataset_name`: Training dataset with prompts
- `--judge_model`: Judge model for scoring (e.g., "gpt-4", "claude-3-opus")
- `--num_generations`: Number of generations per prompt (default: 4)
- `--learning_rate`: Learning rate (default: 1.0e-6)

**Dataset format:** Requires "prompt" column. Generations are scored by the judge model.

### trl kto - Kahneman-Tversky Optimization

Align models using binary feedback (thumbs up/down) without paired preferences.

**Basic usage:**

```bash
trl kto \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/kto_mix \
  --output_dir ./kto_output
```

**Key parameters:**

- `--model_name_or_path`: Pre-trained or SFT model
- `--dataset_name`: Dataset with binary labels
- `--learning_rate`: Learning rate (default: 5.0e-7)
- `--beta`: KTO temperature parameter (default: 0.1)

**Dataset format:** Requires "prompt", "completion", and "label" columns (label: True/False or 1/0).

### trl rloo - Reinforcement Learning with Language Objectives

Online RL training where the model generates text and receives rewards based on custom criteria.

**Basic usage:**

```bash
trl rloo \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/tldr \
  --reward_model_name_or_path sentiment-analysis:nlptown/bert-base-multilingual-uncased-sentiment \
  --output_dir ./rloo_output
```

**Key parameters:**

- `--model_name_or_path`: Policy model to train
- `--dataset_name`: Dataset with prompts
- `--reward_model_name_or_path`: Reward model or pipeline (format: "task:model")
- `--learning_rate`: Learning rate (default: 1.0e-6)
- `--num_ppo_epochs`: PPO epochs per batch (default: 4)

### trl reward - Reward Model Training

Train a reward model to score text quality for RLHF.

**Basic usage:**

```bash
trl reward \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --output_dir ./reward_model
```

**Key parameters:**

- `--model_name_or_path`: Base model (will add classification head)
- `--dataset_name`: Preference dataset
- `--learning_rate`: Learning rate (default: 1.0e-5)

**Dataset format:** Requires "prompt", "chosen", and "rejected" columns.

**Output:** Model with classification head that outputs scalar reward scores.

## Configuration Files

TRL supports YAML configuration files for reproducible training. All CLI arguments can be specified in a config file.

**Example config (sft_config.yaml):**

```yaml
model_name_or_path: Qwen/Qwen2.5-0.5B
dataset_name: trl-lib/Capybara
learning_rate: 2.0e-5
num_train_epochs: 1
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
output_dir: ./sft_output
use_peft: true
lora_r: 16
lora_alpha: 16
report_to: wandb
```

**Launch with config:**

```bash
trl sft --config sft_config.yaml
```

**Override config values:**

```bash
trl sft --config sft_config.yaml --learning_rate 1.0e-5
```

## Distributed Training

TRL integrates with Accelerate for multi-GPU and multi-node training.

**Multi-GPU training:**

```bash
trl sft \
  --config sft_config.yaml \
  --num_processes 4
```

**Use predefined Accelerate configs:**

TRL provides predefined configs: `single_gpu`, `multi_gpu`, `fsdp1`, `fsdp2`, `zero1`, `zero2`, `zero3`

```bash
trl sft \
  --config sft_config.yaml \
  --accelerate_config zero2
```

**Custom Accelerate config:**

```bash
# Generate custom config
accelerate config

# Use custom config
trl sft --config sft_config.yaml --config_file ~/.cache/huggingface/accelerate/default_config.yaml
```

**Fully Sharded Data Parallel (FSDP):**

```bash
trl sft --config sft_config.yaml --accelerate_config fsdp2
```

**DeepSpeed ZeRO:**

```bash
trl sft --config sft_config.yaml --accelerate_config zero3
```

## Dataset Requirements

### SFT Datasets

**Chat format:**
```json
{"messages": [
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi! How can I help?"}
]}
```

**Completion format:**
```json
{"prompt": "Translate to French: Hello", "completion": "Bonjour"}
```

### DPO/Reward Datasets

**Preference format:**
```json
{
  "prompt": "Write a poem about AI",
  "chosen": "Silicon dreams...",
  "rejected": "Beep boop robot..."
}
```

### KTO Datasets

**Binary feedback format:**
```json
{"prompt": "Explain gravity", "completion": "Gravity is...", "label": true}
```

### GRPO/RLOO Datasets

**Prompt-only format:**
```json
{"prompt": "Solve: 2+2=?"}
```

Datasets can be:
- From Hugging Face Hub: `--dataset_name username/dataset-name`
- Local files: `--dataset_name ./path/to/dataset`
- Local directories: `--dataset_name ./path/to/dataset_dir`

## Common Workflows

### 1. Standard SFT Training

```bash
# Train on instruction dataset
trl sft \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset_name trl-lib/Capybara \
  --output_dir ./llama-sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2.0e-5
```

### 2. LoRA Fine-Tuning (Memory Efficient)

```bash
# Train with LoRA adapters
trl sft \
  --model_name_or_path meta-llama/Llama-3.2-1B \
  --dataset_name trl-lib/Capybara \
  --output_dir ./llama-lora \
  --use_peft true \
  --lora_r 64 \
  --lora_alpha 64 \
  --lora_target_modules q_proj v_proj k_proj o_proj \
  --num_train_epochs 3
```

### 3. DPO Alignment After SFT

```bash
# Step 1: SFT (as above)

# Step 2: DPO alignment on preferences
trl dpo \
  --model_name_or_path ./llama-sft \
  --dataset_name trl-lib/ultrafeedback_binarized \
  --output_dir ./llama-dpo \
  --learning_rate 5.0e-7 \
  --beta 0.1 \
  --num_train_epochs 1
```

### 4. GRPO with Custom Judge

```bash
# Train with LLM judge feedback
trl grpo \
  --model_name_or_path ./llama-sft \
  --dataset_name trl-lib/gsm8k \
  --judge_model gpt-4o \
  --output_dir ./llama-grpo \
  --num_generations 8 \
  --learning_rate 1.0e-6
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `--per_device_train_batch_size`
- Increase `--gradient_accumulation_steps`
- Enable `--use_peft` for LoRA training
- Use `--gradient_checkpointing` to save memory
- Try smaller model or longer sequence truncation

### Dataset Loading Issues

- Verify dataset exists: check Hugging Face Hub or local path
- Check dataset format matches expected columns
- Use `--dataset_config` for multi-config datasets
- Inspect dataset: `from datasets import load_dataset; ds = load_dataset(name)`

### Model Loading Issues

- Verify model exists on Hugging Face Hub
- Check if gated model requires authentication: `huggingface-cli login`
- For local models, provide absolute path
- Ensure sufficient disk space and memory

### Slow Training

- Enable dataset `--packing` for short sequences
- Use larger `--per_device_train_batch_size` if memory allows
- Enable `--tf32` for faster computation on Ampere GPUs
- Use `--bf16` on supported hardware
- Consider multi-GPU training with `--num_processes`

### Generation Issues (GRPO/RLOO)

- Check prompt format in dataset
- Adjust `--temperature` and `--top_p` for generation
- Verify judge model has API access (for GRPO)
- Check reward model compatibility (for RLOO)

## Additional Resources

- **Documentation**: https://huggingface.co/docs/trl
- **GitHub**: https://github.com/huggingface/trl
- **Examples**: https://github.com/huggingface/trl/tree/main/examples

## Best Practices

1. **Start with SFT**: Always fine-tune base models with SFT before preference alignment
2. **Use LoRA for efficiency**: Enable `--use_peft` for faster training and lower memory
3. **Monitor training**: Use `--report_to trackio` (or `--report_to wandb` or `--report_to tensorboard`) for tracking
4. **Save checkpoints**: TRL automatically saves checkpoints in `--output_dir`
5. **Test on small datasets first**: Verify pipeline works before full training
6. **Use configuration files**: Create YAML configs for reproducibility
7. **Leverage Accelerate**: Use multi-GPU training for faster iteration

When helping users with TRL:
- Always check which training method is appropriate for their use case
- Verify dataset format matches the expected schema
- Recommend starting with smaller models for testing
- Suggest LoRA for resource-constrained environments
- Point to specific documentation sections for advanced features
