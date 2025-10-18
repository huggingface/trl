# Multi-Token Prediction (MTP) with SFT Trainer

[![All_models-MTP-green](https://img.shields.io/badge/All_models-MTP-green)](https://huggingface.co/models?other=mtp,trl)

## Overview

TRL supports Multi-Token Prediction (MTP) training as an extension to the Supervised Fine-Tuning (SFT) method. MTP is a training technique that enables models to predict multiple future tokens simultaneously during training, as described in the paper [Better & Faster Large Language Models via Multi-token Prediction](https://huggingface.co/papers/2404.19737) by Meta AI.

The abstract from the paper is the following:

> Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, each trained using the standard next-token prediction loss. We call this approach multi-token prediction (MTP). Across a range of model sizes and training configurations, we show that MTP consistently accelerates training. For a fixed training budget, MTP improves performance on a range of downstream tasks. Moreover, MTP can be used to speed-up inference through speculative decoding: we show that an MTP-trained model can serve as its own draft model, resulting in up to 3× inference speedup.

MTP extends the standard next-token prediction objective by training additional prediction heads to forecast multiple future tokens in parallel. This approach provides richer training signals and can significantly improve both training efficiency and inference speed.

## Quick start

This example demonstrates how to train a language model using Multi-Token Prediction with the [`SFTTrainer`]. We train a [Qwen 2.5 0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) model on the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara) with MTP enabled.

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=SFTConfig(
        output_dir="Qwen2.5-0.5B-MTP",
        mtp_enabled=True,                    # Enable Multi-Token Prediction
        mtp_num_predictions=2,               # Predict 2 future tokens
        mtp_loss_weight=0.5,                 # Weight for MTP auxiliary loss
    ),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

## Looking deeper into the MTP method

Multi-Token Prediction extends the standard language modeling objective by training the model to predict multiple future tokens at each position during training. While traditional language models only optimize for the next token prediction:

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})$$

MTP adds auxiliary prediction heads that jointly optimize for multiple future positions:

$$\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{K} w_k \cdot \text{CE}(p_\theta(y_{t+k} \mid y_{\leq t}), y_{t+k})$$

The total loss combines both objectives:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{NTP}} + \alpha \cdot \mathcal{L}_{\text{MTP}}$$

where $K$ is the number of future tokens to predict, $w_k$ are position-specific weights, and $\alpha$ is the MTP loss weight.

### Benefits of MTP

1. **Training Efficiency**: Provides multiple supervision signals per forward pass
2. **Better Representations**: Forces the model to encode longer-range dependencies
3. **Inference Speedup**: Enables speculative decoding for up to 3× faster generation
4. **Improved Performance**: Better understanding of sequence structure and patterns

## MTP Configuration

### Core MTP Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mtp_enabled` | bool | `False` | Enable Multi-Token Prediction |
| `mtp_num_predictions` | int | `2` | Number of future tokens to predict |
| `mtp_loss_weight` | float | `0.5` | Weight for MTP auxiliary loss |

### Head Architecture Options

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `mtp_head_type` | str | `"linear"` | `linear`, `ffn`, `mha_ffn`, `cnn`, `identical` | MTP head architecture type |
| `mtp_num_layers` | int | `1` | `1`, `2`, `3`, `4+` | Number of layers per MTP head |
| `mtp_init_strategy` | str | `"default"` | `default`, `kaiming_uniform`, `kaiming_normal`, `xavier_uniform`, `xavier_normal`, `copy_lm_head` | Parameter initialization strategy |

### Advanced Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

| `mtp_dropout_prob` | float | `0.1` | Dropout probability for regularization |
| `mtp_weight_decay_strategy` | str | `"uniform"` | Loss weighting: `uniform` or `harmonic` |

## Head Architecture Types

### 1. `identical` (Recommended)
- **Structure**: Exactly same as original LM head
- **Benefits**: Perfect alignment, preserves model characteristics
- **Use case**: General fine-tuning, maintaining model behavior
- **Best with**: `copy_lm_head` initialization

```python
# Example: Identical head structure with parameter copying
training_args = SFTConfig(
    mtp_enabled=True,
    mtp_head_type="identical",
    mtp_init_strategy="copy_lm_head",
    mtp_num_layers=1,
)
```

### 2. `linear`
- **Structure**: Simple linear projection(s)
- **Benefits**: Lightweight, fast training
- **Use case**: Quick experiments, resource-constrained scenarios
- **Best with**: `default` or `xavier_uniform` initialization

### 3. `ffn`
- **Structure**: Feed-forward network(s)
- **Benefits**: Good balance of capacity and efficiency
- **Use case**: Most general applications
- **Best with**: `kaiming_uniform` initialization

### 4. `mha_ffn`
- **Structure**: Multi-head attention + FFN layers
- **Benefits**: Maximum modeling capacity, attention mechanisms
- **Use case**: Complex tasks, long-range dependencies
- **Best with**: `xavier_normal` initialization

### 5. `cnn`
- **Structure**: 1D convolutional layers
- **Benefits**: Local pattern modeling, parameter sharing
- **Use case**: Structured text, code generation, NOT SURE YET
- **Best with**: `kaiming_uniform` initialization

## Usage Examples

### Example 1: Basic MTP Training

```python
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and tokenizer
dataset = load_dataset("trl-lib/Capybara", split="train")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Configure MTP training
training_args = SFTConfig(
    output_dir="./qwen-mtp",
    mtp_enabled=True,
    mtp_num_predictions=2,
    mtp_loss_weight=0.5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

# Train with MTP
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

### Example 2: Identical Head Structure

Perfect for preserving model characteristics while adding MTP capability:

```python
# Qwen model with identical head structure
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=SFTConfig(
        output_dir="./qwen-identical-mtp",
        mtp_enabled=True,
        mtp_head_type="identical",           # Same structure as LM head
        mtp_init_strategy="copy_lm_head",    # Copy original parameters
        mtp_num_layers=1,                    # Single layer like original
        mtp_num_predictions=2,
        mtp_loss_weight=0.3,
        num_train_epochs=3,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

### Example 3: Multi-Layer Deep Heads

For complex tasks requiring more model capacity:

```python
# Llama model with multi-layer FFN heads
trainer = SFTTrainer(
    model="meta-llama/Llama-3.2-1B",
    args=SFTConfig(
        output_dir="./llama-multilayer-mtp",
        mtp_enabled=True,
        mtp_head_type="ffn",                 # FFN head architecture
        mtp_num_layers=3,                    # Multi-layer heads
        mtp_init_strategy="kaiming_uniform", # Advanced initialization
        mtp_num_predictions=3,               # Predict 3 steps ahead
        mtp_loss_weight=0.3,
        mtp_weight_decay_strategy="harmonic", # Decay weights by distance
        num_train_epochs=2,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

### Example 4: Advanced Deep Transformer Heads

For maximum modeling capacity:

```python
# DeepSeek model with deep MHA+FFN heads
trainer = SFTTrainer(
    model="deepseek-ai/deepseek-llm-7b-base",
    args=SFTConfig(
        output_dir="./deepseek-deep-mtp",
        mtp_enabled=True,
        mtp_head_type="mha_ffn",             # Complex head architecture
        mtp_num_layers=4,                    # Deep multi-layer heads
        mtp_init_strategy="xavier_normal",   # Advanced initialization
        mtp_num_predictions=4,               # Predict 4 steps ahead
        mtp_loss_weight=0.2,
        mtp_weight_decay_strategy="harmonic",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

### Example 5: Parameter-Efficient Setup

For resource-constrained scenarios:

```python
# Mistral model with parameter-efficient MTP
trainer = SFTTrainer(
    model="mistralai/Mistral-7B-v0.1",
    args=SFTConfig(
        output_dir="./mistral-efficient-mtp",
        mtp_enabled=True,
        mtp_head_type="linear",              # Simple heads
        mtp_num_layers=1,                    # Single layer

        mtp_init_strategy="default",         # Default initialization
        mtp_dropout_prob=0.0,               # No dropout for efficiency
        mtp_num_predictions=2,
        mtp_loss_weight=0.4,
        num_train_epochs=3,
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
)
```

## Command Line Interface

You can also use the TRL Command Line Interface (CLI) to train with MTP:

```bash
# Basic MTP training
trl sft --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-0.5B-MTP \
    --mtp_enabled \
    --mtp_num_predictions 2 \
    --mtp_loss_weight 0.5

# Advanced MTP with identical heads
trl sft --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --output_dir Llama-3.2-1B-MTP \
    --mtp_enabled \
    --mtp_head_type identical \
    --mtp_init_strategy copy_lm_head \
    --mtp_num_layers 1 \
    --mtp_num_predictions 3

# Multi-layer deep heads
trl sft --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
    --dataset_name trl-lib/Capybara \
    --output_dir DeepSeek-7B-MTP \
    --mtp_enabled \
    --mtp_head_type mha_ffn \
    --mtp_num_layers 4 \
    --mtp_init_strategy xavier_normal \
    --mtp_num_predictions 4 \
    --mtp_loss_weight 0.2
```

## Initialization Strategies

### When to Use Each Strategy

| Strategy | Best For | Description |
|----------|----------|-------------|
| `copy_lm_head` | `identical` heads | Copy original LM head parameters |
| `kaiming_uniform` | ReLU-based networks | He initialization (uniform distribution) |
| `kaiming_normal` | ReLU-based networks | He initialization (normal distribution) |
| `xavier_uniform` | Tanh/Sigmoid networks | Xavier initialization (uniform) |
| `xavier_normal` | Tanh/Sigmoid networks | Xavier initialization (normal) |
| `default` | General use | PyTorch default initialization |

### Recommended Combinations

```python
# For preserving model characteristics
SFTConfig(
    mtp_head_type="identical",
    mtp_init_strategy="copy_lm_head",
    mtp_num_layers=1,
)

# For increased capacity
SFTConfig(
    mtp_head_type="ffn",
    mtp_init_strategy="kaiming_uniform",
    mtp_num_layers=2,
)

# For maximum performance
SFTConfig(
    mtp_head_type="mha_ffn",
    mtp_init_strategy="xavier_normal",
    mtp_num_layers=3,
)
```

## Multi-Layer Head Support

MTP heads can have multiple layers to increase modeling capacity:

- **Single Layer (N=1)**: Fast, lightweight, good for simple tasks
- **Few Layers (N=2-3)**: Better capacity without too much overhead  
- **Many Layers (N=4+)**: Maximum capacity for complex modeling

```python
# Single layer (default)
SFTConfig(mtp_num_layers=1)

# Multi-layer for increased capacity
SFTConfig(
    mtp_head_type="ffn",
    mtp_num_layers=3,  # 3-layer FFN heads
    mtp_init_strategy="kaiming_uniform",
)

# Deep transformer heads
SFTConfig(
    mtp_head_type="mha_ffn", 
    mtp_num_layers=4,  # 4-layer transformer heads
    mtp_init_strategy="xavier_normal",
)
```

## Parameter Initialization Strategy

The `mtp_init_strategy` parameter controls how MTP head parameters are initialized:

### Initialization Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `copy_lm_head` | Initialize from original LM head parameters | `identical` heads, preserving model characteristics |
| `kaiming_uniform` | He initialization (uniform distribution) | ReLU-based networks (`ffn`, `cnn`) |
| `kaiming_normal` | He initialization (normal distribution) | ReLU-based networks |
| `xavier_uniform` | Xavier initialization (uniform) | Tanh/Sigmoid networks (`mha_ffn`) |
| `xavier_normal` | Xavier initialization (normal) | Tanh/Sigmoid networks |
| `default` | PyTorch default initialization | General use |

### Copy LM Head Strategy

When using `mtp_init_strategy="copy_lm_head"`, the initialization behavior depends on head compatibility:

```python
# For identical heads: Copy entire structure and parameters
if head_type == "identical":
    # MTP heads get exact copy of LM head structure and parameters
    mtp_head = copy_structure_and_parameters(lm_head)

# For compatible layers: Copy only applicable parameters  
elif head_type in ["linear", "ffn"]:
    # Only vocabulary projection layers are initialized from LM head
    final_linear_layer.weight.data.copy_(lm_head.weight.data)

# For incompatible heads: Use fallback initialization
elif head_type in ["cnn", "mha_ffn"]:
    # CNN/MHA layers use their default initialization
    # Only final projection initialized from LM head if applicable
```

This ensures that:
- 🎯 **Good starting point** from proven LM head parameters
- 🧠 **Preserved diversity** through independent training
- ⚡ **No parameter conflicts** during gradient updates
- 🔄 **Clear semantics** for each head type

## Weight Decay Strategies

MTP supports different weighting strategies for multi-step predictions:

### Uniform Weighting
All prediction steps have equal weight:
```python
SFTConfig(mtp_weight_decay_strategy="uniform")  # w_1 = w_2 = w_3 = 1.0
```

### Harmonic Weighting  
Distant predictions have lower weight:
```python
SFTConfig(mtp_weight_decay_strategy="harmonic")  # w_1 = 1.0, w_2 = 0.5, w_3 = 0.33
```

## Logged metrics

When MTP is enabled, additional metrics are logged during training:

- `ntp_loss`: Standard next-token prediction loss
- `mtp_loss`: Multi-token prediction auxiliary loss  
- `train_loss`: Combined total loss (NTP + α × MTP)
- Standard SFT metrics: `entropy`, `mean_token_accuracy`, etc.

## Example script

We provide an example script to train a model using MTP. The script is available in [`examples/scripts/sft_with_mtp.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_with_mtp.py).

To test MTP with the [Qwen2.5 0.5B model](https://huggingface.co/Qwen/Qwen2.5-0.5B) on the [Capybara dataset](https://huggingface.co/datasets/trl-lib/Capybara), run:

```bash
python examples/scripts/sft_with_mtp.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-0.5B-MTP \
    --mtp_enabled \
    --mtp_num_predictions 2 \
    --mtp_loss_weight 0.5 \
    --num_train_epochs 3
```

## Advanced Usage Patterns

### Progressive Complexity

Start simple and increase complexity based on results:

```python
# Stage 1: Baseline with identical heads
config_simple = SFTConfig(
    mtp_enabled=True,
    mtp_head_type="identical",
    mtp_init_strategy="copy_lm_head",
    mtp_num_layers=1,
    mtp_num_predictions=2,
)

# Stage 2: Add capacity with multi-layer
config_medium = SFTConfig(
    mtp_enabled=True,
    mtp_head_type="ffn", 
    mtp_num_layers=2,
    mtp_init_strategy="kaiming_uniform",
    mtp_num_predictions=3,
)

# Stage 3: Maximum capacity
config_advanced = SFTConfig(
    mtp_enabled=True,
    mtp_head_type="mha_ffn",
    mtp_num_layers=4,
    mtp_init_strategy="xavier_normal",
    mtp_num_predictions=4,
)
```

### Model-Specific Optimization

Different models may benefit from different configurations:

```python
# For Qwen models (good with identical structure)
qwen_config = SFTConfig(
    mtp_head_type="identical",
    mtp_init_strategy="copy_lm_head",
    mtp_num_layers=1,
)

# For Llama models (benefits from FFN capacity)  
llama_config = SFTConfig(
    mtp_head_type="ffn",
    mtp_num_layers=2,
    mtp_init_strategy="kaiming_uniform",
)

# For DeepSeek models (can handle complex heads)
deepseek_config = SFTConfig(
    mtp_head_type="mha_ffn",
    mtp_num_layers=3,
    mtp_init_strategy="xavier_normal",
)
```

## Performance Considerations

### Memory Usage
- **Identical heads**: Similar overhead to original LM head
- **Multi-layer heads**: Scales with number of layers
- **Parameter tying**: Reduces memory footprint significantly

### Training Speed
- **Single layer**: Fastest training
- **Multi-layer**: Moderate overhead
- **Deep heads**: Higher computational cost but better capacity

### Convergence
- **Copy LM head**: Fast convergence, preserves learned patterns
- **Advanced initialization**: Better for complex architectures
- **Multi-layer**: May need more epochs but achieves better performance

## Distributed Training

MTP is fully compatible with all distributed training strategies supported by TRL. Since MTP heads are added as regular PyTorch modules after the model is prepared by Accelerate, they automatically participate in distributed training without requiring any additional configuration.

### Multi-GPU Training with Accelerate

Use Accelerate's DDP (Distributed Data Parallel) for multi-GPU training:

```bash
# Create accelerate config
accelerate config

# Launch multi-GPU training
accelerate launch examples/scripts/sft_with_mtp.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_name trl-lib/Capybara \
    --output_dir ./results/qwen-mtp-ddp \
    --mtp_enabled \
    --mtp_head_type identical \
    --mtp_init_strategy copy_lm_head \
    --mtp_num_predictions 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2
```

### FSDP (Fully Sharded Data Parallel)

For large models that don't fit on a single GPU, use FSDP:

```bash
# Launch FSDP training with predefined config
accelerate launch --config_file examples/accelerate_configs/fsdp1.yaml \
    examples/scripts/sft_with_mtp.py \
    --model_name_or_path meta-llama/Llama-3.2-7B \
    --dataset_name HuggingFaceH4/ultrachat_200k \
    --output_dir ./results/llama-mtp-fsdp \
    --mtp_enabled \
    --mtp_head_type ffn \
    --mtp_num_layers 3 \
    --mtp_init_strategy kaiming_uniform \
    --mtp_num_predictions 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### DeepSpeed ZeRO

For maximum memory efficiency with large models:

```bash
# Launch DeepSpeed ZeRO-2 training
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/sft_with_mtp.py \
    --model_name_or_path deepseek-ai/deepseek-llm-7b-base \
    --dataset_name trl-lib/Capybara \
    --output_dir ./results/deepseek-mtp-zero2 \
    --mtp_enabled \
    --mtp_head_type mha_ffn \
    --mtp_num_layers 4 \
    --mtp_init_strategy xavier_normal \
    --mtp_num_predictions 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16
```

### Key Points for Distributed Training

1. **Automatic Integration**: MTP heads are added after `accelerator.prepare()`, so they automatically participate in:
   - Parameter sharding (FSDP/DeepSpeed)
   - Gradient synchronization (DDP)
   - Memory optimization (ZeRO)

2. **No Additional Configuration**: The same MTP configuration works across all distributed setups

3. **Batch Size Scaling**: Remember to adjust batch sizes for distributed training:
   ```python
   # Effective batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps
   SFTConfig(
       per_device_train_batch_size=2,    # Per GPU
       gradient_accumulation_steps=4,    # Accumulation
       # With 4 GPUs: effective batch size = 2 × 4 × 4 = 32
   )
   ```

4. **Memory Considerations**: MTP heads add parameters that are automatically managed by the distributed strategy:
   - **FSDP**: MTP parameters are sharded across GPUs
   - **DeepSpeed**: MTP parameters participate in ZeRO optimization
   - **DDP**: MTP gradients are synchronized across replicas

## Compatibility

MTP is fully compatible with:

- **PEFT Integration**: Works with LoRA, QLoRA, and other adapters
- **Distributed Training**: Automatic support for DDP, DeepSpeed, and FSDP
- **Mixed Precision**: Compatible with fp16 and bf16 training
- **All Model Architectures**: Works with any causal language model
- **Memory Optimization**: Integrates with gradient checkpointing and activation offloading

## DataCollatorForMTPLanguageModeling

[[autodoc]] trainer.mtp_data_collator.DataCollatorForMTPLanguageModeling
