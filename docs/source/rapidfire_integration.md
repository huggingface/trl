# RapidFire AI Integration

RapidFire AI is an open-source experiment execution framework that integrates with TRL to turn "train one configuration at a time" into **real-time, side-by-side comparison of many configurations on the same GPU(s)** — so you can iterate on hyperparameters, LoRA settings, prompt schemes, and ablations **16–24× faster with no extra hardware**.

Links: [GitHub](https://github.com/RapidFireAI/rapidfireai) · [Docs](https://oss-docs.rapidfire.ai) · [Try in Colab](http://tinyurl.com/rapidfireai-colab)

## Why use RapidFire AI with TRL?

When fine-tuning or post-training with TRL, you typically need to:

- Try different hyperparameter configurations
- Compare different LoRA settings
- Test different prompt schemes
- Run ablation studies

| Scenario: comparing N training configs on the same GPU(s) | TRL alone | TRL + RapidFire AI |
| --- | --- | --- |
| Training strategy | Run N configs sequentially | Run N configs concurrently |
| When can you compare configs? | After all runs finish | Live, from the first chunk |
| Stop losers / clone winners mid-training | No | Yes (Interactive Control Operations) |

### How It Works

RapidFire AI employs **adaptive chunk-based scheduling**:

```
GPU Timeline (Single GPU):
Chunk 1: [Config A] → [Config B] → [Config C] → [Config D]
Chunk 2: [Config A] → [Config B] → [Config C] → [Config D]
Chunk 3: [Config A] → [Config B] → [Config C] → [Config D]
```

This enables:

- Early comparison of configurations on same data subsets incrementally
- Efficient GPU utilization and minimizing idle times
- Real-time and automated experiment metrics tracking
- Dynamic control over runs in flight to incentivize more experimentation

## Key Features

- **16-24× higher experimentation throughput** compared to sequential training.
- **Almost no code changes** - simple drop-in config APIs that just wrap around existing TRL and PEFT config APIs.
- **Interactive Control Operations** - real-time control to stop, resume, and clone-modify (with or without warm starting) training runs in flight.
- **Integration with Fully Sharded Data Parallel (FSDP)** for training large models that do not fit on a single GPU by sharding parameters, gradients, and optimizer states across multiple GPUs.
- **Full compatibility** with transformers, PEFT, SFTTrainer, DPOTrainer, and GRPOTrainer.
- **Pluggable experiment tracking**: MLflow (default), TensorBoard, and Trackio, enabled individually or in combination.
- **Zero-setup Google Colab support**: one-click tutorial notebooks for SFT, DPO, and GRPO on free T4 GPUs.
- **Production-Ready**: Already used in production environments with complete working examples.

## Installation

### Prerequisites

- Python 3.12.x
- NVIDIA GPU with Compute Capability 7.x or 8.x (multiple GPUs required for FSDP)
- CUDA Toolkit 11.8+
- PyTorch 2.8+

### pip install

```bash
pip install rapidfireai
```

Once installed, authenticate with Hugging Face and initialize RapidFire AI:

```bash
# Authenticate with Hugging Face
hf auth login --token YOUR_TOKEN

# Workaround for current issue: https://github.com/huggingface/xet-core/issues/527
pip uninstall -y hf-xet

# Initialize RapidFire AI
rapidfireai init

# Start the RapidFire AI server
rapidfireai start
```

The dashboard will be available at `http://localhost:8853` where you can monitor and control experiments in real-time.

## Quick Start: SFT Training with Multiple Configs

Here's a complete example showing how to train multiple SFT configurations concurrently:

```python
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
train_dataset = dataset["train"].select(range(128)).shuffle(seed=42)
eval_dataset = dataset["train"].select(range(100, 124)).shuffle(seed=42)

# Define data formatting function
def formatting_function(row):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": row["instruction"]},
        ],
        "completion": [
            {"role": "assistant", "content": row["response"]}
        ]
    }

# Initialize experiment
experiment = Experiment(experiment_name="sft-customer-support")

# Define multiple LoRA configurations to compare
peft_configs = List([
    RFLoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, 
                 target_modules=["q_proj", "v_proj"], bias="none"),
    RFLoraConfig(r=32, lora_alpha=64, lora_dropout=0.1,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
])

# Define multiple training configurations
# 2 base configs × 2 PEFT configs = 4 total training runs
config_set = List([
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=peft_configs,
        training_args=RFSFTConfig(  # Wraps TRL's SFTConfig
            learning_rate=1e-3,
            per_device_train_batch_size=4,
            max_steps=128,
            fp16=True,
        ),
        model_type="causal_lm",
        model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
        formatting_func=formatting_function,
    ),
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=peft_configs,
        training_args=RFSFTConfig(
            learning_rate=1e-4,  # Different learning rate
            per_device_train_batch_size=4,
            max_steps=128,
            fp16=True,
        ),
        model_type="causal_lm",
        model_kwargs={"device_map": "auto", "torch_dtype": "auto", "use_cache": False},
        formatting_func=formatting_function,
    )
])

# Define model creation function
def create_model(model_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"], 
        **model_config["model_kwargs"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    return (model, tokenizer)

# Create grid search over all configurations
config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

# Run all 4 configurations concurrently with chunk-based scheduling
experiment.run_fit(config_group, create_model, train_dataset, eval_dataset, 
                   num_chunks=4, seed=42)

# End experiment
experiment.end()
```

### What Happens During Execution

When you run this example:

1. **Config Expansion**: 2 base configurations × 2 PEFT configs = 4 total training runs
2. **Chunk-based Scheduling**: Training data is divided into chunks, and all 4 configs train concurrently
3. **GPU Swapping**: Models are swapped in/out of GPU memory based on chunk boundaries
4. **Real-time Tracking**: All metrics visible in the dashboard at `http://localhost:8853`
5. **Interactive Control**: Stop, resume, or clone-modify any configuration from the dashboard

This delivers **16-24× higher throughput** compared to training each configuration sequentially!

## Supported TRL Trainers

### SFTTrainer

Use `RFSFTConfig` as a drop-in replacement for `SFTConfig`:

```python
from rapidfireai.automl import RFSFTConfig

training_args = RFSFTConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    max_length = 512,
    # ... all other SFTConfig parameters supported
)
```

**Example Notebook**: [SFT for Customer Support](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-lite.ipynb)

### DPOTrainer

Use `RFDPOConfig` as a drop-in replacement for `DPOConfig`:

```python
from rapidfireai.automl import RFDPOConfig

training_args = RFDPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
    learning_rate=5e-4,
    # ... all other DPOConfig parameters supported
)
```

**Example Notebook**: [DPO for Preference Alignment](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment-lite.ipynb)

### GRPOTrainer

Use `RFGRPOConfig` as a drop-in replacement for `GRPOConfig`:

```python
from rapidfireai.automl import RFGRPOConfig

training_args = RFGRPOConfig(
    learning_rate=5e-6,
    num_generations=8,
    max_completion_length=256,
    # ... all other GRPOConfig parameters supported
)
```

**Example Notebook**: [GRPO for Math Reasoning](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-grpo-mathreasoning-lite.ipynb)

## Core Concepts

### Chunk-Based Concurrent Training

RapidFire AI divides training data into chunks and alternates between configurations:

```
GPU Timeline (Single GPU):
Chunk 1: [Config A] → [Config B] → [Config C] → [Config D]
Chunk 2: [Config A] → [Config B] → [Config C] → [Config D]
Chunk 3: [Config A] → [Config B] → [Config C] → [Config D]
...
```

This approach maximizes GPU utilization and enables early comparison of configurations while maintaining training stability through automatic checkpointing.

### Interactive Control Operations (IC Ops)

Through the RapidFire AI dashboard, you can dynamically control running experiments:

- **Stop**: Pause a configuration (checkpointed automatically)
- **Resume**: Continue from last checkpoint
- **Clone-Modify**: Duplicate a configuration with modifications (new run starts from scratch)
- **Clone-Modify with Warm Start**: Clone-modify and initialize from the parent's weights
- **Delete**: Remove failed or unwanted runs

This enables adaptive experimentation where you can stop underperforming configs early and clone promising ones with tweaked hyperparameters.

### Multi-Config Experimentation

Use `RFGridSearch` or `RFRandomSearch` to automatically generate configuration combinations:

```python
# Grid search: tests all combinations
config_group = RFGridSearch(configs=config_list, trainer_type="SFT")

# Random search: samples N configurations
config_group = RFRandomSearch(configs=config_list, trainer_type="DPO", num_samples=10)
```

## Advanced Features

### PEFT/LoRA Integration

Full support for parameter-efficient fine-tuning:

```python
from rapidfireai.automl import RFLoraConfig
from peft import TaskType

lora_config = RFLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)
```

### Custom Reward Functions (GRPO)

Define multiple reward functions for GRPO training:

```python
def correctness_reward(prompts, completions, answer, **kwargs):
    """Reward for correct answers"""
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def format_reward(completions, **kwargs):
    """Reward for proper formatting"""
    import re
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Use in model config
config = RFModelConfig(
    reward_funcs=[correctness_reward, format_reward],
    # ... other parameters
)
```

### Multi-GPU Support

RapidFire AI automatically detects and utilizes all available GPUs. By default, the scheduler distributes independent configurations across GPUs (data-parallel across configs), so no special setup is required to run `N` configs on `N` GPUs concurrently.

For models that do not fit on a single GPU, RapidFire AI also supports **Fully Sharded Data Parallel (FSDP)** to shard a single configuration across multiple GPUs — see the next section.

### Multi-GPU Training with FSDP

When a model is too large for a single GPU, enable FSDP directly through the training args of `RFSFTConfig` or `RFDPOConfig` — the same `fsdp` and `fsdp_config` fields exposed by Hugging Face `TrainingArguments`:

```python
from rapidfireai.automl import RFModelConfig, RFSFTConfig, RFLoraConfig

model_config = RFModelConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    peft_config=RFLoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    ),
    training_args=RFSFTConfig(
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "sharding_strategy": "FULL_SHARD",
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "backward_prefetch": "backward_pre",
            "forward_prefetch": True,
            "use_orig_params": False,
            "cpu_ram_efficient_loading": True,
            "offload_params": True,
            "sync_module_states": True,
            "limit_all_gathers": True,
        },
    ),
    model_type="causal_lm",
    model_kwargs={"torch_dtype": "auto"},
)
```

Key points:

- FSDP works transparently with RapidFire AI's chunk-based scheduling, IC Ops (stop / resume / clone-modify with or without warm-starting), and all supported metric tracking backends.
- FSDP is fully compatible with PEFT / LoRA — LoRA adapter weights are collected efficiently across shards when saving checkpoints.
- FSDP composes with grid search and random search: each expanded config gets its own sharded training run.

**Example Notebooks**:
- [SFT with FSDP (lite, small model)](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-fsdp-lite.ipynb)
- [SFT with FSDP (large model)](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-tutorial-sft-chatqa-fsdp-large.ipynb)
- [DPO with FSDP](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/post-training/rf-tutorial-dpo-alignment-fsdp-lite.ipynb)

### Experiment Tracking Backends

RapidFire AI supports three metric logging backends that can be used individually or together: **MLflow** (the default for local installs), **TensorBoard** (the default in Google Colab), and **Trackio**.

Select one or more backends at server startup with the `--tracking-backends` flag:

```bash
# MLflow only (default on local installs)
rapidfireai start --tracking-backends mlflow

# TensorBoard only
rapidfireai start --tracking-backends tensorboard

# Any combination
rapidfireai start --tracking-backends mlflow tensorboard trackio
```

Equivalent environment variables are also available:

- `RF_MLFLOW_ENABLED` (default `true`, or `false` in Colab)
- `RF_TENSORBOARD_ENABLED` (default `false`, or `true` in Colab)
- `RF_TRACKIO_ENABLED` (default `false`)

All three backends receive the same metrics (loss, evaluation scores, learning rate, etc.) and respect IC Ops run lifecycle events, so you can use, for example, Trackio for lightweight sharing alongside MLflow for a full local dashboard.

### Running in Google Colab

RapidFire AI runs on free Google Colab T4 GPUs, with tutorial notebooks for SFT, DPO, GRPO, and RAG / context-engineering workflows. In Colab, TensorBoard is the default tracking backend (MLflow is disabled for simplicity), and the usual `rapidfireai init` / `rapidfireai start` commands run directly from notebook cells — no terminal access required.

Get started: [RapidFire AI in Google Colab](http://tinyurl.com/rapidfireai-colab).

## Best Practices

### Tuning Chunk Granularity

The `num_chunks` parameter controls swap frequency:

```python
# Fewer chunks = less overhead, less frequent comparison
experiment.run_fit(..., num_chunks=2)

# More chunks = more overhead, more frequent comparison
experiment.run_fit(..., num_chunks=16)
```

**Rule of thumb**: Start with `num_chunks=4` and adjust based on dataset size and number of configurations.

### Memory Management

For large models, use quantization:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_kwargs = {
    "quantization_config": bnb_config,
    "device_map": "auto",
}
```

## Performance Benchmarks

Based on internal benchmarks comparing sequential vs. RapidFire AI concurrent training:

| Scenario | Sequential Time | RapidFire AI Time | Speedup |
|----------|----------------|-------------------|---------|
| 4 configs, 1 GPU | 120 min | 7.5 min | 16× |
| 8 configs, 1 GPU | 240 min | 12 min | 20× |
| 4 configs, 2 GPUs | 60 min | 4 min | 15× |
| 8 configs, 4 GPUs | 60 min | 3 min | 20× |

*Benchmarks performed on NVIDIA A100 40GB with TinyLlama-1.1B and Llama-3.2-1B models*

## Troubleshooting

For troubleshooting guidance, see the [RapidFire AI Troubleshooting Guide](https://oss-docs.rapidfire.ai/en/latest/troubleshooting.html).

## Additional Resources
- **Colab Notebook**: [RapidFire AI in Google Colab](http://tinyurl.com/rapidfireai-colab)
- **Documentation**: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai)
- **GitHub**: [RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai)
- **PyPI**: [pypi.org/project/rapidfireai](https://pypi.org/project/rapidfireai/)
- **Discord**: [Join our Discord](https://discord.gg/6vSTtncKNN)
- **Tutorial Notebooks**: [GitHub Repository](https://github.com/RapidFireAI/rapidfireai/tree/main/tutorial_notebooks)

Learn more about RapidFire AI in their [official repository](https://github.com/RapidFireAI/rapidfireai) and [documentation](https://oss-docs.rapidfire.ai).

