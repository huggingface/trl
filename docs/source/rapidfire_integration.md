# RapidFire AI Integration

RapidFire AI is an open-source experiment execution framework that enables concurrent training of multiple TRL configurations on the same GPU(s) through intelligent chunk-based scheduling.

## Key Features

- **16-24× higher experimentation throughput** compared to sequential training.
- **Almost no code changes** - drop-in configuration wrappers around TRL's and PEFT's existing configs.
- **Interactive Control Operations** - real-time control to stop, resume, clone, and modify training runs in flight
- **Automatic multi-GPU orchestration** with intelligent scheduling
- **Full compatibility** with transformers, PEFT, SFTTrainer, DPOTrainer, and GRPOTrainer
- **Full MLflow Integration**: Automatic experiment tracking and visualization
- **Production-Ready**: Already used in production environments with complete working examples.

### Problem It Solves

When fine-tuning or post-training with TRL, AI developers often need to:
- Try different hyperparameter configurations
- Compare different LoRA settings
- Test different prompt schemes
- Run ablation studies


**Current approach**: Train each config one after another → slow and inefficient process

**With RapidFire AI**: Train all configs in one go even on a single GPU → 16-24× faster process

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


## Installation

### Prerequisites

- Python 3.12.x
- NVIDIA GPU with Compute Capability 7.x or 8.x
- CUDA Toolkit 11.8+
- PyTorch 2.7.1+

### pip install

```bash
pip install rapidfireai
```

Once installed, authenticate with Hugging Face and initialize RapidFire AI:

```bash
# Authenticate with Hugging Face
huggingface-cli login --token YOUR_TOKEN

# Workaround for current issue: https://github.com/huggingface/xet-core/issues/527
pip uninstall -y hf-xet

# Initialize RapidFire AI
rapidfireai init

# Start the RapidFire AI server
rapidfireai start
```

The dashboard will be available at `http://0.0.0.0:3000` where you can monitor and control experiments in real-time.

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
4. **Real-time Tracking**: All metrics visible in the dashboard at `http://localhost:3000`
5. **Interactive Control**: Stop, resume, or clone any configuration from the dashboard

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

**Example Notebook**: [SFT for Customer Support](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rf-tutorial-sft-chatqa-lite.ipynb)

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

**Example Notebook**: [DPO for Preference Alignment](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rf-tutorial-dpo-alignment-lite.ipynb)

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

**Example Notebook**: [GRPO for Math Reasoning](https://github.com/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rf-tutorial-grpo-mathreasoning-lite.ipynb)

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
- **Clone**: Duplicate a configuration with modifications
- **Clone & Warm Start**: Clone and initialize from parent's weights
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

RapidFire AI automatically detects and utilizes all available GPUs. No special configuration needed - the scheduler automatically distributes configurations across GPUs.

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

