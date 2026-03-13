# DPPO Trainer

TRL supports the Decoupled Proximal Policy Optimization (DPPO) algorithm, which is a variant of PPO that decouples the optimization of the policy and value function for improved training stability. This implementation is based on the [Stable-RL](https://github.com/sail-sg/Stable-RL) paper.

## Overview

DPPO (Decoupled Proximal Policy Optimization) addresses some of the training stability issues in standard PPO by separating the optimization of the policy and value function. The key improvements include:

1. **Separate Optimizers**: Policy and value function use independent optimizers with different learning rates
2. **Decoupled Training Loops**: Policy and value function are trained in separate phases
3. **Configurable Update Frequency**: Value function updates can occur less frequently than policy updates
4. **Improved Stability**: The decoupling reduces interference between policy and value function optimization

## Quick Start

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl.experimental.dppo import DPPOConfig, DPPOTrainer

# Load models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped", padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

policy = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped")
ref_policy = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped")
value_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-1b-deduped", num_labels=1)
reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-1b-deduped", num_labels=1)

# Load and prepare dataset
dataset = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")

def tokenize(element):
    outputs = tokenizer(element["prompt"], padding=False)
    return {"input_ids": outputs["input_ids"]}

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Configure DPPO training
config = DPPOConfig(
    output_dir="dppo_model",
    learning_rate=3e-6,
    vf_learning_rate=1e-4,  # Separate learning rate for value function
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    total_episodes=10000,
    num_ppo_epochs=4,
    num_vf_epochs=1,  # Number of value function training epochs
    vf_update_frequency=1,  # Update value function every N policy updates
)

# Create trainer
trainer = DPPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=dataset,
)

# Train
trainer.train()
```

## Key Configuration Parameters

DPPO inherits most configuration from PPO but adds specific parameters for decoupled optimization:

### DPPO-Specific Parameters

- **`vf_learning_rate`** (`float`, *optional*): Learning rate for the value function optimizer. If not set, uses the same learning rate as the policy. Typically set higher than the policy learning rate for faster value function convergence.
  
- **`num_vf_epochs`** (`int`, defaults to `1`): Number of epochs to train the value function per batch. Setting this > 1 allows the value function to be trained more thoroughly.

- **`vf_update_frequency`** (`int`, defaults to `1`): How often to update the value function (every N policy updates). Setting this > 1 further decouples the training.

### Example Configurations

**Standard DPPO** (decoupled learning rates):
```python
config = DPPOConfig(
    learning_rate=3e-6,           # Policy learning rate
    vf_learning_rate=1e-4,        # Higher value function learning rate
    num_ppo_epochs=4,
    num_vf_epochs=1,
    vf_update_frequency=1,
)
```

**Aggressive Decoupling** (less frequent value updates):
```python
config = DPPOConfig(
    learning_rate=3e-6,
    vf_learning_rate=1e-4,
    num_ppo_epochs=4,
    num_vf_epochs=2,              # More value training per update
    vf_update_frequency=2,        # Update value every 2 policy updates
)
```

## Differences from PPO

| Feature | PPO | DPPO |
|---------|-----|------|
| Optimizer | Single optimizer for both policy and value | Separate optimizers |
| Learning Rate | Single learning rate | Independent learning rates |
| Training Loop | Coupled optimization | Decoupled phases |
| Update Frequency | Same for policy and value | Configurable for value |
| Stability | Can have interference | Improved stability |

## Example Script

A complete example script is available at [`examples/scripts/ppo/dppo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/dppo.py).

Run it with:

```bash
python examples/scripts/ppo/dppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --vf_learning_rate 1e-4 \
    --output_dir pythia-1b-dppo \
    --per_device_train_batch_size 64 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped
```

Or with DeepSpeed:

```bash
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/dppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --vf_learning_rate 1e-4 \
    --output_dir pythia-1b-dppo \
    --model_name_or_path EleutherAI/pythia-1b-deduped
```

## When to Use DPPO

Consider using DPPO over standard PPO when:

1. **Training is unstable**: DPPO's decoupled optimization can help stabilize training
2. **Value function converges slowly**: The separate learning rate and additional epochs can help
3. **Policy and value interference**: When you suspect the two objectives are interfering with each other
4. **Large-scale training**: DPPO's stability improvements are more pronounced at scale

## References

- [Stable-RL Repository](https://github.com/sail-sg/Stable-RL)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## DPPOConfig

[[autodoc]] experimental.dppo.DPPOConfig

## DPPOTrainer

[[autodoc]] experimental.dppo.DPPOTrainer
