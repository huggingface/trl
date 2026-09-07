# IW-OPD Trainer

The Iterative Weighted Offline Preference Distillation (IW-OPD) Trainer implements the IW-OPD algorithm for training language models from offline preference datasets without online reward model queries.

## Paper

**📜 Paper**: https://huggingface.co/papers/2410.20629

**Title**: Iterative Weighted Offline Preference Distillation

**Authors**: Anonymous (under review)

## Overview

IW-OPD addresses the challenge of training from offline preference data by iteratively reweighting the dataset based on the current policy's likelihood of generating the preferred response. This approach avoids the need for online reward model evaluation while maintaining strong performance on preference alignment tasks.

## Installation

```bash
pip install trl
```

## Usage

```python
from trl import IWOPDConfig, IWOPOTrainer
from datasets import load_dataset

# Load an offline preference dataset
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Configure IW-OPD training
training_args = IWOPDConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    max_length=1024,
    max_prompt_length=512,
    # IW-OPD specific parameters
    iw_opd_beta=0.1,      # Weight temperature
    iw_opd_gamma=0.9,     # Discount factor for iterative reweighting
    iw_opd_num_iterations=3,  # Number of reweighting iterations
)

trainer = IWOPOTrainer(
    model="meta-llama/Llama-3.1-8B-Instruct",
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `iw_opd_beta` | Temperature for importance weight computation | 0.1 |
| `iw_opd_gamma` | Discount factor for iterative reweighting | 0.9 |
| `iw_opd_num_iterations` | Number of reweighting iterations | 3 |
| `loss_type` | Loss function type (`"sigmoid"`, `"ipo"`, `"kto"`) | `"sigmoid"` |

## How It Works

1. **Initialization**: Start with uniform weights over the offline preference dataset
2. **Policy Training**: Train the policy model on the weighted dataset
3. **Reweighting**: Compute new weights based on the current policy's likelihood ratio
4. **Iteration**: Repeat steps 2-3 for `iw_opd_num_iterations` iterations

The weight for each preference pair $(x, y_w, y_l)$ at iteration $t$ is:
$$w^{(t)}(x, y_w, y_l) \propto \exp\left(\frac{1}{\beta} \log \frac{\pi^{(t-1)}(y_w|x)}{\pi^{(t-1)}(y_l|x)}\right)$$

## Expected Dataset Format

The trainer expects a dataset with the following columns:

- `prompt`: The input prompt
- `chosen`: The preferred response
- `rejected`: The dispreferred response

## References

- [Iterative Weighted Offline Preference Distillation](https://huggingface.co/papers/2410.20629)
- [TRL DPO Trainer](../dpo_trainer) - For standard DPO training
- [TRL KTO Trainer](../kto_trainer) - For KTO training