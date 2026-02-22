# DPPO (Decoupled Proximal Policy Optimization)

Implementation of DPPO from [Stable-RL](https://github.com/sail-sg/Stable-RL).

## Overview

DPPO is a variant of PPO that decouples the optimization of the policy and value function for improved training stability. The key innovation is separating the policy and value function optimization into independent phases with separate optimizers and learning rates.

## Key Features

- **Separate Optimizers**: Independent optimizers for policy and value function
- **Decoupled Learning Rates**: Different learning rates for policy and value function optimization
- **Independent Training Phases**: Policy and value function trained separately 
- **Configurable Update Frequency**: Control how often value function is updated
- **Improved Stability**: Reduced interference between policy and value optimization

## Usage

```python
from trl.experimental.dppo import DPPOConfig, DPPOTrainer

config = DPPOConfig(
    learning_rate=3e-6,        # Policy learning rate
    vf_learning_rate=1e-4,     # Value function learning rate (typically higher)
    num_ppo_epochs=4,          # Policy training epochs
    num_vf_epochs=1,           # Value function training epochs
    vf_update_frequency=1,     # Update value every N policy updates
)

trainer = DPPOTrainer(
    args=config,
    model=policy_model,
    ref_model=reference_model,
    value_model=value_model,
    reward_model=reward_model,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
```

## Configuration Parameters

### DPPO-Specific
- `vf_learning_rate`: Learning rate for value function (default: same as policy)
- `num_vf_epochs`: Number of value function training epochs per batch (default: 1)
- `vf_update_frequency`: Update value every N policy updates (default: 1)

### Inherited from PPO
All standard PPO parameters are supported (num_ppo_epochs, cliprange, etc.)

## Example

See [`examples/scripts/ppo/dppo.py`](../../../examples/scripts/ppo/dppo.py) for a complete example.

## References

- [Stable-RL Repository](https://github.com/sail-sg/Stable-RL)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
