# Paper Index

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Group Sequence Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2507.18071

GSPO is a GRPO variant that computes importance sampling weights at the sequence level instead of per-token. To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    beta=0.0,  # GSPO set kl regularization to zero: https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306 
    epsilon=3e-4,  # GSPO paper (v2), section 5.1
    epsilon_high=4e-4,  # GSPO paper (v2), section 5.1
    gradient_accumulation_steps=1,
    steps_per_generation=4,  # partition rollout batch into 4 mini-batches. GSPO paper (v2), section 5.1. Must be 4 times gradient_accumulation_steps
)
```

## DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**📜 Paper**: https://huggingface.co/papers/2503.14476

The DAPO algorithm, which includes decoupled clip and dynamic sampling policy optimization, enables open-source, high-performance reinforcement learning training for large-scale language models, enhancing reproducibility in the field. To reproduce the paper's setting, use this configuration (we don't support dynamic sampling right now):

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    loss_type="dapo",
    per_device_train_batch_size=512, # mini-batch size for training in the paper, DAPO paper: section 4.1
    num_generations = 16, # number of sample responses in the paper, DAPO paper: section 4.1
    max_completion_length = 20480, #  maximum number of tokens for generation in the paper, DAPO paper: section 4.1
    epsilon_high = 0.28, # DAPO paper: section 4.1
    epsilon = 0.2, # DAPO paper: section 4.1,
    mask_truncated_completions = True
)
```

## AlphaPO -- Reward shape matters for LLM alignment

**📜 Paper**: https://huggingface.co/papers/2501.03884

AlphaPO is a new Direct Alignment Algorithms (DAAs) method that leverages an alpha-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. To reproduce the paper's setting, use this configuration:

```python
from trl import CPOConfig

# Mistral-Instruct from Table 3 of the paper
training_args = CPOConfig(
    loss_type="alphapo",
    alpha=0.25,
    beta=2.5,
    simpo_gamma=0.1,
    learning_rate=7e-7,
    ...
)
```

## EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes

**📜 Paper**: https://huggingface.co/papers/2508.00180

Bias-Corrected Exponential Moving Average (BEMA) improves the stability and efficiency of language model fine-tuning by reducing stochasticity and eliminating bias. To use BEMA with SFT as described in the paper, you can use the [`BEMACallback`]:

```python
from trl import BEMACallback, SFTTrainer

trainer = SFTTrainer(
    ...
    callbacks=[BEMACallback()],
)
```

## Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)

**📜 Paper**: https://huggingface.co/papers/2508.08221

The authors of this paper find that the combination of:

1. scaling rewards by the standard deviation computed over the entire batch and
2. aggregating loss over the total number of tokens

can unlock the learning capability of critic-free policies using vanilla PPO loss. Their results demonstrate that this simple combination consistently improves performance, surpassing strategies like GRPO and [DAPO](https://huggingface.co/papers/2503.14476).

TRL supports using these learnings to train a GRPO model by:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...
    scale_rewards="group",
    loss_type="bnpo",
    # Other parameters used
    beta=0.0,  # = init_kl_coef in the paper
    top_p=0.99,
    top_k=100,
    temperature=0.99,
    num_completions=8, # = num_return_sequences in the paper
    num_iterations=1,  # = ppo_epochs in the paper
    per_device_train_batch_size=4
    gradient_accumulation_steps=32,
    steps_per_generation=8,  # (rollout_batch_size*num_return_sequences) / (per_device_train_batch_size*gradient_accumulation_steps)
)
```

Note that when using gradient accumulation, the loss is aggregated over the total number of tokens in the batch, but not over the accumulated batch. For more details, see the [GRPO Trainer - Loss types](grpo_trainer#loss_types).
