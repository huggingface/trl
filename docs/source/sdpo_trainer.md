# SDPO

Self-Distillation Policy Optimization (SDPO) was introduced in [Reinforcement Learning via Self-Distillation](https://huggingface.co/papers/2601.20802) by Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause.

> Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain why an attempt failed. We formalize this setting as reinforcement learning with rich feedback and introduce Self-Distillation Policy Optimization (SDPO), which converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy. In this way, SDPO leverages the model's ability to retrospectively identify its own mistakes in-context. Across scientific reasoning, tool use, and competitive programming on LiveCodeBench v6, SDPO improves sample efficiency and final accuracy over strong RLVR baselines. Notably, SDPO also outperforms baselines in standard RLVR environments that only return scalar feedback by using successful rollouts as implicit feedback for failed attempts. Finally, applying SDPO to individual questions at test time accelerates discovery on difficult binary-reward tasks, achieving the same discovery probability as best-of-k sampling or multi-turn conversations with 3x fewer attempts.

The SDPO trainer extends [`GRPOTrainer`] with a self-distillation loss. The key idea is to use the model's own successful rollouts (or feedback-conditioned predictions) as a teacher signal, distilling them back into the policy via a generalized Jensen-Shannon divergence with top-K distillation and importance sampling clipping.

## Usage

```python
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

training_args = SDPOConfig(
    output_dir="sdpo-model",
    distillation_alpha=0.5,          # Jensen-Shannon divergence (recommended)
    distillation_topk=100,           # Top-K distillation
    distillation_is_clip=2.0,        # Importance sampling clipping
    distillation_weight=1.0,         # Weight for self-distillation loss
    use_successful_as_teacher=True,  # Use successful rollouts as teacher
    ema_update_rate=0.05,            # Teacher EMA update rate
    ...
)

trainer = SDPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_func,
    args=training_args,
)
trainer.train()
```

## SDPOConfig

[[autodoc]] experimental.sdpo.SDPOConfig

## SDPOTrainer

[[autodoc]] experimental.sdpo.SDPOTrainer
    - train
    - save_model
    - push_to_hub
