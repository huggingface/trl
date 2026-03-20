# SDPO

Self-Distillation Policy Optimization (SDPO) was introduced in [Reinforcement Learning via Self-Distillation](https://huggingface.co/papers/2601.20802) by [Jonas Hübotter](https://huggingface.co/jonhue), Frederike Lübeck, Lejs Behric, [Anton Baumann](https://huggingface.co/antonbaumann), Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause.

> Large language models are increasingly post-trained with reinforcement learning in verifiable domains such as code and math. Yet, current methods for reinforcement learning with verifiable rewards (RLVR) learn only from a scalar outcome reward per attempt, creating a severe credit-assignment bottleneck. Many verifiable environments actually provide rich textual feedback, such as runtime errors or judge evaluations, that explain why an attempt failed. We formalize this setting as reinforcement learning with rich feedback and introduce Self-Distillation Policy Optimization (SDPO), which converts tokenized feedback into a dense learning signal without any external teacher or explicit reward model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed next-token predictions back into the policy. In this way, SDPO leverages the model's ability to retrospectively identify its own mistakes in-context. Across scientific reasoning, tool use, and competitive programming on LiveCodeBench v6, SDPO improves sample efficiency and final accuracy over strong RLVR baselines. Notably, SDPO also outperforms baselines in standard RLVR environments that only return scalar feedback by using successful rollouts as implicit feedback for failed attempts. Finally, applying SDPO to individual questions at test time accelerates discovery on difficult binary-reward tasks, achieving the same discovery probability as best-of-k sampling or multi-turn conversations with 3x fewer attempts.

The SDPO trainer is built on TRL's experimental shared self-distillation stack. It keeps the online rollout-and-reward training flow, then builds a teacher-conditioned view of the same completions from successful rollouts and optional environment feedback.

In the current TRL implementation:

- the default SDPO policy loss mode is `distillation_only`
- `hybrid` mode is also available to combine the base policy loss with the self-distillation loss
- supported teacher regularization modes are `ema` and `none`
- `distillation_topk` is only valid when `full_logit_distillation=True`
- when `full_logit_distillation=False`, SDPO uses token-level reverse KL and requires `distillation_alpha=1.0`
- environment feedback can be injected into teacher reprompts when the dataset exposes a `privileged_context` column

## Expected dataset columns

Each example must provide:

- `prompt`: the student-facing prompt
- `privileged_context`: optional privileged text, such as environment feedback, used when `include_environment_feedback=True`

## Usage

```python
from datasets import Dataset

from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["Your earlier answer used the wrong format."],
    }
)

training_args = SDPOConfig(
    output_dir="sdpo-model",
    distillation_topk=100,                 # Top-K logit distillation approximation
    full_logit_distillation=True,          # Required for top-K; enables non-reverse divergences
    include_environment_feedback=True,     # Use dataset privileged_context for teacher reprompts
)

trainer = SDPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_func,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

SDPO always requires a `prompt` column. To use environment feedback, also include a `privileged_context` column and set `include_environment_feedback=True`. SDPO will use successful rollouts and, when enabled, that text to build teacher reprompts for self-distillation.

## Callbacks

The trainer emits a small set of callback hooks that are useful for debugging, observability, and tests. These hooks are intended as practical integration points for experimental self-distillation workflows.

Shared self-distillation hooks:

- `on_self_distillation_batch_prepared`: fired when a self-distillation batch is ready. The payload includes `prompt_ids`, `completion_ids`, and `old_per_token_logps` when importance-sampling clipping inputs are available.
- `on_generation_batch_built`: fired when a new buffered generation batch is created. The payload includes `generate_every` and `steps_per_generation`.

SDPO-specific hook:

- `on_teacher_context_built`: fired after SDPO constructs the teacher-conditioned inputs. The payload includes `teacher_input_ids`, `teacher_attention_mask`, `completion_mask`, and `self_distillation_mask`.

## SDPOConfig

[[autodoc]] experimental.sdpo.SDPOConfig

## SDPOTrainer

[[autodoc]] experimental.sdpo.SDPOTrainer
    - train
    - save_model
    - push_to_hub
