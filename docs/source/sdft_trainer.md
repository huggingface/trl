# SDFT

Self-Distilled Fine-Tuning (SDFT) is described in [Self-Training with On-Policy Self-Distillation for Language Model Alignment](https://huggingface.co/papers/2601.19897).

The TRL implementation adapts SDFT to the experimental trainer API while reusing the shared self-distillation infrastructure also used by SDPO.

In the current TRL implementation:

- the teacher is the model itself (base weights with adapter disabled for PEFT, or the same model under `no_grad` for non-PEFT); use `sync_ref_model=True` for an EMA teacher
- the dataset must provide both `prompt` and `privileged_context`
- `privileged_context` contains only the extra teacher-only information; the trainer combines it with `prompt` to build the teacher prompt
- `teacher_prompt_template` controls how `prompt` and `privileged_context` are combined into the teacher prompt
- on-policy generation can use either the student prompt or the teacher-conditioned prompt via `generate_from_teacher`
- `num_loss_tokens_to_skip` can exclude initial completion tokens from the distillation loss
- SDFT currently supports text-only training and does not support `use_vllm=True`
- the shared dataset contract is `prompt` plus `privileged_context`

## Usage

```python
from datasets import Dataset

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

dataset = Dataset.from_dict(
    {
        "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
        "privileged_context": ["Example answer: 4."],
    }
)

training_args = SDFTConfig(
    output_dir="sdft-model",
    distillation_alpha=0.5,
    distillation_topk=5,
    max_completion_length=64,
)

trainer = SDFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

To generate from the teacher-conditioned prompt instead of the student prompt, set `generate_from_teacher=True`.
To customize how the teacher prompt is built, set `teacher_prompt_template` on [`SDFTConfig`].

## Expected dataset columns

Each example must provide:

- `prompt`: the student-facing prompt
- `privileged_context`: only the extra teacher-only information, such as a demonstration, hint, or privileged feedback

Both standard text prompts and conversational prompts are supported by the trainer prompt handling.

## Callbacks

The trainer emits a small set of callback hooks that are useful for debugging, observability, and tests. These hooks are intended as practical integration points for experimental self-distillation workflows.

Shared self-distillation hooks:

- `on_self_distillation_batch_prepared`: fired when a self-distillation batch is ready. The payload includes `prompt_ids`, `completion_ids`, and `old_per_token_logps` when importance-sampling clipping inputs are available.
- `on_generation_batch_built`: fired when a new buffered generation batch is created. The payload includes `generate_every` and `steps_per_generation`.

SDFT-specific hook:

- `on_generation_prompts_selected`: fired when SDFT chooses the prompt source for on-policy generation. The payload includes the selected `generation_prompts` and the corresponding `generation_prompt_text`.

## SDFTConfig

[[autodoc]] experimental.sdft.SDFTConfig

## SDFTTrainer

[[autodoc]] experimental.sdft.SDFTTrainer
    - train
    - save_model
    - push_to_hub
