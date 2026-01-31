# Self-Distillation Fine-Tuning (SDFT) Trainer

## Overview

Self-Distillation Fine-Tuning (SDFT) is described in [Self-Distillation for Language Models](https://arxiv.org/pdf/2601.19897).
SDFT trains a student model using a teacher model on the student's generated completions, using a divergence between
student and teacher distributions.

The abstract from the paper is the following:

> Continual learning, enabling models to acquire new skills and knowledge without degrading existing capabilities, remains a fundamental challenge for foundation models. While on-policy reinforcement learning can reduce forgetting, it requires explicit reward functions that are often unavailable. Learning from expert demonstrations, the primary alternative, is dominated by supervised fine-tuning (SFT), which is inherently offpolicy. We introduce Self-Distillation Fine-Tuning (SDFT), a simple method that enables on-policy learning directly from demonstrations. SDFT leverages in-context learning by using a demonstration-conditioned model as its own teacher, generating on-policy training signals that preserve prior capabilities while acquiring new skills. Across skill learning and knowledge acquisition tasks, SDFT consistently outperforms SFT, achieving higher new-task accuracy while substantially reducing catastrophic forgetting. In sequential learning experiments, SDFT enables a single model to accumulate multiple skills over time without performance regression, establishing on-policy distillation as a practical path to continual learning from demonstrations. 

> [!WARNING]
> **Experimental:** APIs under `trl.experimental` may change or be removed without notice.

## Usage tips

- Provide a teacher model via `ref_model`. If you omit it, the trainer will create a teacher from the same checkpoint
  as the student.
- Your dataset must contain `prompt` and `teacher_prompt`. If you do not have distinct teacher prompts, set
  `teacher_prompt = prompt`.
- Set `generate_from_teacher=True` to generate completions using the teacher model instead of the student.

## Quick Start

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.sdft import SDFTConfig, SDFTTrainer

student_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

train_dataset = Dataset.from_dict(
    {
        "prompt": ["Write a haiku about the ocean."],
        "teacher_prompt": ["Write a haiku about the ocean."],
    }
)

training_args = SDFTConfig(output_dir="sdft-model", per_device_train_batch_size=1)
trainer = SDFTTrainer(
    model=student_model,
    ref_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```

### Expected dataset type

The dataset must be formatted with the following columns:

- `prompt`: text or conversational messages for the student input.
- `teacher_prompt`: text or conversational messages for the teacher input.

## SDFTTrainer

[[autodoc]] experimental.sdft.SDFTTrainer
    - train
    - save_model
    - push_to_hub

## SDFTConfig

[[autodoc]] experimental.sdft.SDFTConfig
