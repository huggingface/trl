# General Online Logit Distillation (GOLD) Trainer

[![All_models-GOLD-blue](https://img.shields.io/badge/All_models-GOLD-blue)](https://huggingface.co/models?other=sft,gold)

## Overview

General Online Logit Distillation (GOLD) is an extension of Universal Logit Distillation (ULD) that supports
student/teacher pairs with different tokenizers. It aligns the textual spans produced by both tokenizers and merges the
associated logits so no completion tokens are dropped. This enables cross-tokenizer knowledge distillation, including
mixed model families (for example, LLaMA students with Qwen teachers).

Key capabilities:

1. **Cross-tokenizer alignment** – GOLD incrementally decodes the student and teacher tokens, groups passages with the same visible text, and merges probabilities inside each group. This guarantees loss terms are computed over the full completion even when token boundaries differ.
2. **Hybrid ULD loss** – when `uld_use_hybrid_loss` is enabled, GOLD compares exact vocabulary matches directly and falls back to the original sorted-probability ULD loss for unmatched tokens. This improves stability for students whose vocabularies only partially overlap with the teacher.
3. **Seamless integration with GKD** – GOLD inherits the on-policy vs. off-policy scheduling from the [`experimental.gkd.GKDTrainer`], so you can combine sequence-level KD, generalized JSD, and cross-tokenizer distillation in a single training run.

> [!NOTE]
> GOLD is currently part of the `trl.experimental` namespace. APIs may change without notice while the feature is iterated on.

## Usage tips

The [`GOLDTrainer`] subclasses [`SFTTrainer`] and accepts the same datasets as other TRL trainers (lists of ChatML style
messages). Important configuration flags on [`GOLDConfig`] include:

* `use_uld_loss` – toggles Universal Logit Distillation. Set this to `True` for cross-tokenizer setups.
* `teacher_tokenizer_name_or_path` – required when `use_uld_loss=True`; GOLD uses the teacher tokenizer to align tokens.
* `uld_use_hybrid_loss`, `uld_hybrid_matched_weight`, `uld_hybrid_unmatched_weight` – enables and weights the hybrid
  matched/unmatched loss.
* `beta`, `lmbda`, `seq_kd` – inherited from [`experimental.gkd.GKDConfig`], controlling the generalized JSD interpolation and on-policy
  sampling ratio.
* `use_privileged_context`, `teacher_prompt_template`, `privileged_context_column` – lets the teacher see extra
  per-example context while the student prompt remains unchanged. This is currently limited to text, off-policy,
  standard-JSD training.
* `num_generations`, `generation_batch_size` – control buffered rollout generation across gradient accumulation windows.
  `generation_batch_size` is the number of unique prompts per worker per optimizer step.

A minimal end-to-end example:

```python
from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer

train_dataset = load_dataset(
    "HuggingFaceTB/OpenR1-Math-220k-default-verified",
    "all",
    split="train[:1024]",
)

trainer = GOLDTrainer(
    model="meta-llama/Llama-3.2-1B-Instruct",
    teacher_model="Qwen/Qwen2.5-0.5B-Instruct",
    args=GOLDConfig(output_dir="gold-model", use_uld_loss=True, teacher_tokenizer_name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
    train_dataset=train_dataset,
)
trainer.train()
```

For quick-start workflows you can rely on string identifiers as shown above—the trainer will load the model and tokenizer for you. Explicitly instantiating `AutoModelForCausalLM`, `AutoTokenizer`, or populating `GOLDConfig` is recommended only for advanced use cases where you need fine-grained control over initialization.

A more explicit setup might look like this when you need to customise model loading, tokenizer settings, or training arguments:

```python
from datasets import load_dataset
from trl.experimental.gold import GOLDConfig, GOLDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "meta-llama/Llama-3.2-1B-Instruct"
teacher_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(student_name)
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name)

train_dataset = load_dataset(
    "HuggingFaceTB/Countdown-Task-GOLD",
    "verified_Qwen2.5-0.5B-Instruct",
    split="train",
)

training_args = GOLDConfig(
    output_dir="gold-model",
    per_device_train_batch_size=1,
    teacher_model_name_or_path=teacher_name,
    teacher_tokenizer_name_or_path=teacher_name,
    use_uld_loss=True,
    uld_use_hybrid_loss=True,
)

trainer = GOLDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```

> [!NOTE]
> GOLD buffers one full optimizer-window generation batch (`per_device_train_batch_size * gradient_accumulation_steps`)
> and reuses it across accumulation steps. If the final batch is undersized, GOLD warns and drops that last batch
> (`Dropping last batch due to unexpected batch size`). Set `dataloader_drop_last=True` to avoid this warning.

### Expected dataset type

GOLD requires a [conversational](dataset_formats#conversational) [language modeling](dataset_formats#language-modeling) dataset, e.g.:

```python
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}
```

`GOLDTrainer` keeps the raw messages so the ChatML collator can construct prompts and completions with the correct
boundaries.

## Privileged Context

Privileged-context distillation gives the teacher extra information that is unavailable to the student at inference
time. Add a context column to a normal prompt/completion dataset, then enable the feature explicitly:

```python
from datasets import Dataset

dataset = Dataset.from_list(
    [
        {
            "prompt": [{"role": "user", "content": "What caused the outage?"}],
            "completion": [{"role": "assistant", "content": "A timeout change caused the outage."}],
            "privileged_context": "The deployment changed the timeout immediately before errors began.",
        }
    ]
)

args = GOLDConfig(
    output_dir="gold-model",
    lmbda=0.0,
    use_privileged_context=True,
)
```

The student is trained on the original prompt and completion. The teacher receives a separate prompt built from
`"{prompt}\n\n{privileged_context}"` by default, then scores the exact same completion. For conversational data,
GOLD rewrites only the final user turn before reapplying the chat template; system, tool, and assistant turns remain
unchanged. Set `teacher_prompt_template` to customize this prompt, and `privileged_context_column` to use a column
other than `"privileged_context"`.

The teacher prompt is never truncated, so ensure that it fits within the teacher model's context window. This initial
support is intentionally limited to text, off-policy standard JSD distillation: `lmbda` must be `0`, and
`seq_kd=True`, `use_uld_loss=True`, packing, VLMs, and Liger Kernel are not supported with privileged context.

## How Token Merging Works

When student and teacher use different tokenizers, the same text may be split differently:

- **Student**: `"Hugging Face"` → 1 token
- **Teacher**: `"Hugging"`, `" Face"` → 2 tokens

GOLD aligns these sequences and merges the teacher's multi-token probabilities into a single distribution that can be compared with the student's single-token distribution.

### Probability Merging

For a teacher sequence of tokens `[token₀, token₁, ..., tokenₖ]` that maps to a single student token, GOLD computes:

```
P_merged(y) = P(y | context) × P(token₁ | token₀, context) × ... × P(tokenₖ | ..., context)
```

where:
- `P(y | context)` is the marginal probability distribution over all vocabulary tokens at the first position
- `P(tokenᵢ | ..., context)` are **scalar** conditional probabilities of the actual tokens that were generated

**Key insight**: Only the conditional probabilities of the **actual continuation tokens** are extracted as scalars. The full marginal distribution at the first position is then scaled by multiplying these scalar probabilities.

This ensures:
1. **Correct joint probability** for the actual generated sequence (by the chain rule)
2. **Reasonable approximation** for counterfactual tokens (scaled by the same continuation likelihood)
3. **Unnormalized distributions** that preserve the correct relative probabilities for ULD loss computation

### Example

Given:
```
P(x₀):         ["HF": 0.6,  "is": 0.3,  "cool": 0.1]
P(x₁ | "HF"):  ["HF": 0.05, "is": 0.9,  "cool": 0.05]
```

If tokens 0 and 1 are merged, and the actual sequence was `["HF", "is"]`:
```
P_merged("HF")   = 0.6 × 0.9 = 0.54  ✓ (correct joint probability)
P_merged("is")   = 0.3 × 0.9 = 0.27
P_merged("cool") = 0.1 × 0.9 = 0.09
```

The merged distribution is unnormalized (sums to 0.81), but this is intentional and correct for ULD loss computation, which uses sorting and L1 distance.

## Example script

Use [`examples/scripts/gold.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/gold.py) to launch GOLD training from the command line. The script supports full training and LoRA via the standard `ModelConfig` flags.

```bash
python examples/scripts/gold.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gold-model \
    --num_train_epochs 1 \
    --push_to_hub
```

## Training Vision Language Models

[`GOLDTrainer`] supports VLM-to-VLM distillation. Both student and teacher must be vision-language models. To train a VLM, provide a dataset with either an `image` column (single image per sample) or an `images` column (list of images per sample). For more information on the expected dataset structure, see the [Dataset Format — Vision datasets](dataset_formats#vision-datasets) section.

When the student and teacher share the same architecture and tokenizer (e.g. Qwen3-VL-8B to Qwen3-VL-2B), the standard generalized JSD loss applies directly. When they have different `model_type` (e.g. Qwen3-VL to LFM2.5-VL), set `use_uld_loss=True` to enable cross-tokenizer alignment via Universal Logit Distillation. Images are processed separately through each model's processor.

```python
from datasets import load_dataset
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl.experimental.gold import GOLDConfig, GOLDTrainer

student_name = "Qwen/Qwen3-VL-2B-Instruct"
teacher_name = "Qwen/Qwen3-VL-8B-Instruct"

processor = AutoProcessor.from_pretrained(student_name, padding_side="left")
student_model = AutoModelForImageTextToText.from_pretrained(student_name, dtype=torch.bfloat16)
teacher_model = AutoModelForImageTextToText.from_pretrained(teacher_name, dtype=torch.bfloat16)

train_dataset = load_dataset("trl-lib/llava-instruct-mix", split="train")

trainer = GOLDTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=GOLDConfig(
        output_dir="gold-vlm-model",
        max_length=None,
        teacher_model_name_or_path=teacher_name,
        use_uld_loss=False,
    ),
    train_dataset=train_dataset,
    processing_class=processor,
)
trainer.train()
```

For cross-family distillation, set `use_uld_loss=True` and `teacher_tokenizer_name_or_path` to the teacher model name.

Use [`trl/experimental/gold/gold_vlm.py`](https://github.com/huggingface/trl/blob/main/trl/experimental/gold/gold_vlm.py) to launch GOLD VLM training from the command line:

```bash
# Same-family distillation (JSD loss, vLLM enabled)
accelerate launch trl/experimental/gold/gold_vlm.py \
    --student_model_name Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct

# Cross-family distillation (ULD loss, local generation)
accelerate launch trl/experimental/gold/gold_vlm.py \
    --student_model_name LiquidAI/LFM2.5-VL-1.6B \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct \
    --use_uld_loss \
    --no-use_vllm
```

> [!TIP]
> For VLMs, `truncation_mode='keep_end'` is not supported because image tokens reside in the prompt portion of the sequence and may be silently dropped. Use `truncation_mode='keep_start'` (the default) or set `max_length=None` in the [`GOLDConfig`]. This allows the model to process the full sequence length without truncating image tokens.
>
> ```python
> GOLDConfig(max_length=None, ...)
> ```
>
> Only use `max_length` when you've verified that truncation won't remove image tokens for the entire dataset.

> [!NOTE]
> Cross-architecture VLM distillation requires `use_uld_loss=True`. The trainer will raise an error if you attempt cross-architecture distillation without ULD loss.

## GOLDTrainer

[[autodoc]] experimental.gold.GOLDTrainer
    - train
    - generate_on_policy_outputs
    - save_model
    - push_to_hub

## GOLDConfig

[[autodoc]] experimental.gold.GOLDConfig
