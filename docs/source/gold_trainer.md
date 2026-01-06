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
from trl import GOLDConfig, GOLDTrainer
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
    teacher_model=teacher_name,
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

### Expected dataset type

GOLD requires a [conversational](dataset_formats#conversational) [language modeling](dataset_formats#language_modeling) dataset, e.g.:

```python
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}
```

`GOLDTrainer` keeps the raw messages so the ChatML collator can construct prompts and completions with the correct
boundaries.

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

## GOLDTrainer

[[autodoc]] experimental.gold.GOLDTrainer
    - train
    - generate_on_policy_outputs
    - save_model
    - push_to_hub

## GOLDConfig

[[autodoc]] experimental.gold.GOLDConfig
