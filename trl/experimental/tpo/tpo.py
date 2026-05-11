# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl[peft]",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Triple Preference Optimization (TPO) training.

TPO requires a *triple-preference* dataset where each example contains a `chosen`, a `rejected` and a `reference`
(gold) completion for the same prompt. Two dataset paths are supported out of the box:

- Use the published
  [`tpo-alignment/triple-preference-ultrafeedback-40K`](https://huggingface.co/datasets/tpo-alignment/triple-preference-ultrafeedback-40K)
  dataset directly. It already has the `prompt` / `reference` / `chosen` / `rejected` schema.
- Pass `--dataset_name openbmb/UltraFeedback` and the script automatically builds the triple-preference dataset as
  described in the TPO paper (Saeidi et al., 2025): the response with the highest `overall_score` becomes `reference`,
  the second-highest becomes `chosen`, and the lowest becomes `rejected`.

In both cases, if the dataset is in standard (plain-string) format it is auto-wrapped into the conversational format so
that the model's chat template is applied — this matches how Instruct models like `Qwen/Qwen3-0.6B` are trained.

Usage:

Full training:

```bash
python trl/experimental/tpo/tpo.py \
    --dataset_name tpo-alignment/triple-preference-ultrafeedback-40K \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --beta 0.01 \
    --tpo_alpha 1.0 \
    --output_dir Qwen3-0.6B-TPO \
    --no_remove_unused_columns
```

TPO-L (length-normalized variant with target reward margin):

```bash
python trl/experimental/tpo/tpo.py \
    --dataset_name tpo-alignment/triple-preference-ultrafeedback-40K \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --beta 0.01 \
    --tpo_alpha 1.0 \
    --loss_type tpo-l \
    --tpo_l_gamma 0.5 \
    --output_dir Qwen3-0.6B-TPO-L \
    --no_remove_unused_columns
```

LoRA:

```bash
python trl/experimental/tpo/tpo.py \
    --dataset_name tpo-alignment/triple-preference-ultrafeedback-40K \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --output_dir Qwen3-0.6B-TPO-LoRA \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.tpo import TPOConfig, TPOTrainer


def build_triple_preference_from_ultrafeedback(example):
    """
    Build a TPO triple-preference example from a raw UltraFeedback row.

    Following the TPO paper (Saeidi et al., 2025), completions are sorted by `overall_score` and we pick:
    - the highest-scored response as the gold `reference`,
    - the second-highest as `chosen`,
    - the lowest as `rejected`.

    Emits the *conversational* format so that [`TPOTrainer`] applies the model's chat template automatically (see
    `trl.data_utils.is_conversational`). Completions with a missing `overall_score` or `response` are filtered out; if
    fewer than 3 valid completions remain, the returned example contains `None` values and should be filtered out
    downstream.
    """
    scored = [c for c in example["completions"] if c.get("overall_score") is not None and c.get("response")]
    if len(scored) < 3:
        return {"prompt": None, "reference": None, "chosen": None, "rejected": None}
    scored.sort(key=lambda c: c["overall_score"], reverse=True)
    return {
        "prompt": [{"role": "user", "content": example["instruction"]}],
        "reference": [{"role": "assistant", "content": scored[0]["response"]}],
        "chosen": [{"role": "assistant", "content": scored[1]["response"]}],
        "rejected": [{"role": "assistant", "content": scored[-1]["response"]}],
    }


def to_conversational(example):
    """
    Wrap a standard-format triple-preference example (plain strings) in the *conversational* format, so that
    [`TPOTrainer`] applies the model's chat template automatically. This is the format expected by Instruct models; for
    non-Instruct base models the standard format can be used directly.
    """
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "reference": [{"role": "assistant", "content": example["reference"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Auto-build the triple-preference schema from raw UltraFeedback.
    first_example = next(iter(dataset[script_args.dataset_train_split]))
    if "completions" in first_example and "instruction" in first_example:
        dataset = dataset.map(
            build_triple_preference_from_ultrafeedback,
            remove_columns=list(first_example.keys()),
        )
        dataset = dataset.filter(lambda ex: ex["reference"] is not None)
        first_example = next(iter(dataset[script_args.dataset_train_split]))

    # Auto-wrap standard-format triple-preference data (plain strings) into conversational messages so the
    # model's chat template gets applied. This matches how Instruct models are trained and is what the TPO
    # paper's data preparation produces.
    if {"prompt", "chosen", "rejected", "reference"}.issubset(first_example) and isinstance(
        first_example["prompt"], str
    ):
        dataset = dataset.map(to_conversational)

    ################
    # Training
    ################
    trainer = TPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # train and save the model
    trainer.train()

    # Run a final evaluation pass and persist the metrics
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
