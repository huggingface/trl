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
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Simple Self-Distillation (SSD) training for code generation.

Implements the method from "Embarrassingly Simple Self-Distillation Improves Code Generation" (Zhang et al., 2026):
sample completions from the model at a training-time temperature and truncation, then fine-tune on those raw,
unverified samples with standard cross-entropy loss. No reward model, verifier, teacher, or RL needed.

The dataset only requires a ``prompt`` column containing coding problem prompts.

Example:

```bash
python trl/experimental/ssd/ssd.py \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_name microsoft/rStar-Coder \
    --dataset_config seed_sft \
    --prompt_column question \
    --output_dir outputs/ssd-qwen3-4b \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --max_prompt_length 1024 \
    --max_completion_length 65536 \
    --temperature 1.6 \
    --top_k 20 \
    --top_p 0.8 \
    --num_train_epochs 1 \
    --bf16 \
    --report_to trackio
```
"""

from dataclasses import dataclass, field

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.ssd import SSDConfig, SSDTrainer


@dataclass
class SSDScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides `dataset_name`."},
    )
    prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column containing the problem prompts."},
    )


if __name__ == "__main__":
    parser = TrlParser((SSDScriptArguments, SSDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if model_args.model_name_or_path is None:
        raise ValueError("`model_name_or_path` is required.")

    if model_args.dtype in ["auto", None]:
        if training_args.bf16:
            dtype = torch.bfloat16
        elif training_args.fp16:
            dtype = torch.float16
        else:
            dtype = "auto"
    else:
        dtype = getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if script_args.dataset_path is not None:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            streaming=script_args.dataset_streaming,
        )

    # Ensure the dataset has a `prompt` column
    def _prepare_split(ds):
        if script_args.prompt_column != "prompt" and script_args.prompt_column in ds.column_names:
            ds = ds.rename_column(script_args.prompt_column, "prompt")
        return ds.select_columns(["prompt"])

    if isinstance(dataset, DatasetDict):
        train_dataset = _prepare_split(dataset[script_args.dataset_train_split])
        eval_dataset = None
        if script_args.dataset_test_split in dataset:
            eval_dataset = _prepare_split(dataset[script_args.dataset_test_split])
    else:
        train_dataset = _prepare_split(dataset)
        eval_dataset = None

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False if training_args.gradient_checkpointing else True

    trainer = SSDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)
