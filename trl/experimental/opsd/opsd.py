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

"""
Train a model with On-Policy Self-Distillation (OPSD).

The dataset must provide a problem and its ground-truth solution. Each example needs either:

- `prompt` and `privileged_context` columns (the shared self-distillation contract), or
- gsm8k-style `question`/`problem` and `answer`/`solution` columns, which are mapped automatically. Use
  `--solution_column` to pick a differently named solution column.

Example (mirrors the official Qwen3-1.7B configuration):

```bash
accelerate launch trl/experimental/opsd/opsd.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --dataset_name open-thoughts/OpenThoughts-114k \
    --output_dir opsd-qwen3-1.7b \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_completion_length 1024 \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --dtype bfloat16 \
    --bf16 true
```
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.opsd import OPSDConfig, OPSDTrainer


@dataclass
class OPSDScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional local dataset path to load with `load_from_disk`. Overrides `dataset_name`."},
    )
    eval_num_prompts: int | None = field(
        default=8,
        metadata={"help": "Number of prompts to log during evaluation. Set to 0 to disable completion logging."},
    )
    solution_column: str = field(
        default="solution",
        metadata={"help": "Dataset column holding the ground-truth solution, forwarded as `privileged_context`."},
    )


def _to_opsd_example(example: dict[str, Any], solution_column: str) -> dict[str, Any]:
    prompt = example.get("prompt")
    if prompt is None and "problem" in example:
        prompt = [{"role": "user", "content": example["problem"]}]
    if prompt is None and "question" in example:
        prompt = [{"role": "user", "content": example["question"]}]
    if prompt is None:
        raise ValueError("Each example must provide one of: `prompt`, `problem`, or `question`.")

    solution = example.get("privileged_context")
    if solution is None:
        solution = example.get(solution_column)
    if solution is None and "answer" in example:
        solution = example["answer"]
    if solution is None:
        raise ValueError(f"Each example must provide one of: `privileged_context`, `{solution_column}`, or `answer`.")

    return {"prompt": prompt, "privileged_context": solution}


def _prepare_split(dataset, solution_column: str):
    return dataset.map(
        _to_opsd_example, fn_kwargs={"solution_column": solution_column}, remove_columns=dataset.column_names
    )


if __name__ == "__main__":
    parser = TrlParser((OPSDScriptArguments, OPSDConfig, ModelConfig))
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

    if script_args.dataset_path is not None:
        dataset = load_from_disk(script_args.dataset_path)
    else:
        dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            streaming=script_args.dataset_streaming,
        )

    if not isinstance(dataset, DatasetDict):
        raise ValueError("OPSD example expects a dataset with named splits.")

    train_dataset = _prepare_split(dataset[script_args.dataset_train_split], script_args.solution_column)
    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split in dataset:
        eval_dataset = _prepare_split(dataset[script_args.dataset_test_split], script_args.solution_column)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False if training_args.gradient_checkpointing else True

    trainer = OPSDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if eval_dataset is not None and script_args.eval_num_prompts:
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        trainer.add_callback(
            LogCompletionsCallback(trainer, generation_config, num_prompts=script_args.eval_num_prompts)
        )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name or script_args.dataset_path)
