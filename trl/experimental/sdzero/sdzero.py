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
#     "math-verify>=0.5.2",
# ]
# ///

"""Training script for [`SDZeroTrainer`] — SD-Zero Phase 2 (On-Policy Self-Distillation).

Trains a model with on-policy self-distillation via revision feedback. The model (typically a Phase 1 SRT
checkpoint) acts as both student and frozen teacher. The student generates responses on-policy, a binary verifier
determines correctness, and the student is trained to match the teacher's revision distribution.

The dataset must expose a `problem` column (the question) and an `answer` column (the gold final answer). A HF
hub dataset or a local disk dataset saved with `datasets.save_to_disk` is accepted.

Example:

```bash
python trl/experimental/sdzero/sdzero.py \\
    --model_name_or_path path/to/srt-checkpoint \\
    --dataset_name open-r1/OpenR1-Math-220k \\
    --output_dir outputs/sdzero-qwen2.5-0.5b \\
    --per_device_train_batch_size 1 \\
    --max_completion_length 256 \\
    --max_steps 100 \\
    --logging_steps 1
```
"""

from dataclasses import dataclass, field

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.sdzero import SDZeroConfig, SDZeroTrainer


@dataclass
class SDZeroScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Local path to a dataset saved with `datasets.save_to_disk`. Overrides `dataset_name`."},
    )
    problem_column: str = field(
        default="problem",
        metadata={"help": "Column name containing the problem / question text."},
    )
    answer_column: str = field(
        default="answer",
        metadata={"help": "Column name containing the gold final answer."},
    )


def _prepare_dataset(dataset, problem_column: str, answer_column: str):
    """Convert dataset rows to the `{"prompt": [...], "answer": ...}` format expected by SDZeroTrainer."""
    missing = [c for c in [problem_column, answer_column] if c not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Present: {dataset.column_names}")

    def _to_prompt_answer(example):
        return {
            "prompt": [{"role": "user", "content": example[problem_column]}],
            "answer": example[answer_column],
        }

    return dataset.map(_to_prompt_answer, remove_columns=dataset.column_names)


if __name__ == "__main__":
    parser = TrlParser((SDZeroScriptArguments, SDZeroConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    dtype = model_args.dtype if model_args.dtype in ("auto", None) else getattr(torch, model_args.dtype)
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.dataset_path is not None:
        raw_dataset = load_from_disk(script_args.dataset_path)
    else:
        raw_dataset = load_dataset(
            script_args.dataset_name,
            name=script_args.dataset_config,
            split=script_args.dataset_train_split,
        )

    dataset = _prepare_dataset(raw_dataset, script_args.problem_column, script_args.answer_column)

    training_args.model_init_kwargs = model_kwargs
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    trainer = SDZeroTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
