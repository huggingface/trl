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
# ]
# ///

"""Training script for [`SRTTrainer`].

Trains a model with the self-revision objective via [`SRTTrainer`]. The dataset must be saved locally with
`datasets.save_to_disk` and contain columns `problem`, `y_init`, `control_prompt`, and `y_revised`.

Example:

```bash
uv run python trl/experimental/sdzero/srt.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_path /tmp/sdzero_gsm8k_srt \
    --output_dir outputs/sdzero-srt-qwen2.5-0.5b \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 --max_steps 50 --logging_steps 1
```
"""

from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.sdzero import SRTConfig, SRTTrainer


@dataclass
class SRTScriptArguments(ScriptArguments):
    dataset_path: str | None = field(
        default=None,
        metadata={"help": "Local path to a self-revision dataset saved with `datasets.save_to_disk`."},
    )


if __name__ == "__main__":
    parser = TrlParser((SRTScriptArguments, SRTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.dataset_path is None:
        raise ValueError("`--dataset_path` is required (pointing to a self-revision dataset).")

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
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(script_args.dataset_path)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    trainer = SRTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
