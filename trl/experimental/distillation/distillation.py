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

# docstyle-ignore
"""
# Full training (off-policy only, lmbda=0):
```
python trl/experimental/distillation/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lmbda 0.0 \
    --output_dir distilled-model \
    --num_train_epochs 1
```

# Mixed on/off-policy (lmbda=0.5):
```
python trl/experimental/distillation/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lmbda 0.5 \
    --beta 0.5 \
    --output_dir distilled-model \
    --num_train_epochs 1
```

# LoRA:
```
python trl/experimental/distillation/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lmbda 0.0 \
    --output_dir distilled-model \
    --num_train_epochs 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16
```
"""

import argparse
import os


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main(script_args, training_args, model_args):
    from datasets import load_dataset
    from transformers import GenerationConfig

    from trl import (
        LogCompletionsCallback,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )
    from trl.experimental.distillation import DistillationTrainer

    ################
    # Model init kwargs
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=training_args.teacher_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Training
    ################
    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "dev" in dataset:
            eval_dataset = dataset["dev"]

    trainer = DistillationTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    from trl import ModelConfig, ScriptArguments, TrlParser
    from trl.experimental.distillation import DistillationConfig

    dataclass_types = (ScriptArguments, DistillationConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "distillation", help="Run the distillation training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args)
