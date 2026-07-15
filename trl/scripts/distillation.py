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
Run the on-policy distillation training script with the commands below.

# Full training:
```bash
python trl/scripts/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --output_dir Qwen2.5-0.5B-Distill
```

# LoRA:
```bash
python trl/scripts/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --output_dir Qwen2.5-0.5B-Distill-LoRA \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""

import argparse
from dataclasses import dataclass, field

from trl import ScriptArguments


@dataclass
class DistillationScriptArguments(ScriptArguments):
    """
    Script arguments for the distillation training script.

    Args:
        teacher_model_name_or_path (`str`, *optional*):
            Model checkpoint for the teacher model.
    """

    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Model checkpoint for the teacher model."},
    )


def main(script_args, training_args, model_args, dataset_args):
    from accelerate.logging import get_logger
    from datasets import load_dataset

    from trl import DistillationTrainer, get_dataset, get_peft_config, get_quantization_config

    logger = get_logger(__name__)

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    training_args.teacher_model_init_kwargs = dict(
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the distillation trainer
    trainer = DistillationTrainer(
        model=model_args.model_name_or_path,
        teacher_model=script_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        quantization_config=get_quantization_config(model_args),
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    from trl import DatasetMixtureConfig, DistillationConfig, ModelConfig, TrlParser

    dataclass_types = (DistillationScriptArguments, DistillationConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "distillation", help="Run the distillation training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args, dataset_args)
