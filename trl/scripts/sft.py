# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse
import os
from dataclasses import dataclass

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from trl.wandb_utils import setup_wandb


logger = logging.get_logger(__name__)


class SaveStep0CallBack(TrainerCallback):
    def __init__(self, trainer: SFTTrainer, trial=None):
        self.trainer = trainer
        self.trial = trial

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer._save_checkpoint(self.trainer.model_wrapped, self.trial)


class WandbLoggingCallback(TrainerCallback):
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        wandb_logs = {f"train/{key}": value for key, value in logs.items()}
        if self.wandb_logger is not None:
            self.wandb_logger.log(wandb_logs, step=state.global_step, commit=True)


@dataclass
class WandbArguments:
    wandb_entity: str = "eLLM-han2024"
    wandb_project: str = "minillm-trl"
    wandb_run_name: str = None
    wandb_mode: str = "disabled"
    wandb_job_type: str = "sft"
    wandb_group: str = None


def main(script_args, training_args: TrainingArguments, model_args, dataset_args, wandb_args: WandbArguments):
    ################
    # Model init kwargs
    ################

    if os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS") == "auto":
        os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(training_args.gradient_accumulation_steps)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

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

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.add_callback(SaveStep0CallBack(trainer))
    # Train the model
    try:
        resume_from_checkpoint = eval(training_args.resume_from_checkpoint)
    except Exception:
        pass
    if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)

    if resume_from_checkpoint is not None:
        trainer.accelerator.print(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    wandb_config = {
        "entity": wandb_args.wandb_entity,
        "project": wandb_args.wandb_project,
        "name": wandb_args.wandb_run_name.strip("/") if wandb_args.wandb_run_name else None,
        "mode": wandb_args.wandb_mode,
        "job_type": wandb_args.wandb_job_type,
        "group": wandb_args.wandb_group.strip("/") if wandb_args.wandb_group else None,
    }
    if trainer.accelerator.is_main_process:
        wandb_logger = setup_wandb(wandb_config, wandb_dir=os.path.join(training_args.output_dir, "wandb"), resume=(resume_from_checkpoint is not None))
    else:
        wandb_logger = None
    trainer.add_callback(WandbLoggingCallback(wandb_logger))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    if wandb_logger is not None:
        wandb_logger.finish()

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig, WandbArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, wandb_args, additional_args = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    print(additional_args)
    main(script_args, training_args, model_args, dataset_args, wandb_args)