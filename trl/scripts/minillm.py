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

import argparse
import os
from dataclasses import dataclass, field

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback, TrainingArguments


from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)
from trl.experimental.minillm.minillm_trainer import MiniLLMTrainer, MiniLLMConfig


logger = logging.get_logger(__name__)



class SaveStep0CallBack(TrainerCallback):
    def __init__(self, trainer: MiniLLMTrainer, trial=None):
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



@dataclass
class TeacherArguments:
    teacher_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to the teacher model for generating demonstrations."}
    )
    teacher_model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use from the model repository."}
    )
    teacher_trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code when loading the teacher model."},
    )
    teacher_attn_implementation: str = field(
        default=None,
        metadata={"help": "The attention implementation to use in the teacher model."},
    )
    teacher_dtype: str = field(
        default=None,
        metadata={"help": "The data type to use in the teacher model."},
    )
    

def load_model(model_name_or_path, model_revision, trust_remote_code, attn_implementation, dtype):
    model_kwargs = dict(
        revision=model_revision,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    return model


def main(
        teacher_args: TeacherArguments,
        script_args: ScriptArguments, 
        training_args: MiniLLMConfig,
        model_args: ModelConfig, 
        dataset_args: DatasetMixtureConfig
    ):
    # Get the reward models and functions

    if os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS") == "auto":
        os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(training_args.gradient_accumulation_steps)

    model = load_model(
        model_args.model_name_or_path,
        model_args.model_revision,
        model_args.trust_remote_code,
        model_args.attn_implementation,
        model_args.dtype,
    )

    teacher_model = load_model(
        teacher_args.teacher_model_name_or_path,
        teacher_args.teacher_model_revision,
        teacher_args.teacher_trust_remote_code,
        teacher_args.teacher_attn_implementation,
        teacher_args.teacher_dtype,
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

    # Initialize the MiniLLM trainer
    trainer = MiniLLMTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
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


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (TeacherArguments, ScriptArguments, MiniLLMConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("minillm", help="Run the MiniLLM training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    teacher_args, script_args, training_args, model_args, dataset_args, additional_args = \
        parser.parse_args_and_config(return_remaining_strings=True)
    print(additional_args)
    main(teacher_args, script_args, training_args, model_args, dataset_args)