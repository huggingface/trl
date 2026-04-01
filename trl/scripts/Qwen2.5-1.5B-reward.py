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

import argparse
import os

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainerCallback

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


class RewardHeadCheckCallback(TrainerCallback):
    """Validate that the reward head outputs a single scalar per sample."""

    def __init__(self, expected_out_features: int = 1, fail_on_mismatch: bool = True):
        self.expected_out_features = expected_out_features
        self.fail_on_mismatch = fail_on_mismatch

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            logger.warning("RewardHeadCheckCallback: model is None; skip head validation.")
            return

        head_name = None
        head_module = None

        # SequenceClassification models usually expose `score`; this also works for many PEFT-wrapped models.
        if hasattr(model, "score") and isinstance(model.score, torch.nn.Module):
            head_name = "score"
            head_module = model.score
        else:
            for name, module in model.named_modules():
                if name.endswith("score"):
                    head_name = name
                    head_module = module
                    break

        if head_module is None:
            logger.warning(
                "RewardHeadCheckCallback: could not find a classification head named 'score'; "
                "skip out_features check."
            )
            return

        out_features = getattr(head_module, "out_features", None)
        num_labels = getattr(model.config, "num_labels", None)

        logger.info(
            "Reward head check: head=%s, out_features=%s, config.num_labels=%s",
            head_name,
            out_features,
            num_labels,
        )

        if out_features != self.expected_out_features:
            msg = (
                f"Reward head mismatch: expected out_features={self.expected_out_features}, "
                f"got {out_features} (head: {head_name}). "
                "Set num_labels=1 when loading AutoModelForSequenceClassification."
            )
            if self.fail_on_mismatch:
                raise ValueError(msg)
            logger.warning(msg)


def main(script_args, training_args, model_args, dataset_args):
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

    dataset = dataset.select_columns(["chosen","rejected"])
    print(next(iter(dataset["train"])))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        problem_type="regression",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Initialize the RewardTrainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=[RewardHeadCheckCallback(expected_out_features=1, fail_on_mismatch=True)],
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
    dataclass_types = (ScriptArguments, RewardConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "reward", help="Run the reward training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
