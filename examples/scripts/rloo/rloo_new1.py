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
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
# ]
# ///

import argparse
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl.trainer.rloo_final_2 import RLOOConfig_NEW, RLOOTrainer_NEW
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

"""
python -i examples/scripts/rloo/rloo_new1.py \
    --output_dir rloo_new1 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --reward_model_name_or_path EleutherAI/pythia-1b-deduped \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_generations 2 \
    --per_device_train_batch_size 64 \
    --importance_sampling_level sequence \
    --max_steps 100
"""
"""
python -i examples/scripts/rloo/rloo_new1.py \
    --output_dir rloo_new1 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --reward_model_name_or_path EleutherAI/pythia-1b-deduped \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_generations 2 \
    --per_device_train_batch_size 64 \
    --importance_sampling_level sequence \
    --max_steps 100 
"""


@dataclass
class RLOOScriptArguments(ScriptArguments):
    """
    Script arguments for the RLOO training script.

    Args:
        reward_model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
    """

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )


def main(script_args, training_args, model_args):
    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
        )
        reward_funcs.append(reward_model)


    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_config)

    # Initialize the new RLOO trainer
    trainer = RLOOTrainer_NEW(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (RLOOScriptArguments, RLOOConfig_NEW, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("rloo", help="Run the RLOO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
