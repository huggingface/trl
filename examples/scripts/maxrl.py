#!/usr/bin/env python
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
# MaxRL Training Example

This script demonstrates how to train a model using Maximum Likelihood Reinforcement Learning (MaxRL).
MaxRL is similar to GRPO but uses p-normalization (dividing by mean) instead of standard normalization,
which helps prevent bias towards questions with different difficulty levels.

## Usage

```bash
# Basic usage
python examples/scripts/maxrl.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --output_dir maxrl_model \
    --learning_rate 1e-6 \
    --num_generations 8 \
    --max_completion_length 256

# With accelerate for multi-GPU training
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/maxrl.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name trl-lib/DeepMath-103K \
    --output_dir maxrl_model \
    --learning_rate 1e-6
```

For more details on the algorithm, see the paper:
[Maximum Likelihood Reinforcement Learning](https://huggingface.co/papers/2602.02710)
"""

from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import HfArgumentParser

from trl import MaxRLConfig, MaxRLTrainer
from trl.rewards import accuracy_reward


@dataclass
class ScriptArguments:
    """
    Arguments specific to this script.
    """

    dataset_name: str = field(
        default="trl-lib/DeepMath-103K",
        metadata={"help": "The dataset name on Hugging Face Hub"},
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "The dataset split to use for training"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, MaxRLConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    # The model and processing_class are loaded from training_args.model_name_or_path inside the trainer

    ################
    # Dataset
    ################
    raw_dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_train_split)

    # In this example, we are using the accuracy_reward function which verifies math solutions.
    # The dataset should contain a "prompt" column with the problem and a "solution" column with the expected answer.
    # The model generates completions which are checked against the solution.

    ################
    # Training
    ################
    trainer = MaxRLTrainer(
        model=training_args.model_name_or_path,
        reward_funcs=accuracy_reward,
        args=training_args,
        train_dataset=raw_dataset,
    )

    trainer.train()

    # Save the final model
    trainer.save_model(training_args.output_dir)
