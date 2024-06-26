# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer
from trl.dataset_processor import (
    CHAT_TEMPLATES,
    INPUT_IDS_CHOSEN_KEY,
    DatasetConfig,
    PreferenceDatasetProcessor,
    visualize_token,
)
from trl.trainer.utils import layer_init


"""
# LEVEL 0: interactive debugging
python -i examples/scripts/rm/rm.py \
    --dataset_name trl-internal-testing/sentiment-trl-style \
    --dataset_train_split train \
    --dataset_eval_split test \
    --model_name_or_path EleutherAI/pythia-160m \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --output_dir models/rm/rm \
    --sanity_check \

# LEVEL 1: single GPU model training; adjust your `per_device_train_batch_size` and
# `gradient_accumulation_steps` accordingly
# you can also use the `trl-internal-testing/descriptiveness-trl-style` dataset
python examples/scripts/rm/rm.py \
    --dataset_name trl-internal-testing/sentiment-trl-style \
    --dataset_train_split train \
    --dataset_eval_split test \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --output_dir models/rm/rm_sentiment_1b \
    --push_to_hub \

# LEVEL 2: multi-gpu training using DS2 with the TL;DR summarization dataset
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rm/rm.py \
    --dataset_name trl-internal-testing/tldr-preference-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --model_name_or_path EleutherAI/pythia-2.8b-deduped \
    --chat_template simple_concat_with_space \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 512 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --output_dir models/rm/rm_tldr_2.8b \
    --bf16 \

# LEVEL 2: multi-gpu training using DS2 with the anthropic HH dataset
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rm/rm.py \
    --dataset_name trl-internal-testing/hh-rlhf-trl-style \
    --dataset_train_split train \
    --dataset_eval_split validation \
    --model_name_or_path EleutherAI/pythia-2.8b-deduped \
    --chat_template simple_chat \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 2048 \
    --max_prompt_token_lenth 1024 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --bf16 \
    --output_dir models/rm/rm_hh_2.8b \

# LEVEL 3: multi-gpu training using DS2 with the ultrafeedback dataset
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rm/rm_zephyr.py \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split train_prefs \
    --dataset_eval_split test_prefs \
    --chat_template zephyr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 1024 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --bf16 \
    --output_dir models/rm/rm_zephyr_7b \
"""


if __name__ == "__main__":
    parser = HfArgumentParser((DatasetConfig, ModelConfig, RewardConfig))
    dataset_config, model_config, args = parser.parse_args_into_dataclasses()
    # backward compatibility `max_length`
    args.max_length = dataset_config.max_token_length

    ################
    # Tokenizer & Dataset
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]

    ################
    # Dataset
    ################
    dataset = load_dataset(dataset_config.dataset_name)
    dataset_processor = PreferenceDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    dataset_processor.sanity_check_(dataset)
    dataset = dataset_processor.tokenize(dataset)
    dataset = dataset_processor.filter(dataset)
    dataset_processor.get_token_length_visualization(dataset, save_path="tmp.png")
    train_dataset = dataset[dataset_config.dataset_train_split]
    eval_dataset = dataset[dataset_config.dataset_eval_split]
    visualize_token(train_dataset[0][INPUT_IDS_CHOSEN_KEY], tokenizer)

    ################
    # Model & Training
    ################
    model = AutoModelForSequenceClassification.from_pretrained(model_config.model_name_or_path, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.score = layer_init(
        nn.Linear(model.config.hidden_size, 1),
        std=1 / np.sqrt(model.config.hidden_size + 1),
    )
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path
    )  # reset tokenizer (without pad token)
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()
    trainer.evaluate()
    if "wandb" in args.report_to:
        import wandb

        if wandb.run is not None:
            for item in [dataset_config, model_config]:
                wandb.config.update(item, allow_val_change=True)
