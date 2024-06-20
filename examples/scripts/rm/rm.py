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

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import RewardConfig, RewardTrainer
from trl.dataset_processor import INPUT_IDS_CHOSEN_KEY, DatasetConfig, PreferenceDatasetProcessor, visualize_token


"""
python examples/scripts/rm/rm.py \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --evaluation_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 128 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=5 \
    --output_dir models/minimal/rm \
"""
CONCAT_CHAT_TEMPLATE = (
    """{% for message in messages %}{{' ' if not loop.first else ''}}{{message['content']}}{% endfor %}{{eos_token}}"""
)


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, DatasetConfig))
    args, dataset_config = parser.parse_args_into_dataclasses()
    # backward compatibility `max_length`
    args.max_length = dataset_config.max_token_length
    base_model = "EleutherAI/pythia-160m"

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = CONCAT_CHAT_TEMPLATE
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("trl-internal-testing/sentiment-trl-style")
    dataset_processor = PreferenceDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    dataset_processor.sanity_check_(raw_datasets)
    train_dataset = dataset_processor.tokenize(raw_datasets["train"])
    eval_dataset = dataset_processor.tokenize(raw_datasets["test"])
    train_dataset = dataset_processor.filter(train_dataset)
    eval_dataset = dataset_processor.filter(eval_dataset)
    visualize_token(train_dataset[0][INPUT_IDS_CHOSEN_KEY], tokenizer)

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()
