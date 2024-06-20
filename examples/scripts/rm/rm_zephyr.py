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

from trl import ModelConfig, RewardConfig, RewardTrainer
from trl.dataset_processor import INPUT_IDS_CHOSEN_KEY, DatasetConfig, PreferenceDatasetProcessor, visualize_token


"""
python -i examples/scripts/minimal/rm_zephyr.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-full \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --logging_steps 1 \
    --eval_strategy steps \
    --max_token_length 1024 \
    --max_prompt_token_lenth 128 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --eval_steps=100 \
    --sanity_check \
    --output_dir models/minimal/rm_zephyr \

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/minimal/rm_zephyr.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-full \
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
    --output_dir models/minimal/rm_zephyr \
"""
ZEPHYR_CHAT_TEMPLATE = """{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"""


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, DatasetConfig, ModelConfig))
    args, dataset_config, model_config = parser.parse_args_into_dataclasses()
    # backward compatibility `max_length`
    args.max_length = dataset_config.max_token_length

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = ZEPHYR_CHAT_TEMPLATE
    model = AutoModelForSequenceClassification.from_pretrained(model_config.model_name_or_path, num_labels=1)
    model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    dataset_processor = PreferenceDatasetProcessor(tokenizer=tokenizer, config=dataset_config)
    dataset_processor.sanity_check_(raw_datasets)
    train_dataset = dataset_processor.tokenize(raw_datasets["train_prefs"])
    eval_dataset = dataset_processor.tokenize(raw_datasets["test_prefs"])
    train_dataset = dataset_processor.filter(train_dataset)
    eval_dataset = dataset_processor.filter(eval_dataset)
    visualize_token(train_dataset[0][INPUT_IDS_CHOSEN_KEY], tokenizer)
    dataset_processor.get_token_length_visualization(train_dataset, save_path="tmp.png")

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
    trainer.evaluate()
