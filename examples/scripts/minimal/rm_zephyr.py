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
import shutil

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import RewardConfig, RewardTrainer
from trl.trainer.model_config import ModelConfig


"""
python -i examples/scripts/minimal/rm_zephyr.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/rm_test \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --num_train_epochs 1 \
    --remove_unused_columns False \
    --max_length 1024 \
    --bf16 \


accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.7.yaml \
    examples/scripts/minimal/rm_zephyr.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/rm_zephyr_new2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta \
    --remove_unused_columns False \
    --max_length 2048 \
    --bf16 \
    --logging_steps 10 \
"""


if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(args.output_dir, ignore_errors=True)
    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        # a default chat template to simply concatenate the messages
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    ################
    # Dataset
    ################
    raw_datasets = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_dataset = raw_datasets["train_prefs"]
    eval_dataset = raw_datasets["test_prefs"]

    def process(row):
        chosen = tokenizer.apply_chat_template(row["chosen"], tokenize=False).strip()
        rejected = tokenizer.apply_chat_template(row["rejected"], tokenize=False).strip()
        row["chosen"] = chosen
        row["rejected"] = rejected
        tokenize_chosen = tokenizer(chosen)
        tokenize_rejected = tokenizer(rejected)
        row["input_ids_chosen"] = tokenize_chosen["input_ids"]
        row["attention_mask_chosen"] = tokenize_chosen["attention_mask"]
        row["input_ids_rejected"] = tokenize_rejected["input_ids"]
        row["attention_mask_rejected"] = tokenize_rejected["attention_mask"]
        return row

    train_dataset = train_dataset.map(process)
    eval_dataset = eval_dataset.map(process)
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length and len(x["input_ids_rejected"]) <= args.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= args.max_length and len(x["input_ids_rejected"]) <= args.max_length
    )
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
    trainer.push_to_hub()
