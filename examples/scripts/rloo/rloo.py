# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


"""
python -i examples/scripts/rloo/rloo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/rloo/rloo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/rloo \
    --rloo_k 2 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )
    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
