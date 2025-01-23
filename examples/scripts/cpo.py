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

"""
Run the CPO training script with the following command with some example arguments.
In general, the optimal configuration for CPO will be similar to that of DPO:

# regular:
python examples/scripts/cpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-cpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/cpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-lora-aligned-cpo" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import CPOConfig, CPOTrainer, ModelConfig, ScriptArguments, get_peft_config
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, CPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Training
    ################
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # train and save the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
