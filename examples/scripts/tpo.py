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
#     "trl[peft]",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Run the TPO training script with the following command with some example arguments.
TPO requires a *triple-preference* dataset where each example contains a `chosen`, a `rejected` and a `reference`
(gold) completion for the same prompt.

# Full training:
python examples/scripts/tpo.py \
    --dataset_name tpo-alignment/ultrafeedback_triple_preference \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --eval_strategy steps \
    --eval_steps 500 \
    --beta 0.01 \
    --tpo_alpha 1.0 \
    --output_dir Qwen2-0.5B-TPO \
    --no_remove_unused_columns

# TPO-L (length-normalized variant with target reward margin):
python examples/scripts/tpo.py \
    --dataset_name tpo-alignment/ultrafeedback_triple_preference \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 8 \
    --beta 0.01 \
    --tpo_alpha 1.0 \
    --loss_type tpo-l \
    --tpo_l_gamma 5.4 \
    --output_dir Qwen2-0.5B-TPO-L \
    --no_remove_unused_columns

# LoRA:
python examples/scripts/tpo.py \
    --dataset_name tpo-alignment/ultrafeedback_triple_preference \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --output_dir Qwen2-0.5B-TPO-LoRA \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, ScriptArguments, get_peft_config
from trl.experimental.tpo import TPOConfig, TPOTrainer


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TPOConfig, ModelConfig))
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

    ################
    # Training
    ################
    trainer = TPOTrainer(
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
