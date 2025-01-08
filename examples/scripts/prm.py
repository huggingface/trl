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
Full training:
python examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/prm800k \
    --output_dir Qwen2-0.5B-Reward \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-5 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50

LoRA:
python examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/prm800k \
    --output_dir Qwen2-0.5B-Reward-LoRA \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --learning_rate 1.0e-4 \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import warnings

import torch
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, HfArgumentParser

from trl import (
    ModelConfig,
    PRMConfig,
    PRMTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PRMConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=2, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_config.use_peft and model_config.lora_task_type != "TOKEN_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `TOKEN_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type TOKEN_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    dataset = dataset.filter(lambda x: len(x["completions"]) > 0)

    ##########
    # Training
    ##########
    trainer = PRMTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        peft_config=get_peft_config(model_config),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
