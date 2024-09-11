# flake8: noqa
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
"""
# Full training:
python examples/scripts/gkd.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name andito/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gkd-model \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing

# LoRA:
python examples/scripts/gkd.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --dataset_name andito/chatbot_arena_completions \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --output_dir gkd-model \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 16
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    GKDConfig,
    GKDTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import SFTScriptArguments, TrlParser
from trl.trainer.callbacks import LogCompletionsCallback


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, GKDConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    train_dataset = raw_datasets[args.dataset_train_split]
    try:
        eval_dataset = raw_datasets[args.dataset_test_split]
        prompts = eval_dataset["messages"][:8]
    except KeyError:
        eval_dataset = None
        prompts = train_dataset["messages"][:8]

    # remove the last assistant message from the prompts messages and then apply chat template to the prompts
    prompts = [prompts[i][:-1] for i in range(len(prompts))]
    prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]

    ################
    # Training
    ################
    trainer = GKDTrainer(
        model=model_config.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )
    log_completions_callback = LogCompletionsCallback(prompts)
    trainer.add_callback(log_completions_callback)
    trainer.train()

    trainer.save_model(training_args.output_dir)
