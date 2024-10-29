# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Usage for a 4 GPU setup:

accelerate launch --num_processes 3 examples/scripts/dpo_online_async.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0
"""

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import (
    AsyncOnlineDPOConfig,
    AsyncOnlineDPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)
from trl.trainer.sync_online_dpo_trainer import SyncOnlineDPOTrainer


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, AsyncOnlineDPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    script_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # make sure using same base model
    training_args.sft_model_path = model_config.model_name_or_path

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    ################
    # Dataset
    ################
    train_dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_train_split).select(range(1000))
    eval_dataset = load_dataset(script_args.dataset_name, split=script_args.dataset_test_split)

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element["prompt"],
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

    if training_args.sync_fallback is True:
        TrainerCls = SyncOnlineDPOTrainer
    else:
        TrainerCls = AsyncOnlineDPOTrainer

    trainer = TrainerCls(
        config=training_args,
        processing_class=tokenizer,
        policy=model,
        ref_policy=ref_model,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    # generation_config = GenerationConfig(
    #     max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
    # )
    # completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
    # trainer.add_callback(completions_callback)
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
