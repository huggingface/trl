# flake8: noqa
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
Usage:

python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --push_to_hub
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOScriptArguments,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    get_kbit_device_map,
    get_quantization_config,
)
from trl.commands.cli_utils import TrlParser
from trl.trainer.callbacks import LogCompletionsCallback
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, OnlineDPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    args.gradient_checkpointing_kwargs = {"use_reentrant": True}

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
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # prompt = 'What are some synonyms for the word "beautiful"?'
    # response = "Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant"
    # messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    # input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    # output = reward_model(input_ids)

    dataset = load_dataset(args.dataset_name)

    def prepare_dataset(row):
        prompt = tokenizer.apply_chat_template([row["messages"]], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt}

    with PartialState().local_main_process_first():
        dataset = dataset.map(prepare_dataset, num_proc=training_args.dataset_num_proc)

    prompts = dataset[args.dataset_test_split]["prompt"][:8]

    trainer = OnlineDPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
    )
    log_completions_callback = LogCompletionsCallback(prompts)
    trainer.add_callback(log_completions_callback)
    trainer.train()
