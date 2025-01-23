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
Usage:

python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0

With LoRA:
python examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-6 \
    --output_dir pythia-1b-tldr-online-dpo \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --use_peft
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    if training_args.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            num_labels=1,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
            truncation=True,
            truncation_side="left",  # since we judge the completion, truncating left is more appropriate
        )
    else:
        reward_model = None
        reward_tokenizer = None

    if training_args.judge is not None:
        judge_cls = JUDGES[training_args.judge]
        judge = judge_cls()
    else:
        judge = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        reward_processing_class=reward_tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
