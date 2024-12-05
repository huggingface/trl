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

python examples/scripts/xpo.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-tldr-xpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 3 \
    --max_new_tokens 64 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --push_to_hub
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    XPOConfig,
    XPOTrainer,
    get_kbit_device_map,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, XPOConfig, ModelConfig))
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
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    if training_args.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            num_labels=1,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        reward_model = None

    if training_args.judge is not None:
        judge_cls = JUDGES[training_args.judge]
        judge = judge_cls()
    else:
        judge = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    dataset = load_dataset(script_args.dataset_name)

    trainer = XPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
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
