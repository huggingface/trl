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
python examples/scripts/rloo/rloo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rloo/rloo_tldr.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_test_split validation \
    --output_dir models/minimal/rloo_tldr \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos
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
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
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
