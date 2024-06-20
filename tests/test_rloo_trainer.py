# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


def test():
    #     command = """\
    # python examples/scripts/rloo/rloo.py \
    #     --learning_rate 3e-6 \
    #     --output_dir models/minimal/rloo \
    #     --per_device_train_batch_size 4 \
    #     --gradient_accumulation_steps 1 \
    #     --total_episodes 10 \
    #     --model_name_or_path EleutherAI/pythia-14m \
    #     --non_eos_penalty \
    #     --stop_token eos \
    # """
    # subprocess.run(
    #     command,
    #     shell=True,
    #     check=True,
    # )
    config, model_config = RLOOConfig(), ModelConfig()
    config.learning_rate = 3e-6
    config.output_dir = "models/minimal/rloo"
    config.per_device_train_batch_size = 4
    config.gradient_accumulation_steps = 1
    config.total_episodes = 10
    config.non_eos_penalty = True
    config.stop_token = "eos"

    model_config.model_name_or_path = "EleutherAI/pythia-14m"

    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")
    eval_samples = 20
    train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
    eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
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
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=4,  # multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )

    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()


def test_rloo_reward():
    local_batch_size = 3
    rloo_k = 4
    # fmt: off
    rlhf_reward = torch.tensor([
        1, 2, 3, # first rlhf reward for three prompts
        2, 3, 4, # second rlhf reward for three prompts
        5, 6, 7, # third rlhf reward for three prompts
        8, 9, 10, # fourth rlhf reward for three prompts
    ]).float()
    # fmt: on

    baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
    advantages = torch.zeros_like(rlhf_reward)
    for i in range(0, len(advantages), local_batch_size):
        other_response_rlhf_rewards = []
        for j in range(0, len(advantages), local_batch_size):
            if i != j:
                other_response_rlhf_rewards.append(rlhf_reward[j : j + local_batch_size])
        advantages[i : i + local_batch_size] = rlhf_reward[i : i + local_batch_size] - torch.stack(
            other_response_rlhf_rewards
        ).mean(0)
    assert (1 - (2 + 5 + 8) / 3 - advantages[0].item()) < 1e-6
    assert (6 - (3 + 2 + 9) / 3 - advantages[7].item()) < 1e-6

    # vectorized impl
    rlhf_reward = rlhf_reward.reshape(rloo_k, local_batch_size)
    baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
    vec_advantages = rlhf_reward - baseline
    torch.testing.assert_close(vec_advantages.flatten(), advantages)
