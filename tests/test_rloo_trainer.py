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
import platform
import subprocess
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import RLOOConfig, RLOOTrainer


def test():
    command = """\
python examples/scripts/rloo/rloo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/rloo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10 \
    --model_name_or_path EleutherAI/pythia-14m \
    --sft_model_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --missing_eos_penalty 1.0 \
    --save_strategy no \
    --stop_token eos
"""
    if platform.system() == "Windows":
        # windows CI does not work with subprocesses for some reason
        # e.g., https://github.com/huggingface/trl/actions/runs/9600036224/job/26475286210?pr=1743
        return
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


class RLOOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.sft_model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.reward_model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"

        self.policy_model = AutoModelForCausalLM.from_pretrained(self.sft_model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_id)
        self.policy_ref_model = AutoModelForCausalLM.from_pretrained(self.sft_model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_id, padding_side="left")
        self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def test_rloo_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                total_episodes=1,
                report_to="none",
            )

            dummy_text = {"content": "Hello World!", "role": "user"}
            dummy_data = self.tokenizer.apply_chat_template(dummy_text)
            dummy_dataset = Dataset.from_dict({"input_ids": dummy_data})

            trainer = RLOOTrainer(
                config=training_args,
                policy=self.policy_model,
                reward_model=self.reward_model,
                ref_policy=self.policy_ref_model,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            trainer._save_checkpoint(trainer.model, trial=None)

    def test_rloo_reward(self):
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
        self.assertLess((1 - (2 + 5 + 8) / 3 - advantages[0].item()), 1e-6)
        self.assertLess((6 - (3 + 2 + 9) / 3 - advantages[7].item()), 1e-6)

        # vectorized impl
        rlhf_reward = rlhf_reward.reshape(rloo_k, local_batch_size)
        baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
        vec_advantages = rlhf_reward - baseline
        torch.testing.assert_close(vec_advantages.flatten(), advantages)
