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

import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import RLOOConfig, RLOOTrainer


class RLOOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        self.policy_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.policy_ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def test_rloo_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                total_episodes=1,
                report_to="none",
            )

            dummy_text = [{"content": "Hello World!", "role": "user"}]
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
        sequence_length = 5  # Add sequence length for testing token-level rewards

        # fmt: off
        rlhf_reward = torch.tensor([
            1, 2, 3, # first rlhf reward for three prompts
            2, 3, 4, # second rlhf reward for three prompts
            5, 6, 7, # third rlhf reward for three prompts
            8, 9, 10, # fourth rlhf reward for three prompts
        ]).float()

        # Create padding mask where 1 indicates valid token, 0 indicates padding
        padding_mask = torch.ones(local_batch_size * rloo_k, sequence_length)
        # Set padding based on sequence lengths
        sequence_lengths = torch.tensor([
            3, 4, 3,  # lengths for first batch
            4, 3, 4,  # lengths for second batch
            3, 4, 3,  # lengths for third batch
            4, 3, 4,  # lengths for fourth batch
        ])
        for i, length in enumerate(sequence_lengths):
            padding_mask[i, length:] = 0

        # Add kl tensor for testing token-level rewards
        kl = torch.ones(local_batch_size * rloo_k, sequence_length)  # Dummy KL values
        # fmt: on

        # Test token-level KL rewards following OpenRLHF implementation
        kl_coef = 0.1
        kl_reward = -kl_coef * kl

        # Find last non-padded position
        eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)

        # Create last reward tensor
        last_reward = torch.zeros_like(kl)
        last_reward.scatter_(dim=1, index=eos_indices, src=rlhf_reward.reshape(-1, 1))

        # Test last_reward - should have rlhf_reward at the last non-padded position
        for i, (length, reward) in enumerate(zip(sequence_lengths, rlhf_reward)):
            # Check reward is at correct position
            self.assertEqual(last_reward[i, length - 1].item(), reward.item())
            # Check zeros elsewhere
            self.assertTrue(torch.all(last_reward[i, : length - 1] == 0))
            self.assertTrue(torch.all(last_reward[i, length:] == 0))

        # Combine rewards
        reward = last_reward + kl_reward
        non_score_reward = kl_reward.sum(1)
        token_level_rlhf_reward = reward.sum(1)

        # Test reward components
        # KL reward should be -0.1 for each token in sequence length
        expected_kl_reward = -0.1 * sequence_length  # Each position gets -0.1 KL reward
        torch.testing.assert_close(non_score_reward, torch.tensor(expected_kl_reward).expand_as(non_score_reward))

        # Total reward should be rlhf_reward + kl_reward
        expected_total = rlhf_reward + expected_kl_reward
        torch.testing.assert_close(token_level_rlhf_reward, expected_total)

        # Test sequence-level rewards (existing test)
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

        # Test vectorized implementation
        rlhf_reward = rlhf_reward.reshape(rloo_k, local_batch_size)
        baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
        vec_advantages = rlhf_reward - baseline
        torch.testing.assert_close(vec_advantages.flatten(), advantages)

    def test_rloo_training(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                total_episodes=1,
                num_train_epochs=1,
                max_steps=2,
                report_to="none",
            )

            # Create a simple dataset
            dummy_text = [{"content": "Hello World!", "role": "user"}]
            dummy_data = self.tokenizer.apply_chat_template(dummy_text)
            dummy_dataset = Dataset.from_dict({"input_ids": [dummy_data, dummy_data]})

            trainer = RLOOTrainer(
                config=training_args,
                policy=self.policy_model,
                reward_model=self.reward_model,
                ref_policy=self.policy_ref_model,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            # Test that training completes without errors
            trainer.train()

            # Check if objective/rlhf_reward is available
            self.assertIn("objective/rlhf_reward", trainer.state.log_history[-1])

    def test_rloo_training_with_custom_reward(self):
        # dummy reward function
        def reward_function(texts):
            # based on length of text
            rewards = [len(text) for text in texts]
            return rewards

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = RLOOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                total_episodes=1,
                num_train_epochs=1,
                max_steps=2,
                report_to="none",
            )

            # Create a simple dataset
            dummy_text = [{"content": "Hello World!", "role": "user"}]
            dummy_data = self.tokenizer.apply_chat_template(dummy_text)
            dummy_dataset = Dataset.from_dict({"input_ids": [dummy_data, dummy_data]})

            trainer = RLOOTrainer(
                config=training_args,
                policy=self.policy_model,
                reward_model=reward_function,
                ref_policy=self.policy_ref_model,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            # Test that training completes without errors
            trainer.train()

            # Check if objective/rlhf_reward is available
            self.assertIn("objective/rlhf_reward", trainer.state.log_history[-1])
