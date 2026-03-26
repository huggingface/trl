# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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


import unittest

import torch
from datasets import load_dataset
from transformers.utils import is_peft_available

from trl import GRPOConfig
from trl.experimental.gspo_token import GRPOTrainer as GSPOTokenTrainer

from ..testing_utils import TrlTestCase


if is_peft_available():
    pass


class TestDAPOLossAggregation(unittest.TestCase):
    def test_dapo_prompt_level_vs_batch_level_differ_for_unequal_lengths(self):
        """
        Verifies that prompt-level averaging (DAPO paper) produces a different loss
        than batch-level averaging when prompts have unequal completion lengths.

        Prompt-level averaging: average tokens within each prompt, then average across prompts.
        Batch-level averaging (old, incorrect): sum all tokens and divide by total token count.

        With equal-length completions both methods agree; with unequal lengths they diverge
        because batch-level averaging implicitly up-weights longer sequences.
        """
        torch.manual_seed(0)
        batch_size = 3
        max_len = 6

        # Per-token losses: arbitrary positive values
        per_token_loss = torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],  # prompt 0: 3 tokens
                [4.0, 5.0, 0.0, 0.0, 0.0, 0.0],  # prompt 1: 2 tokens
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],  # prompt 2: 6 tokens
            ]
        )
        completion_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.float,
        )

        num_items_in_batch = batch_size  # single process

        # --- New (correct): prompt-level averaging via .mean() ---
        per_prompt_loss = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        loss_prompt_level = per_prompt_loss.mean()

        # --- Old (incorrect): batch-level averaging ---
        loss_batch_level = (per_token_loss * completion_mask).sum() / num_items_in_batch

        # The two values must differ because sequence lengths are unequal
        self.assertNotAlmostEqual(
            loss_prompt_level.item(),
            loss_batch_level.item(),
            places=5,
            msg="Expected prompt-level and batch-level losses to differ for unequal-length completions",
        )

        # Sanity-check the prompt-level value manually:
        # prompt 0: (1+2+3)/3 = 2.0
        # prompt 1: (4+5)/2   = 4.5
        # prompt 2: (6+7+8+9+10+11)/6 = 8.5
        # mean of [2.0, 4.5, 8.5] / 3 == (2.0 + 4.5 + 8.5) / 3
        expected_prompt_level = (2.0 + 4.5 + 8.5) / 3
        self.assertAlmostEqual(loss_prompt_level.item(), expected_prompt_level, places=5)

        # Batch-level: (1+2+3+4+5+6+7+8+9+10+11) / 3 = 66/3 = 22.0
        expected_batch_level = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11) / 3
        self.assertAlmostEqual(loss_batch_level.item(), expected_batch_level, places=5)


class TestGSPOTokenTrainer(TrlTestCase):
    def test_training(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            importance_sampling_level="sequence_token",
            report_to="none",
        )
        trainer = GSPOTokenTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
