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

import pytest
import torch
from datasets import Dataset, load_dataset

from trl.experimental.vimpo import VIMPOConfig, VIMPOTrainer
from trl.experimental.vimpo.vimpo_trainer import _compute_vimpo_gae

from ..testing_utils import TrlTestCase


class TestVIMPOConfig:
    def test_defaults(self):
        args = VIMPOConfig("dummy", use_cpu=True)
        assert args.scale_rewards == "none"
        assert args.vimpo_beta == 5e-4
        assert args.vimpo_actor_coeff == 5e-3
        assert args.vimpo_gae_lambda == 1.0

    def test_invalid_vimpo_beta(self):
        with pytest.raises(ValueError, match="vimpo_beta"):
            VIMPOConfig("dummy", vimpo_beta=0.0, use_cpu=True)


class TestVIMPOTrainer(TrlTestCase):
    def test_vimpo_gae_masks_padding(self):
        token_advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

        advantages = _compute_vimpo_gae(token_advantages, mask, lam=0.5)

        expected = torch.tensor([[2.75, 3.5, 3.0], [6.5, 5.0, 0.0]])
        torch.testing.assert_close(advantages, expected)

    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train").select(range(3))

        training_args = VIMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_steps=1,
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=2,  # keep the exact full-distribution KL path cheap
            vimpo_beta=0.1,
            vimpo_actor_coeff=5e-3,
            report_to="none",
            use_cpu=True,
        )
        trainer = VIMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "vimpo/value_loss" in trainer.state.log_history[-1]
        assert "vimpo/actor_advantage" in trainer.state.log_history[-1]
        assert "kl" in trainer.state.log_history[-1]

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_rejects_inherited_beta(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = VIMPOConfig(output_dir=self.tmp_dir, beta=0.1, report_to="none", use_cpu=True)

        with pytest.raises(ValueError, match="vimpo_beta"):
            VIMPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_rejects_reward_scaling(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = VIMPOConfig(output_dir=self.tmp_dir, scale_rewards="group", report_to="none", use_cpu=True)

        with pytest.raises(ValueError, match="scale_rewards"):
            VIMPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )
