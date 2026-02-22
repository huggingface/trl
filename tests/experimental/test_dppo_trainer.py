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
from datasets import load_dataset

from trl.experimental.dppo import DPPOConfig, DPPOTrainer

from ..testing_utils import TrlTestCase


class TestDPPODivergenceMask:
    """Unit tests for _compute_divergence_mask with synthetic inputs."""

    def _make_trainer(self, divergence_type="binary_tv", epsilon=0.2, epsilon_high=0.28):
        """Create a minimal DPPOTrainer-like object with just the attributes needed for _compute_divergence_mask."""

        class Stub:
            pass

        stub = Stub()
        stub.divergence_type = divergence_type
        stub.epsilon_low = epsilon
        stub.epsilon_high = epsilon_high
        return stub

    def test_binary_tv_no_masking_within_threshold(self):
        stub = self._make_trainer("binary_tv", epsilon=0.2, epsilon_high=0.28)
        # Policies are very close — no tokens should be masked
        sampling_logps = torch.log(torch.tensor([[0.5, 0.3, 0.7]]))
        current_logps = torch.log(torch.tensor([[0.51, 0.29, 0.71]]))
        advantages = torch.tensor([[1.0]])
        completion_mask = torch.ones(1, 3)

        mask = DPPOTrainer._compute_divergence_mask(stub, current_logps, sampling_logps, advantages, completion_mask)
        assert mask.shape == (1, 3)
        assert (mask == 1.0).all()

    def test_binary_tv_masks_positive_advantage_high_divergence(self):
        stub = self._make_trainer("binary_tv", epsilon=0.01, epsilon_high=0.01)
        # π much higher than μ, positive advantage → should be masked (invalid_pos)
        sampling_logps = torch.log(torch.tensor([[0.1]]))
        current_logps = torch.log(torch.tensor([[0.5]]))
        advantages = torch.tensor([[1.0]])
        completion_mask = torch.ones(1, 1)

        mask = DPPOTrainer._compute_divergence_mask(stub, current_logps, sampling_logps, advantages, completion_mask)
        assert mask.item() == 0.0

    def test_binary_tv_masks_negative_advantage_low_divergence(self):
        stub = self._make_trainer("binary_tv", epsilon=0.01, epsilon_high=0.01)
        # π much lower than μ, negative advantage → should be masked (invalid_neg)
        sampling_logps = torch.log(torch.tensor([[0.5]]))
        current_logps = torch.log(torch.tensor([[0.1]]))
        advantages = torch.tensor([[-1.0]])
        completion_mask = torch.ones(1, 1)

        mask = DPPOTrainer._compute_divergence_mask(stub, current_logps, sampling_logps, advantages, completion_mask)
        assert mask.item() == 0.0

    def test_binary_tv_respects_completion_mask(self):
        stub = self._make_trainer("binary_tv", epsilon=0.01, epsilon_high=0.01)
        # Even though divergence is huge, padding tokens stay 0
        sampling_logps = torch.log(torch.tensor([[0.1, 0.5]]))
        current_logps = torch.log(torch.tensor([[0.9, 0.9]]))
        advantages = torch.tensor([[1.0]])
        completion_mask = torch.tensor([[1.0, 0.0]])

        mask = DPPOTrainer._compute_divergence_mask(stub, current_logps, sampling_logps, advantages, completion_mask)
        assert mask[0, 1].item() == 0.0

    def test_topk_tv_requires_topk_inputs(self):
        stub = self._make_trainer("topk_tv")
        B, T, K = 1, 2, 4
        sampling_logps = torch.log(torch.full((B, T), 0.3))
        current_logps = torch.log(torch.full((B, T), 0.31))
        advantages = torch.tensor([[1.0]])
        completion_mask = torch.ones(B, T)

        # Build top-K distributions that are nearly identical
        topk_probs = torch.softmax(torch.randn(B, T, K), dim=-1)
        sampling_topk_logps = torch.log(topk_probs)
        current_topk_logps = torch.log(topk_probs + 0.001)

        mask = DPPOTrainer._compute_divergence_mask(
            stub,
            current_logps,
            sampling_logps,
            advantages,
            completion_mask,
            current_topk_logps=current_topk_logps,
            sampling_topk_logps=sampling_topk_logps,
        )
        assert mask.shape == (B, T)
        assert (mask == 1.0).all()


@pytest.mark.low_priority
class TestDPPOTrainer(TrlTestCase):
    @pytest.mark.parametrize("divergence_type", ["binary_tv", "binary_kl"])
    def test_training_binary(self, divergence_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            divergence_type=divergence_type,
            report_to="none",
        )
        trainer = DPPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_training_with_custom_reward_func(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def dummy_reward(completions, **kwargs):
            return [float(len(c)) for c in completions]

        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = DPPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=dummy_reward,
            args=config,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_training_conversational(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        config = DPPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = DPPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=config,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
