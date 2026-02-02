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

import torch
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

from ..testing_utils import TrlTestCase


class TestSDPOConfig(TrlTestCase):
    def test_defaults(self):
        """Test that SDPOConfig has correct default values."""
        config = SDPOConfig(output_dir=self.tmp_dir, report_to="none")

        assert config.distillation_alpha == 1.0
        assert config.distillation_topk == 20
        assert config.full_logit_distillation is False
        assert config.distillation_is_clip == 2.0
        assert config.distillation_add_tail is False
        assert config.dont_reprompt_on_self_success is True
        assert config.ema_update_rate == 0.01
        assert config.max_reprompt_len == 10240
        assert config.distillation_weight == 1.0
        assert config.use_successful_as_teacher is True

    def test_custom_values(self):
        """Test that SDPOConfig accepts custom values."""
        config = SDPOConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            distillation_alpha=0.5,
            distillation_topk=50,
            full_logit_distillation=True,
            distillation_is_clip=3.0,
            distillation_add_tail=True,
            distillation_weight=0.5,
            use_successful_as_teacher=False,
        )

        assert config.distillation_alpha == 0.5
        assert config.distillation_topk == 50
        assert config.full_logit_distillation is True
        assert config.distillation_is_clip == 3.0
        assert config.distillation_add_tail is True
        assert config.distillation_weight == 0.5
        assert config.use_successful_as_teacher is False

    def test_is_subclass_of_grpo_config(self):
        assert issubclass(SDPOConfig, GRPOConfig)


class TestSDPOTrainer(TrlTestCase):
    def test_is_subclass_of_grpo_trainer(self):
        assert issubclass(SDPOTrainer, GRPOTrainer)

    def test_has_sdpo_methods(self):
        assert hasattr(SDPOTrainer, "_compute_self_distillation_loss")
        assert hasattr(SDPOTrainer, "_compute_self_distillation_loss_core")
        assert hasattr(SDPOTrainer, "_compute_token_level_distillation_loss")
        assert hasattr(SDPOTrainer, "_apply_importance_sampling_clipping")
        assert hasattr(SDPOTrainer, "_get_teacher_log_probs")


class TestSDPOLossFunctions(TrlTestCase):
    """Unit tests for the core SDPO loss computation functions."""

    def setUp(self):
        super().setUp()
        # Create a minimal trainer instance for method access
        # We instantiate SDPOTrainer indirectly by testing the static-like methods
        self.B, self.T = 2, 4

    def test_token_level_distillation_loss(self):
        """Test reverse KL token-level distillation loss."""
        student_log_probs = torch.tensor([[-1.0, -2.0, -0.5, -1.5], [-0.8, -1.2, -0.3, -1.0]])
        teacher_log_probs = torch.tensor([[-0.9, -1.8, -0.6, -1.4], [-0.7, -1.0, -0.4, -0.9]])

        log_ratio = student_log_probs - teacher_log_probs
        expected = log_ratio.detach() * student_log_probs

        # Call the static computation directly
        actual = SDPOTrainer._compute_token_level_distillation_loss(None, student_log_probs, teacher_log_probs)
        torch.testing.assert_close(actual, expected)

    def test_token_level_distillation_loss_identical(self):
        """When student == teacher, loss should be zero."""
        log_probs = torch.tensor([[-1.0, -2.0, -0.5, -1.5]])
        loss = SDPOTrainer._compute_token_level_distillation_loss(None, log_probs, log_probs)
        torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_importance_sampling_clipping(self):
        """Test that IS clipping bounds the ratio correctly."""
        per_token_loss = torch.ones(2, 4)
        student_log_probs = torch.zeros(2, 4)
        old_log_probs = torch.full((2, 4), -10.0)  # Large gap -> large ratio
        clip_coeff = 2.0

        clipped = SDPOTrainer._apply_importance_sampling_clipping(
            None, per_token_loss, student_log_probs, old_log_probs, clip_coeff
        )
        # Ratio should be clamped to clip_coeff
        torch.testing.assert_close(clipped, torch.full((2, 4), clip_coeff))

    def test_importance_sampling_clipping_no_change(self):
        """When student == old, ratio should be 1 and loss unchanged."""
        per_token_loss = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        log_probs = torch.tensor([[-1.0, -2.0, -0.5, -1.5]])

        clipped = SDPOTrainer._apply_importance_sampling_clipping(None, per_token_loss, log_probs, log_probs, 2.0)
        torch.testing.assert_close(clipped, per_token_loss)

    def test_training(self):
        """Test that SDPOTrainer can train (distillation_weight=0 to avoid teacher issues)."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
            distillation_weight=0.0,  # disable distillation loss to test basic training loop
        )
        trainer = SDPOTrainer(
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
