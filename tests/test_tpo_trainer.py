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

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl import TPOConfig, TPOTrainer
from trl.trainer.grpo_trainer import GRPOTrainer

from .testing_utils import TrlTestCase


class TestTPOConfig(TrlTestCase):
    def test_defaults_to_one_step_per_generation(self):
        config = TPOConfig(output_dir=self.tmp_dir, gradient_accumulation_steps=4)

        assert config.loss_type == "tpo"
        assert config.steps_per_generation == 1

    def test_requires_one_step_per_generation(self):
        with pytest.raises(ValueError, match="steps_per_generation=1"):
            TPOConfig(output_dir=self.tmp_dir, steps_per_generation=2)

    def test_trainer_metadata(self):
        assert TPOTrainer._name == "TPO"
        assert TPOTrainer._tag_names == ["trl", "tpo"]


class TestTPOLoss:
    def test_tpo_scores_match_population_whitened_skill(self):
        scores = torch.tensor([0.0, 1.0, 0.0, 0.0])

        tpo_scores = GRPOTrainer.get_tpo_scores(scores, num_generations=2)

        expected = torch.tensor([-1.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(tpo_scores, expected)

    def test_tpo_targets_use_population_whitened_scores(self):
        old_sequence_logps = torch.zeros(2)
        scores = torch.tensor([0.0, 1.0])
        tpo_scores = GRPOTrainer.get_tpo_scores(scores, num_generations=2)

        targets = GRPOTrainer.get_tpo_targets(old_sequence_logps, tpo_scores, num_generations=2)

        expected = torch.softmax(torch.tensor([-1.0, 1.0]), dim=0)
        torch.testing.assert_close(targets, expected)

    def test_tpo_targets_match_closed_form(self):
        old_sequence_logps = torch.log(torch.tensor([0.7, 0.3, 0.2, 0.8]))
        scores = torch.tensor([1.0, -1.0, 0.0, 0.0])

        targets = GRPOTrainer.get_tpo_targets(old_sequence_logps, scores, num_generations=2)

        expected_first_group = torch.softmax(torch.log(torch.tensor([0.7, 0.3])) + torch.tensor([1.0, -1.0]), dim=0)
        expected_second_group = torch.tensor([0.2, 0.8])
        expected = torch.cat([expected_first_group, expected_second_group])
        torch.testing.assert_close(targets, expected)

    def test_tpo_loss_gradient_matches_policy_minus_target(self):
        trainer = object.__new__(GRPOTrainer)
        trainer.loss_type = "tpo"
        trainer.off_policy_mask_threshold = None
        trainer.top_entropy_quantile = 1.0
        trainer.beta = 0.0
        trainer.num_generations = 2
        trainer.num_generations_eval = 2
        trainer.current_gradient_accumulation_steps = 1
        trainer.model = SimpleNamespace(training=True)
        trainer.accelerator = SimpleNamespace(
            gather=lambda tensor: tensor,
            num_processes=1,
            process_index=0,
        )
        trainer._metrics = {"train": defaultdict(list)}

        per_token_logps = torch.tensor([[-0.1, -0.2], [-0.7, -0.3]], requires_grad=True)

        def fake_get_per_token_logps_and_entropies(*args, **kwargs):
            return per_token_logps, torch.zeros_like(per_token_logps)

        trainer._get_per_token_logps_and_entropies = fake_get_per_token_logps_and_entropies

        tpo_targets = torch.tensor([0.25, 0.75])
        inputs = {
            "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
            "prompt_mask": torch.ones((2, 1), dtype=torch.long),
            "completion_ids": torch.ones((2, 2), dtype=torch.long),
            "completion_mask": torch.ones((2, 2), dtype=torch.long),
            "advantages": torch.zeros(2),
            "old_per_token_logps": per_token_logps.detach(),
            "tpo_targets": tpo_targets,
            "num_items_in_batch": torch.tensor(4),
        }

        loss = GRPOTrainer._compute_loss(trainer, model=None, inputs=inputs)
        loss.backward()

        sequence_logps = per_token_logps.detach().sum(dim=-1)
        expected_sequence_grad = torch.softmax(sequence_logps, dim=0) - tpo_targets
        expected_grad = expected_sequence_grad.unsqueeze(1).expand_as(per_token_logps)
        torch.testing.assert_close(per_token_logps.grad, expected_grad)
