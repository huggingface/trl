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
from unittest.mock import patch

import pytest
import torch
from datasets import load_dataset
from transformers import TrainingArguments

from trl import GRPOTrainer, TargetPOConfig, TargetPOTrainer

from .testing_utils import TrlTestCase


class TestTargetPOConfig(TrlTestCase):
    def test_defaults_to_one_step_per_generation(self):
        config = TargetPOConfig(output_dir=self.tmp_dir, gradient_accumulation_steps=4)

        assert config.loss_type == "tpo"
        assert config.steps_per_generation == 1

    def test_allows_multi_step_generation_when_groups_are_whole(self):
        config = TargetPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            num_generations=2,
            steps_per_generation=2,
        )

        assert config.steps_per_generation == 2
        per_step_batch = config.generation_batch_size // config.steps_per_generation
        assert per_step_batch % config.num_generations == 0

    def test_rejects_multi_step_generation_when_groups_are_cleaved(self):
        with pytest.raises(ValueError, match="whole prompt groups"):
            TargetPOConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=2,
                num_generations=4,
                steps_per_generation=2,
            )

    def test_rejects_distributed_multi_step_generation_when_local_step_splits_groups(self):
        with patch.object(TrainingArguments, "world_size", new=property(lambda self: 2)):
            with pytest.raises(ValueError, match="per_device_train_batch_size"):
                TargetPOConfig(
                    output_dir=self.tmp_dir,
                    per_device_train_batch_size=3,
                    num_generations=2,
                    steps_per_generation=2,
                )

    def test_allows_distributed_single_step_generation_with_groups_spanning_ranks(self):
        with patch.object(TrainingArguments, "world_size", new=property(lambda self: 2)):
            config = TargetPOConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=3,
                num_generations=2,
                steps_per_generation=1,
            )

        assert config.generation_batch_size == 6
        assert config.steps_per_generation == 1

    def test_trainer_metadata(self):
        assert TargetPOTrainer._name == "TargetPO"
        assert TargetPOTrainer._tag_names == ["trl", "tpo"]


class TestTargetPOTrainer(TrlTestCase):
    def test_training(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = TargetPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = TargetPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."


class TestTPOLoss:
    def test_tpo_scores_match_population_whitened_skill(self):
        scores = torch.tensor([0.0, 1.0, 0.0, 0.0])

        tpo_scores = TargetPOTrainer.get_tpo_scores(scores, num_generations=2)

        expected = torch.tensor([-1.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(tpo_scores, expected)

    def test_tpo_scores_exclude_invalid_completions(self):
        scores = torch.tensor([0.0, 100.0, 1.0])
        valid_mask = torch.tensor([True, False, True])

        tpo_scores = TargetPOTrainer.get_tpo_scores(scores, num_generations=3, valid_mask=valid_mask)

        expected = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(tpo_scores, expected)

    def test_tpo_targets_use_population_whitened_scores(self):
        old_sequence_logps = torch.zeros(2)
        scores = torch.tensor([0.0, 1.0])
        tpo_scores = TargetPOTrainer.get_tpo_scores(scores, num_generations=2)

        targets = TargetPOTrainer.get_tpo_targets(old_sequence_logps, tpo_scores, num_generations=2)

        expected = torch.softmax(torch.tensor([-1.0, 1.0]), dim=0)
        torch.testing.assert_close(targets, expected)

    def test_tpo_targets_match_closed_form(self):
        old_sequence_logps = torch.log(torch.tensor([0.7, 0.3, 0.2, 0.8]))
        scores = torch.tensor([1.0, -1.0, 0.0, 0.0])

        targets = TargetPOTrainer.get_tpo_targets(old_sequence_logps, scores, num_generations=2)

        expected_first_group = torch.softmax(torch.log(torch.tensor([0.7, 0.3])) + torch.tensor([1.0, -1.0]), dim=0)
        expected_second_group = torch.tensor([0.2, 0.8])
        expected = torch.cat([expected_first_group, expected_second_group])
        torch.testing.assert_close(targets, expected)

    def test_tpo_targets_exclude_invalid_completions(self):
        old_sequence_logps = torch.log(torch.tensor([0.7, 0.3, 0.2, 0.8]))
        scores = torch.tensor([1.0, -1.0, 0.0, 0.0])
        valid_mask = torch.tensor([True, False, True, True])

        targets = TargetPOTrainer.get_tpo_targets(old_sequence_logps, scores, num_generations=2, valid_mask=valid_mask)

        expected = torch.tensor([1.0, 0.0, 0.2, 0.8])
        torch.testing.assert_close(targets, expected)

    @pytest.mark.parametrize("trainer_cls", [GRPOTrainer, TargetPOTrainer])
    def test_tpo_kl_uses_per_step_token_normalizer(self, trainer_cls):
        trainer = object.__new__(trainer_cls)
        trainer.loss_type = "tpo"
        trainer.off_policy_mask_threshold = None
        trainer.top_entropy_quantile = 1.0
        trainer.beta = 0.25
        trainer.num_generations = 2
        trainer.num_generations_eval = 2
        trainer.current_gradient_accumulation_steps = 2
        trainer.tpo_length_normalize_logps = False
        trainer.args = SimpleNamespace(use_bias_correction_kl=False)
        trainer.model = SimpleNamespace(training=True)
        trainer.accelerator = SimpleNamespace(
            gather=lambda tensor: tensor,
            num_processes=1,
            process_index=0,
        )
        trainer._metrics = {"train": defaultdict(list)}

        per_token_logps = torch.zeros((2, 3))

        def fake_get_per_token_logps_and_entropies(*args, **kwargs):
            return per_token_logps, torch.zeros_like(per_token_logps)

        trainer._get_per_token_logps_and_entropies = fake_get_per_token_logps_and_entropies

        completion_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        ref_log_ratio = torch.log(torch.tensor(2.0))
        ref_per_token_logps = per_token_logps + ref_log_ratio
        inputs = {
            "prompt_ids": torch.zeros((2, 1), dtype=torch.long),
            "prompt_mask": torch.ones((2, 1), dtype=torch.long),
            "completion_ids": torch.ones((2, 3), dtype=torch.long),
            "completion_mask": completion_mask,
            "advantages": torch.zeros(2),
            "old_per_token_logps": per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "tpo_targets": torch.tensor([0.5, 0.5]),
            "num_items_in_batch": torch.tensor(6),
        }

        loss = trainer_cls._compute_loss(trainer, model=None, inputs=inputs)

        normalizer = torch.tensor(float(trainer.current_gradient_accumulation_steps))
        tpo_loss = ref_log_ratio / normalizer
        expected_kl_loss = torch.exp(ref_log_ratio) - ref_log_ratio - 1
        expected_loss = tpo_loss + trainer.beta * expected_kl_loss / normalizer
        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(torch.tensor(trainer._metrics["train"]["kl"][0]), expected_kl_loss)

    @pytest.mark.parametrize("length_normalize", [True, False])
    def test_tpo_loss_gradient_matches_policy_minus_target(self, length_normalize):
        trainer = object.__new__(TargetPOTrainer)
        trainer.loss_type = "tpo"
        trainer.off_policy_mask_threshold = None
        trainer.top_entropy_quantile = 1.0
        trainer.beta = 0.0
        trainer.num_generations = 2
        trainer.num_generations_eval = 2
        trainer.current_gradient_accumulation_steps = 1
        trainer.tpo_length_normalize_logps = length_normalize
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

        loss = TargetPOTrainer._compute_loss(trainer, model=None, inputs=inputs)
        loss.backward()

        completion_mask = inputs["completion_mask"].to(per_token_logps.dtype)
        lengths = completion_mask.sum(dim=-1).clamp(min=1)
        summed = (per_token_logps.detach() * completion_mask).sum(dim=-1)
        sequence_logps = summed / lengths if length_normalize else summed
        expected_sequence_grad = torch.softmax(sequence_logps, dim=0) - tpo_targets
        per_token_scale = (1.0 / lengths) if length_normalize else torch.ones_like(lengths)
        expected_grad = (expected_sequence_grad * per_token_scale).unsqueeze(1) * completion_mask
        torch.testing.assert_close(per_token_logps.grad, expected_grad)

    def test_tpo_loss_excludes_invalid_completions_from_group_softmax(self):
        trainer = object.__new__(TargetPOTrainer)
        trainer.loss_type = "tpo"
        trainer.off_policy_mask_threshold = None
        trainer.top_entropy_quantile = 1.0
        trainer.beta = 0.0
        trainer.num_generations = 3
        trainer.num_generations_eval = 3
        trainer.current_gradient_accumulation_steps = 1
        trainer.tpo_length_normalize_logps = True
        trainer.model = SimpleNamespace(training=True)
        trainer.accelerator = SimpleNamespace(
            gather=lambda tensor: tensor,
            num_processes=1,
            process_index=0,
        )
        trainer._metrics = {"train": defaultdict(list)}

        per_token_logps = torch.tensor([[-0.1, -0.2], [5.0, 5.0], [-0.4, -0.6]], requires_grad=True)

        def fake_get_per_token_logps_and_entropies(*args, **kwargs):
            return per_token_logps, torch.zeros_like(per_token_logps)

        trainer._get_per_token_logps_and_entropies = fake_get_per_token_logps_and_entropies

        tpo_targets = torch.tensor([0.6, 0.0, 0.4])
        tpo_valid_mask = torch.tensor([True, False, True])
        inputs = {
            "prompt_ids": torch.zeros((3, 1), dtype=torch.long),
            "prompt_mask": torch.ones((3, 1), dtype=torch.long),
            "completion_ids": torch.ones((3, 2), dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1], [0, 0], [1, 1]]),
            "advantages": torch.zeros(3),
            "old_per_token_logps": per_token_logps.detach(),
            "tpo_targets": tpo_targets,
            "tpo_valid_mask": tpo_valid_mask,
            "num_items_in_batch": torch.tensor(4),
        }

        loss = TargetPOTrainer._compute_loss(trainer, model=None, inputs=inputs)
        loss.backward()

        completion_mask = inputs["completion_mask"].to(per_token_logps.dtype)
        lengths = completion_mask.sum(dim=-1).clamp(min=1)
        sequence_logps = (per_token_logps.detach() * completion_mask).sum(dim=-1) / lengths
        valid_sequence_logps = sequence_logps[tpo_valid_mask]
        expected_valid_grad = torch.softmax(valid_sequence_logps, dim=0) - tpo_targets[tpo_valid_mask]
        expected_grad = torch.zeros_like(per_token_logps)
        expected_grad[0] = (expected_valid_grad[0] / lengths[0]) * completion_mask[0]
        expected_grad[2] = (expected_valid_grad[1] / lengths[2]) * completion_mask[2]
        torch.testing.assert_close(per_token_logps.grad, expected_grad)
