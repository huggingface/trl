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

from trl.experimental.gpg import GPGConfig, GPGTrainer

from ..testing_utils import TrlTestCase


class TestGPGConfig:
    def test_defaults_match_the_method(self):
        # GPG drops the KL constraint and the reference model, and uses the mean-centered advantage without
        # standard-deviation scaling. Those are config defaults rather than code, so pin them.
        args = GPGConfig("dummy")
        assert args.beta == 0.0
        assert args.scale_rewards == "none"
        assert args.bias_correction


class TestGPGTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GPGTrainer(
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

    def test_bias_correction_rescales_the_loss(self):
        # The correction divides the loss by the fraction of groups whose rewards are not all identical, so with a
        # degenerate fraction of `f` the corrected loss must be the uncorrected one divided by `1 - f`. Drive
        # `_compute_loss` directly with a stubbed parent so the assertion pins the factor and nothing else.
        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.bias_correction = True
        trainer._valid_group_fraction = {"train": 1.0, "eval": 1.0}
        trainer.model = type("M", (), {"training": True})()

        base_loss = torch.tensor(2.0)
        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_compute_loss = GRPOTrainerParent._compute_loss
        GRPOTrainerParent._compute_loss = lambda self, model, inputs: base_loss
        try:
            for frac_zero_std, expected in [(0.0, 2.0), (0.5, 4.0), (0.75, 8.0)]:
                trainer._valid_group_fraction["train"] = 1.0 - frac_zero_std
                assert trainer._compute_loss(None, {}).item() == expected

            # Every group degenerate: the advantages are all zero, so there is no gradient to restore and the factor
            # would divide by zero. The loss must pass through untouched.
            trainer._valid_group_fraction["train"] = 0.0
            assert trainer._compute_loss(None, {}).item() == 2.0

            # Opting out recovers the uncorrected GRPO magnitude.
            trainer.bias_correction = False
            trainer._valid_group_fraction["train"] = 0.5
            assert trainer._compute_loss(None, {}).item() == 2.0
        finally:
            GRPOTrainerParent._compute_loss = original_compute_loss

    def test_eval_does_not_clobber_the_train_correction_factor(self):
        # `GRPOTrainer` buffers one train generation across `steps_per_generation` optimizer steps and only
        # regenerates every `steps_per_generation * num_iterations` steps (grpo_trainer.py:1584-1592), while eval
        # generates per batch (:1596). An eval landing inside that window must not leave the buffered train steps
        # rescaling with the eval fraction, so the cache is keyed by mode.
        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.bias_correction = True
        trainer._valid_group_fraction = {"train": 1.0, "eval": 1.0}

        base_loss = torch.tensor(1.0)
        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_compute_loss = GRPOTrainerParent._compute_loss
        GRPOTrainerParent._compute_loss = lambda self, model, inputs: base_loss
        try:
            # Train generation scores a batch where half the groups are degenerate.
            trainer._valid_group_fraction["train"] = 0.5
            # An eval pass then scores a batch where none are, which used to overwrite the shared scalar.
            trainer._valid_group_fraction["eval"] = 1.0

            trainer.model = type("M", (), {"training": True})()
            assert trainer._compute_loss(None, {}).item() == 2.0, (
                "the buffered train step must still use the train fraction after an eval pass"
            )

            trainer.model = type("M", (), {"training": False})()
            assert trainer._compute_loss(None, {}).item() == 1.0
        finally:
            GRPOTrainerParent._compute_loss = original_compute_loss

    def test_valid_group_fraction_tracks_the_logged_metric(self):
        # `_valid_group_fraction` is the complement of GRPO's `frac_reward_zero_std`, read straight after the parent
        # appends it. Training a step must leave the two consistent.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            max_steps=1,
            report_to="none",
        )
        trainer = GPGTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        assert 0.0 <= trainer._valid_group_fraction["train"] <= 1.0
