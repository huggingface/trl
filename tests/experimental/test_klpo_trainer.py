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

from trl.experimental.klpo import KLPOConfig, KLPOTrainer

from ..testing_utils import TrlTestCase


class TestKLPOConfig:
    def test_single_rollout_defaults(self):
        # KLPO is a single-rollout method: num_generations=1 must be accepted (GRPOConfig rejects < 2).
        args = KLPOConfig("dummy")
        assert args.num_generations == 1
        assert args.klpo_beta == 0.1
        assert args.mc_samples == 128
        # The inherited reference-model KL stays disabled; KLPO regularizes toward the sampler instead.
        assert args.beta == 0.0
        # Rewards are consumed raw, so group scaling is disabled for single rollouts.
        assert args.scale_rewards == "none"

    def test_invalid_mc_samples(self):
        with pytest.raises(ValueError, match="mc_samples"):
            KLPOConfig("dummy", mc_samples=0)

    def test_invalid_route_and_estimator(self):
        with pytest.raises(ValueError, match="klpo_route"):
            KLPOConfig("dummy", klpo_route="tokens")
        with pytest.raises(ValueError, match="kl_estimator"):
            KLPOConfig("dummy", kl_estimator="montecarlo")

    def test_sequence_mc_needs_two_samples(self):
        # Sequence MC-KL uses leave-one-out residuals, which require M >= 2.
        with pytest.raises(ValueError, match="leave-one-out"):
            KLPOConfig("dummy", klpo_route="sequence", kl_estimator="mc", mc_samples=1)


class TestKLPOTrainer(TrlTestCase):
    def test_train(self):
        # Single rollout per prompt (num_generations=1), the KLPO default.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,  # single rollout per prompt, the KLPO default
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "kl" in trainer.state.log_history[-1]  # the MC-KL estimate is logged

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multiple_iterations(self):
        # With num_iterations > 1, the batch is reused: the sampler records stay fixed while the policy drifts,
        # exercising the off-policy path (ell != 0).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            num_iterations=2,
            report_to="none",
        )
        trainer = KLPOTrainer(
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

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_conversational(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.parametrize(
        "route, estimator",
        [
            # token + mc is the default and is covered by test_train
            ("token", "topk"),
            ("token", "binary"),
            ("token", "full"),
            ("sequence", "mc"),
            ("sequence", "topk"),
            ("sequence", "binary"),
            ("sequence", "full"),
        ],
    )
    def test_train_routes_and_estimators(self, route, estimator):
        # All route/estimator combinations must train and move the parameters.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            klpo_route=route,
            kl_estimator=estimator,
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            kl_top_k=2,  # small head so the tail bucket is exercised
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "kl" in trainer.state.log_history[-1]

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_num_generations_gt1(self):
        # KLPO does not need groups, but grouped generation must still work (rewards stay raw).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
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
