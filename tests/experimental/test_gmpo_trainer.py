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
from datasets import DatasetDict, load_dataset

from trl.experimental.gmpo import GMPOConfig, GMPOTrainer

from ..testing_utils import TrlTestCase


class TestGMPOConfig:
    def test_default_epsilon_is_log_space(self):
        # GMPO expresses the clip range in log space; default is the paper's (exp(-0.4), exp(0.4)).
        args = GMPOConfig("dummy")
        assert args.epsilon == 0.4
        # epsilon_high is inherited from GRPOConfig and defaults to None, so the range is symmetric.
        assert args.epsilon_high is None


class TestGMPOTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            report_to="none",
        )
        trainer = GMPOTrainer(
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

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # Streaming datasets are not yet supported in GMPO
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = GMPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_conversational(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            report_to="none",
        )
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_logs_policy_loss(self):
        # Issue #7005: the override never logged `policy_loss`. GRPOTrainer disables the HF Trainer's own accumulation
        # rescale (it sets `compute_loss_func`) and divides the returned loss by `current_gradient_accumulation_steps`
        # itself, so with two accumulation steps the reported `loss` is the sum of two half-losses, which equals the
        # mean of the two captured values. `policy_loss` must therefore equal `loss`; capturing it after the rescale
        # would report half of that. In eval there is no rescale, so the two coincide there as well.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=6,
            gradient_accumulation_steps=2,
            num_generations=3,
            max_completion_length=8,
            beta=0.0,
            max_steps=2,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=2,
            report_to="none",
        )
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )
        trainer.train()

        train_logs = [log for log in trainer.state.log_history if "loss" in log]
        assert len(train_logs) == 2
        for log in train_logs:
            assert "policy_loss" in log, f"`policy_loss` missing from {log}"
            assert log["policy_loss"] == pytest.approx(log["loss"], rel=1e-4)

        # The eval set holds 2 prompts, so a batch of 6 evaluates them in one step. With several eval batches the HF
        # loop weights `eval_loss` per example while TRL averages its metrics per batch, and the two no longer agree.
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        assert len(eval_logs) == 1
        assert eval_logs[0]["eval_policy_loss"] == pytest.approx(eval_logs[0]["eval_loss"], rel=1e-4)

    def test_train_with_kl(self):
        # GMPO sequence-averages the KL when beta > 0; exercise that path.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            beta=0.1,  # enable KL regularization toward the reference model
            report_to="none",
        )
        trainer = GMPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "kl" in trainer.state.log_history[-1]  # KL is logged when beta > 0

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
