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

    @pytest.mark.parametrize("beta", [0.0, 0.1])
    def test_train_logs_policy_loss(self, beta):
        # Issue #7005: the override never logged `policy_loss`. GMPO divides the returned loss by
        # `current_gradient_accumulation_steps` itself, so with two accumulation steps each step returns two
        # half-losses, and `policy_loss` (the mean of the two captured values) must equal their sum; capturing it after
        # the rescale would report half of that. In eval there is no rescale. The returned losses are recorded
        # directly: the `loss` in `log_history` is not a usable witness, since transformers < 5 rounds it to 4
        # decimals. With `beta != 0` the KL term is folded into the loss before the capture, so the identity only holds
        # for a capture after that fold; the `beta=0.0` row alone cannot tell the two apart.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        training_args = GMPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            per_device_eval_batch_size=6,  # must be a multiple of num_generations
            gradient_accumulation_steps=2,
            num_generations=3,
            max_completion_length=8,
            beta=beta,
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
        original_compute_loss = trainer._compute_loss
        returned_losses = {"train": [], "eval": []}

        def record_loss(model, inputs):
            loss = original_compute_loss(model, inputs)
            returned_losses["train" if model.training else "eval"].append(loss.item())
            return loss

        trainer._compute_loss = record_loss

        trainer.train()

        accumulation_steps = training_args.gradient_accumulation_steps
        train_logs = [log for log in trainer.state.log_history if "loss" in log]
        assert len(train_logs) == 2
        assert len(returned_losses["train"]) == accumulation_steps * len(train_logs)
        for step, log in enumerate(train_logs):
            assert "policy_loss" in log, f"`policy_loss` missing from {log}"
            step_losses = returned_losses["train"][step * accumulation_steps : (step + 1) * accumulation_steps]
            assert log["policy_loss"] == pytest.approx(sum(step_losses), rel=1e-4)

        # The eval metrics are means over the eval batches, so the identity holds against the mean returned loss.
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        assert len(eval_logs) == 1
        assert returned_losses["eval"]
        mean_eval_loss = sum(returned_losses["eval"]) / len(returned_losses["eval"])
        assert eval_logs[0]["eval_policy_loss"] == pytest.approx(mean_eval_loss, rel=1e-4)

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
