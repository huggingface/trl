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

from types import SimpleNamespace

import pytest
import torch
from datasets import DatasetDict, load_dataset
from transformers import PretrainedConfig

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
    def test_rejects_liger_kernel(self, monkeypatch):
        monkeypatch.setattr("trl.experimental.gmpo.gmpo_trainer.GRPOTrainer.__init__", lambda *args, **kwargs: None)
        args = GMPOConfig(output_dir=self.tmp_dir, use_liger_kernel=True)

        with pytest.raises(NotImplementedError, match="Liger kernel is not supported by GMPOTrainer"):
            GMPOTrainer(PretrainedConfig(), reward_funcs=None, args=args)

    @pytest.mark.parametrize("use_adaptive_entropy", [False, True])
    def test_aux_loss_and_entropy_bonus(self, use_adaptive_entropy):
        trainer = GMPOTrainer.__new__(GMPOTrainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.beta = 0.0
        trainer.top_entropy_quantile = 1.0
        trainer.epsilon_low = 0.4
        trainer.epsilon_high = 0.4
        trainer.current_gradient_accumulation_steps = 1
        trainer.aux_loss_enabled = True
        trainer.router_aux_loss_coef = 0.1
        trainer._entropy_bonus_enabled = True
        trainer.use_adaptive_entropy = use_adaptive_entropy
        trainer.entropy_coef = 0.25
        trainer._last_world_entropy = 1.0
        trainer._entropy_window_stats = None
        trainer.args = SimpleNamespace(
            entropy_target=2.0, entropy_coef_delta=0.01, entropy_coef_max=1.0, entropy_coef_min=0.0
        )
        trainer._metrics = {
            "train": {
                "policy_loss": [],
                "aux_loss": [],
                "entropy_coef": [],
                "entropy": [],
                "clip_ratio/low_mean": [],
                "clip_ratio/high_mean": [],
                "clip_ratio/region_mean": [],
                "clip_ratio/low_min": [],
                "clip_ratio/high_max": [],
            }
        }

        class Accelerator:
            sync_gradients = True

            @staticmethod
            def gather(value):
                return value.reshape(1)

            @staticmethod
            def gather_for_metrics(value):
                return value.reshape(1)

            @staticmethod
            def reduce(value, reduction):
                return value

        trainer.accelerator = Accelerator()
        per_token_logps = torch.zeros((1, 2), requires_grad=True)
        entropies = torch.full((1, 2), 2.0, requires_grad=True)
        aux_loss = torch.tensor(3.0, requires_grad=True)

        def get_per_token_logps_and_entropies(*args, **kwargs):
            assert kwargs["compute_aux_loss"] is True
            assert "spatial_shapes" in kwargs
            assert "num_tiles" in kwargs
            return per_token_logps, entropies, aux_loss

        trainer._get_per_token_logps_and_entropies = get_per_token_logps_and_entropies
        inputs = {
            "prompt_ids": torch.tensor([[1]]),
            "prompt_mask": torch.ones((1, 1)),
            "completion_ids": torch.tensor([[2, 3]]),
            "completion_mask": torch.ones((1, 2)),
            "advantages": torch.tensor([1.0]),
        }

        loss = trainer._compute_loss(trainer.model, inputs)

        torch.testing.assert_close(loss, torch.tensor(-1.2))
        assert trainer._metrics["train"]["policy_loss"] == [-1.0]
        assert trainer._metrics["train"]["aux_loss"] == [3.0]

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
