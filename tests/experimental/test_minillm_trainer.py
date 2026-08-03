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

import types

import pytest
import torch
from datasets import DatasetDict, load_dataset

from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer

from ..testing_utils import TrlTestCase


@pytest.mark.low_priority
class TestMiniLLMTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = MiniLLMConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=32,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = MiniLLMTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
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
        # Streaming datasets are not yet supported in MiniLLM
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = MiniLLMConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = MiniLLMTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
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


class TestMiniLLMComputeAdvantage(TrlTestCase):
    """Unit tests for the discounted advantage (issue #6626). `_compute_advantage` only reads `self.gamma` and
    `self.length_normalization`, so it is exercised with a stub `self` and no model."""

    def test_discount_is_relative_not_absolute(self):
        # advantages_t must be sum_{i>=t} gamma^(i-t) R_i, not gamma^t times that. With a constant reward of
        # 1.0 (teacher=1, student=0) and no length normalization, the closed form is
        # A_t = (1 - gamma^(T-t)) / (1 - gamma). The absolute-index gamma^i weighting scaled every position
        # by an extra gamma^t, suppressing later tokens toward zero.
        gamma, seq_len = 0.9, 64
        student = torch.zeros(1, seq_len)
        teacher = torch.ones(1, seq_len)
        stub = types.SimpleNamespace(gamma=gamma, length_normalization=False)

        advantages = MiniLLMTrainer._compute_advantage(stub, student, teacher)[0]

        t = torch.arange(seq_len)
        expected = (1 - gamma ** (seq_len - t)) / (1 - gamma)
        torch.testing.assert_close(advantages, expected.float())

    def test_length_normalized_is_finite_on_long_sequences(self):
        # With length normalization the gamma^t factor cancels, so a constant reward gives advantage 1.0 at
        # every position. The absolute-index gamma^i weighting underflowed to 0.0 in float32 on long
        # completions, turning the ratio into 0/0 = nan.
        gamma, seq_len = 0.5, 512
        student = torch.zeros(1, seq_len)
        teacher = torch.ones(1, seq_len)
        stub = types.SimpleNamespace(gamma=gamma, length_normalization=True)

        advantages = MiniLLMTrainer._compute_advantage(stub, student, teacher)[0]

        assert torch.isfinite(advantages).all(), "length-normalized advantages contain non-finite values"
        torch.testing.assert_close(advantages, torch.ones(seq_len))
