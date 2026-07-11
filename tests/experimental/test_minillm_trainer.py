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

    def test_dapo_zv_with_rkl_advantage_raises_not_implemented(self):
        """`loss_type="dapo_zv"` combined with `rkl_advantage=True` (the default) must raise a clear error at
        construction: `compute_loss` mutates `inputs["advantages"]` with a reverse-KL term AFTER
        `num_items_in_batch_zv` has already been computed from the pre-mutation reward-tie signal, which would
        silently break the invariant `dapo_zv`'s denominator exclusion depends on."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = MiniLLMConfig(
            output_dir=self.tmp_dir,
            loss_type="dapo_zv",
            rkl_advantage=True,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        with pytest.raises(NotImplementedError, match="dapo_zv"):
            MiniLLMTrainer(
                model="trl-internal-testing/small-Qwen3ForCausalLM",
                teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
                args=training_args,
                train_dataset=dataset,
            )

    def test_dapo_zv_without_rkl_advantage_does_not_raise(self):
        """`loss_type="dapo_zv"` with `rkl_advantage=False` must NOT be blocked: `inputs["advantages"]` is never
        mutated in that case, so `dapo_zv` behaves exactly as it does on the base trainer."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = MiniLLMConfig(
            output_dir=self.tmp_dir,
            loss_type="dapo_zv",
            rkl_advantage=False,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        MiniLLMTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )
