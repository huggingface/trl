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
from transformers.utils import is_torch_xla_available

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

    @pytest.mark.parametrize("poison", [float("nan"), float("inf")])
    def test_nonfinite_loss_is_visible_in_log_history(self, poison):
        """A non-finite loss must reach `log_history`, which `logging_nan_inf_filter` otherwise hides."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        class NonFiniteLossMiniLLMTrainer(MiniLLMTrainer):
            # Both NaN and Inf are injected, because the guard tests `~isfinite` and a suite that only ever injects
            # one of them is passed by the matching `isnan` or `isinf` implementation. Adding rather than
            # multiplying leaves the gradients finite, so the poisoned step does not corrupt the weights, and is
            # invariant to a loss of exactly `0.0`, for which `0.0 * inf` would be NaN.
            def _compute_loss(self, model, inputs):
                loss = super()._compute_loss(model, inputs)
                return loss + poison if self.state.global_step == 1 else loss

        training_args = MiniLLMConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=32,
            max_steps=2,
            logging_steps=1,
            report_to="none",
        )
        trainer = NonFiniteLossMiniLLMTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        healthy_step, poisoned_step = trainer.state.log_history[0], trainer.state.log_history[1]
        assert healthy_step["frac_nonfinite_loss"] == 0.0
        assert poisoned_step["frac_nonfinite_loss"] == 1.0
        # `logging_nan_inf_filter` is enabled by default, so `transformers` discards the step's own non-finite loss
        # and substitutes a value derived from the loss accumulated since the last log. The reported loss therefore
        # stays finite and the failing step is invisible, which is why the metric above is needed. The filter is
        # gated on `not is_torch_xla_available()`, so under XLA the non-finite loss reaches the log unchanged and
        # the substitution this metric compensates for does not happen.
        if is_torch_xla_available():
            assert not torch.isfinite(torch.tensor(poisoned_step["loss"]))
        else:
            assert torch.isfinite(torch.tensor(poisoned_step["loss"]))

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
