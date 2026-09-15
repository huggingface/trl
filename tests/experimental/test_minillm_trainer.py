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


class TestMiniLLMComputeAdvantage:
    def test_length_normalization_ignores_padding(self):
        """A constant reward of 1.0 with gamma=1.0 must give an advantage of exactly 1.0 at every valid position,
        however long the batch is padded. Counting padded slots toward the length, as the old 1e-4 fill did, lowers the
        first position of a short completion (0.9514 for a single valid token padded to 512), and a floor on the
        discounted length would deflate late positions once gamma < 1."""
        for gamma in (1.0, 0.9):
            stub = types.SimpleNamespace(gamma=gamma, length_normalization=True)
            for length in (1, 128, 512):
                mask = torch.zeros(1, 512)
                mask[0, :length] = 1
                advantages = MiniLLMTrainer._compute_advantage(stub, torch.zeros(1, 512), torch.ones(1, 512), mask)
                # With gamma < 1 the discounted length of a late position is tiny (0.9**300 is 1.9e-14), so any floor
                # on the denominator would deflate it; the expected value is exactly the reward at every position.
                torch.testing.assert_close(advantages[0, :length], torch.ones(length))
                torch.testing.assert_close(advantages[0, length:], torch.zeros(512 - length))

            advantages = MiniLLMTrainer._compute_advantage(
                stub, torch.zeros(1, 8), torch.ones(1, 8), torch.zeros(1, 8)
            )
            assert torch.isfinite(advantages).all()
            torch.testing.assert_close(advantages, torch.zeros(1, 8))


class TestMiniLLMComputeLossMask:
    def test_advantage_mask_excludes_padded_completion_tokens(self, monkeypatch):
        """The mask handed to `_compute_advantage` must come from the -100 fill in `labels`; built from `input_ids`
        it was all ones, so padded slots kept a reward and leaked into every earlier advantage."""

        class DummyModel:
            def eval(self):
                return self

            def __call__(self, input_ids, attention_mask, use_cache):
                return types.SimpleNamespace(logits=torch.zeros(*input_ids.shape, 10))

        trainer = object.__new__(MiniLLMTrainer)
        trainer.teacher_model = DummyModel()
        trainer.kd_temperature = 1.0
        trainer.rkl_advantage = True

        completion_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        inputs = {
            "prompt_ids": torch.randint(1, 10, (2, 3)),
            "prompt_mask": torch.ones(2, 3, dtype=torch.long),
            "completion_ids": torch.randint(1, 10, (2, 4)),
            "completion_mask": completion_mask,
        }

        class AdvantageCaptured(Exception):
            pass

        def spy(**kwargs):
            assert torch.equal(kwargs["mask"], completion_mask.bool())
            raise AdvantageCaptured

        monkeypatch.setattr(trainer, "_compute_advantage", spy)
        with pytest.raises(AdvantageCaptured):
            trainer.compute_loss(DummyModel(), inputs)
