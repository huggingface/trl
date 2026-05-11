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
from transformers.utils import is_peft_available

from trl.experimental.tpo import TPOConfig, TPOTrainer
from trl.experimental.tpo.tpo_trainer import DataCollatorForTriplePreference

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


def _add_reference_column(example):
    """Synthesize a `reference` (gold) completion for tests by reusing the chosen completion."""
    example["reference"] = example["chosen"]
    return example


class TestDataCollatorForTriplePreference(TrlTestCase):
    def test_padding_and_masks(self):
        collator = DataCollatorForTriplePreference(pad_token_id=0)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6], "reference_ids": [7, 8]},
            {"prompt_ids": [9, 10], "chosen_ids": [11], "rejected_ids": [12, 13], "reference_ids": [14]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [9, 10, 11, 0, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [9, 10, 12, 13, 0],  # prompt + rejected (example 2, padded)
                [1, 2, 3, 7, 8],  # prompt + reference (example 1)
                [9, 10, 14, 0, 0],  # prompt + reference (example 2, padded)
            ]
        )
        expected_attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
            ]
        )
        expected_completion_mask = torch.tensor(
            [
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 0, 0],
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
        torch.testing.assert_close(result["completion_mask"], expected_completion_mask)

    def test_exclude_reference(self):
        # When `include_reference=False`, the collator only emits the chosen/rejected halves so the per-step
        # compute/memory cost matches DPO's `DataCollatorForPreference`. This is the layout used by
        # `TPOTrainer` when `tpo_alpha=0.0`.
        collator = DataCollatorForTriplePreference(pad_token_id=0, include_reference=False)
        examples = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6], "reference_ids": [7, 8]},
            {"prompt_ids": [9, 10], "chosen_ids": [11], "rejected_ids": [12, 13], "reference_ids": [14]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5],  # prompt + chosen (example 1)
                [9, 10, 11, 0, 0],  # prompt + chosen (example 2, padded)
                [1, 2, 3, 6, 0],  # prompt + rejected (example 1, padded)
                [9, 10, 12, 13, 0],  # prompt + rejected (example 2, padded)
            ]
        )
        assert result["input_ids"].shape == (4, 5)  # 2 * B rows, no reference branch
        torch.testing.assert_close(result["input_ids"], expected_input_ids)
        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}


class TestTPOTrainer(TrlTestCase):
    def test_train(self):
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
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

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo", "tpo-l"])
    def test_train_loss_types(self, loss_type):
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            loss_type=loss_type,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            report_to="none",
            eval_strategy="steps",
            eval_steps=3,
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_conversational(self):
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
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

    def test_train_without_nll(self):
        # Setting tpo_alpha=0.0 disables the NLL term, skips the corresponding cross-entropy, and also drops the
        # reference branch from the collated batch so the model doesn't pay the extra forward-pass cost.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            tpo_alpha=0.0,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        # The default collator should drop the reference branch entirely when `tpo_alpha=0.0`.
        assert isinstance(trainer.data_collator, DataCollatorForTriplePreference)
        assert trainer.data_collator.include_reference is False

        # Verify the collated batch is 2 * per_device_train_batch_size (chosen + rejected only), not 3 * B.
        batch = trainer.data_collator(list(trainer.train_dataset.select(range(2))))
        assert batch["input_ids"].shape[0] == 4  # 2 branches * 2 examples

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_implicit_prompt(self):
        # Implicit-prompt variant: no `prompt` column, the prompt is embedded in `chosen`/`rejected` and (for TPO)
        # also in `reference`. Regression test for the `extract_prompt` bug where the reference column was left
        # untouched, silently doubling the prompt in the reference branch.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        # Synthesize a reference column that shares the same implicit prompt as chosen/rejected
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_implicit_prompt_mismatched_reference_raises(self):
        # When the dataset has no `prompt` column and the `reference` completion does not share the implicit
        # prompt prefix of `chosen`/`rejected`, the trainer must raise a clear error rather than silently
        # corrupting the reference branch.
        dataset = load_dataset("trl-internal-testing/zen", "standard_implicit_prompt_preference", split="train")

        def _set_unrelated_reference(example):
            example["reference"] = "unrelated completion without the shared prompt prefix."
            return example

        dataset = dataset.map(_set_unrelated_reference)

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="implicit prompt"):
            TPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_missing_reference_column_raises(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="reference"):
            TPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    @require_peft
    def test_train_with_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."
