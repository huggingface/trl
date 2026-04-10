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
from transformers import AutoModelForCausalLM
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

    def test_with_pad_to_multiple_of(self):
        collator = DataCollatorForTriplePreference(pad_token_id=0, pad_to_multiple_of=5)
        examples = [
            {"prompt_ids": [1], "chosen_ids": [2], "rejected_ids": [3], "reference_ids": [4]},
            {"prompt_ids": [5, 6], "chosen_ids": [7, 8], "rejected_ids": [9, 10], "reference_ids": [11, 12]},
        ]
        result = collator(examples)

        expected_input_ids = torch.tensor(
            [
                [1, 2, 0, 0, 0],  # prompt + chosen (example 1, padded to multiple of 5)
                [5, 6, 7, 8, 0],  # prompt + chosen (example 2, padded to multiple of 5)
                [1, 3, 0, 0, 0],  # prompt + rejected (example 1, padded to multiple of 5)
                [5, 6, 9, 10, 0],  # prompt + rejected (example 2, padded to multiple of 5)
                [1, 4, 0, 0, 0],  # prompt + reference (example 1, padded to multiple of 5)
                [5, 6, 11, 12, 0],  # prompt + reference (example 2, padded to multiple of 5)
            ]
        )

        assert set(result.keys()) == {"input_ids", "attention_mask", "completion_mask"}
        torch.testing.assert_close(result["input_ids"], expected_input_ids)


class TestTPOTrainer(TrlTestCase):
    def test_train(self):
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        # Initialize the trainer
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

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_model(self):
        # Instantiate the model
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            dtype="float32",
        )

        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        # Initialize the trainer
        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            report_to="none",
        )
        trainer = TPOTrainer(model=model, args=training_args, train_dataset=dataset)

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge", "ipo", "tpo-l"])
    def test_train_loss_types(self, loss_type):
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference")
        dataset = dataset.map(_add_reference_column)

        # Initialize the trainer
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

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_conversational(self):
        # Get the dataset and synthesize a reference (gold) completion
        dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        # Initialize the trainer
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

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

    def test_train_without_nll(self):
        # Setting tpo_alpha=0.0 disables the NLL term and skips the corresponding cross-entropy
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

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"

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
                assert not torch.allclose(param, new_param), f"Parameter {n} has not changed"
