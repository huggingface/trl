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
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from transformers.utils import is_peft_available, is_torch_xla_available

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

    def test_nonfinite_loss_is_visible_in_log_history(self):
        """A non-finite loss must reach `log_history`, which `logging_nan_inf_filter` otherwise hides."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        class NonFiniteLossTPOTrainer(TPOTrainer):
            def _compute_loss(self, model, inputs, return_outputs):
                if self.state.global_step == 1:

                    def poison_logits(module, args, output):
                        output.logits = output.logits * float("nan")
                        return output

                    handle = model.register_forward_hook(poison_logits)
                    try:
                        return super()._compute_loss(model, inputs, return_outputs)
                    finally:
                        handle.remove()
                return super()._compute_loss(model, inputs, return_outputs)

        training_args = TPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            max_steps=2,
            logging_steps=1,
            report_to="none",
        )
        trainer = NonFiniteLossTPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
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

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # Get the dataset and synthesize a reference (gold) completion
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        train_dataset = train_dataset.map(_add_reference_column)

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming
            )
            eval_split = eval_split.map(_add_reference_column)
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "prompt_ids" in next(iter(trainer.eval_dataset))

    def test_evaluate_with_raw_dataset(self):
        # `evaluate` should accept the same (unprocessed) dataset types as the trainer, e.g. a held-out test set
        # passed directly to `evaluate`, mirroring DPO/KTO. See https://github.com/huggingface/trl/issues/6115.
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)

        training_args = TPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = TPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", args=training_args, train_dataset=dataset
        )

        metrics = trainer.evaluate(eval_dataset=dataset)
        assert metrics["eval_loss"] is not None

    def test_trust_remote_code(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        dataset = dataset.map(_add_reference_column)
        model_id = "trl-internal-testing/tiny-RemoteForCausalLM"

        with pytest.raises(ValueError, match="custom code"):
            TPOTrainer(
                model=model_id,
                args=TPOConfig(output_dir=self.tmp_dir, report_to="none"),
                train_dataset=dataset,
            )

        trainer = TPOTrainer(
            model=model_id,
            args=TPOConfig(output_dir=self.tmp_dir, report_to="none", trust_remote_code=True),
            train_dataset=dataset,
        )
        assert type(trainer.model).__name__ == "RemoteForCausalLM"

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
