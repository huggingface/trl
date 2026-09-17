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
from datasets import DatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import is_peft_available

from trl.experimental.xpo import XPOConfig, XPOTrainer

from ..testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model


@pytest.mark.low_priority
class TestXPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_xpo_trainer_training(self, config_name):
        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        trainer = XPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # Streaming datasets are not yet supported in XPO
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = XPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = XPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    def test_evaluate(self):
        # `Trainer.prediction_step` never reaches `compute_loss` for this trainer: it gates on
        # `has_labels or loss_without_labels`, and an evaluation batch carries only prompts, so evaluation used to
        # fall through to calling the model without `input_ids`. See https://github.com/huggingface/trl/issues/2228.
        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=1,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = XPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=self.tokenizer,
        )

        # A stubbed `prediction_step` returning any constant passes a magnitude bound, so pin `eval_loss` to the losses
        # `_compute_loss` produced on the evaluation batches, weighted by batch size as `Trainer` does.
        recorded = []
        original = trainer._compute_loss

        def spy(model, inputs, log_stats=True):
            loss = original(model, inputs, log_stats=log_stats)
            batch_size = len(next(iter(inputs.values())))
            recorded.append((log_stats, batch_size, loss.item()))
            return loss

        trainer._compute_loss = spy
        metrics = trainer.evaluate()

        assert len(recorded) == len(trainer.get_eval_dataloader())
        assert all(log_stats is False for log_stats, _, _ in recorded)
        expected_loss = sum(batch_size * loss for _, batch_size, loss in recorded) / sum(
            batch_size for _, batch_size, _ in recorded
        )
        assert metrics["eval_loss"] == pytest.approx(expected_loss, abs=1e-5)

    def test_evaluate_does_not_pollute_training_stats(self):
        # `self.stats` is averaged and cleared by the training logger, so appending evaluation values to it would
        # shift the training metrics.
        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=1,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = XPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=self.tokenizer,
        )
        # Train first: comparing an empty dict against an empty dict would pass even if `_compute_loss` had
        # stopped recording statistics altogether, so the lengths have to be non-zero for the comparison to mean
        # anything.
        trainer.train()
        lengths = {key: len(value) for key, value in trainer.stats.items()}
        assert lengths and all(length > 0 for length in lengths.values())

        trainer.evaluate()

        assert {key: len(value) for key, value in trainer.stats.items()} == lengths

    @require_peft
    def test_train_with_peft(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = XPOTrainer(
            model=self.model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_train_with_peft_and_ref_model(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            learning_rate=5.0e-7,
            report_to="none",
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = XPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            peft_config=lora_config,
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]

    @require_peft
    def test_train_pre_pefted_model_implicit_ref(self):
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        peft_model_instance = get_peft_model(self.model, lora_config)

        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_steps=2,
            learning_rate=5.0e-7,
            eval_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        )
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        trainer = XPOTrainer(
            model=peft_model_instance,
            ref_model=None,
            reward_funcs=self.reward_model,  # Using reward_model to ensure _generate_completions is used as expected
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dataset,
        )

        trainer.train()

        assert "train_loss" in trainer.state.log_history[-1]
