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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import is_peft_available, is_torch_xla_available

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

    @pytest.mark.parametrize("poison", [float("nan"), float("inf")])
    def test_nonfinite_loss_is_visible_in_log_history(self, poison):
        """A non-finite loss must reach `log_history`, which `logging_nan_inf_filter` otherwise hides."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        class NonFiniteLossXPOTrainer(XPOTrainer):
            # Both NaN and Inf are injected, because the guard tests `~isfinite` and a suite that only ever injects
            # one of them is passed by the matching `isnan` or `isinf` implementation. Adding rather than
            # multiplying leaves the gradients finite, so the poisoned step does not corrupt the weights, and is
            # invariant to a loss of exactly `0.0`, for which `0.0 * inf` would be NaN.
            def _compute_losses(
                self,
                model_logprobs_model_data,
                model_logprobs_ref_data,
                ref_logprobs_ref_data,
                ref_logprobs_model_data,
                chosen_mask,
            ):
                loss, dpo_losses, xpo_losses = super()._compute_losses(
                    model_logprobs_model_data,
                    model_logprobs_ref_data,
                    ref_logprobs_ref_data,
                    ref_logprobs_model_data,
                    chosen_mask,
                )
                if self.state.global_step == 1:
                    loss = loss + poison
                return loss, dpo_losses, xpo_losses

        training_args = XPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=2,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            logging_steps=1,
            learning_rate=9e-1,
            report_to="none",
        )
        trainer = NonFiniteLossXPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            reward_funcs=self.reward_model,
            args=training_args,
            processing_class=self.tokenizer,
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
