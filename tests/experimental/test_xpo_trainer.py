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

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
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

    def test_logps_metrics_contain_only_policy_logps(self):
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(gather_for_metrics=lambda tensor: tensor),
            stats=defaultdict(list),
            beta=0.1,
            alpha=0.2,
            processing_class=SimpleNamespace(eos_token_id=99),
        )
        model_data = {"input_ids": torch.tensor([[0, 1, 99], [0, 1, 2]])}
        ref_data = {"input_ids": torch.tensor([[0, 3, 4], [0, 5, 99]])}
        model_logprobs_model_data = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
        model_logprobs_ref_data = torch.tensor([[4.0, 5.0], [40.0, 50.0]])
        ref_logprobs_ref_data = torch.tensor([[0.4, 0.5], [4.0, 5.0]])
        ref_logprobs_model_data = torch.tensor([[0.1, 0.2], [1.0, 2.0]])

        XPOTrainer._log_statistics(
            trainer,
            model_data,
            ref_data,
            model_logprobs_model_data,
            model_logprobs_ref_data,
            ref_logprobs_ref_data,
            ref_logprobs_model_data,
            torch.tensor([True, False]),
            torch.tensor([0.1, 0.2]),
            torch.tensor([0.3, 0.4]),
            1,
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.5, 1.5]),
        )

        assert trainer.stats["logps/chosen"] == [pytest.approx(46.5)]
        assert trainer.stats["logps/rejected"] == [pytest.approx(19.5)]

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
