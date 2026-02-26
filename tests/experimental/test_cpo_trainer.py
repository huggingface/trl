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
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from trl.experimental.cpo import CPOConfig, CPOTrainer

from ..testing_utils import TrlTestCase, require_peft


class TestCPOTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype="float32")
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    @pytest.mark.parametrize(
        "name, loss_type, config_name",
        [
            ("qwen", "sigmoid", "standard_preference"),
            ("t5", "hinge", "standard_implicit_prompt_preference"),
            ("qwen", "ipo", "conversational_preference"),
            ("qwen", "simpo", "standard_preference"),
            ("t5", "simpo", "standard_implicit_prompt_preference"),
            ("qwen", "hinge", "conversational_preference"),
        ],
    )
    def test_cpo_trainer(self, name, loss_type, config_name):
        training_args = CPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            loss_type=loss_type,
            cpo_alpha=1.0,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        if name == "qwen":
            model = self.model
            tokenizer = self.tokenizer
        elif name == "t5":
            model = self.t5_model
            tokenizer = self.t5_tokenizer
            training_args.is_encoder_decoder = True

        trainer = CPOTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:  # ignore 0 biases
                assert not torch.equal(param, new_param)

    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_preference",
            "standard_implicit_prompt_preference",
            "conversational_preference",
            "conversational_implicit_prompt_preference",
        ],
    )
    @require_peft
    def test_cpo_trainer_with_lora(self, config_name):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = CPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            cpo_alpha=1.0,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

        trainer = CPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            peft_config=lora_config,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the parameters have changed
        for n, param in previous_trainable_params.items():
            if "lora" in n:
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    assert not torch.equal(param, new_param)

    def test_compute_metrics(self):
        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        training_args = CPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            remove_unused_columns=False,
            do_eval=True,
            eval_strategy="steps",
            eval_steps=1,
            per_device_eval_batch_size=2,
            report_to="none",
        )

        trainer = CPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            compute_metrics=dummy_compute_metrics,
        )

        trainer.train()

        assert trainer.state.log_history[-2]["eval_test"] == 0.0

    def test_alphapo_trainer(self):
        training_args = CPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_steps=3,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            learning_rate=9e-1,
            eval_strategy="steps",
            beta=0.1,
            loss_type="alphapo",
            alpha=0.5,
            simpo_gamma=0.5,
            report_to="none",
        )

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        trainer = CPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.equal(param, new_param)
