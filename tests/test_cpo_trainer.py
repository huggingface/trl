# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest

import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.testing_utils import require_peft

from trl import CPOConfig, CPOTrainer


class CPOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration-correct-vocab"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

    @parameterized.expand(
        [
            ["gpt2", "sigmoid", "standard_preference"],
            ["t5", "hinge", "standard_implicit_prompt_preference"],
            ["gpt2", "ipo", "conversational_preference"],
            ["t5", "ipo", "conversational_implicit_prompt_preference"],
            ["gpt2", "simpo", "standard_preference"],
            ["t5", "simpo", "standard_implicit_prompt_preference"],
            ["gpt2", "hinge", "conversational_preference"],
        ]
    )
    def test_cpo_trainer(self, name, loss_type, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CPOConfig(
                output_dir=tmp_dir,
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

            if name == "gpt2":
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

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.equal(param, new_param)

    @require_peft
    @parameterized.expand(
        [
            ("standard_preference",),
            ("standard_implicit_prompt_preference",),
            ("conversational_preference",),
            ("conversational_implicit_prompt_preference",),
        ]
    )
    def test_cpo_trainer_with_lora(self, config_name):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CPOConfig(
                output_dir=tmp_dir,
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

            # check the params have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    # check the params have changed - ignore 0 biases
                    if param.sum() != 0:
                        assert not torch.equal(param, new_param)
