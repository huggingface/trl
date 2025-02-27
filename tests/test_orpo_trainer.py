# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from trl import ORPOConfig, ORPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


class ORPOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration"
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.t5_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    @parameterized.expand(
        [
            ("qwen", "standard_preference"),
            ("t5", "standard_implicit_prompt_preference"),
            ("qwen", "conversational_preference"),
            ("t5", "conversational_implicit_prompt_preference"),
        ]
    )
    def test_orpo_trainer(self, name, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = ORPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
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

            trainer = ORPOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.equal(param, new_param))

    @parameterized.expand(
        [
            ("standard_preference",),
            ("standard_implicit_prompt_preference",),
            ("conversational_preference",),
            ("conversational_implicit_prompt_preference",),
        ]
    )
    @require_peft
    def test_orpo_trainer_with_lora(self, config_name):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = ORPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

            trainer = ORPOTrainer(
                model=self.model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                peft_config=lora_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    if param.sum() != 0:  # ignore 0 biases
                        self.assertFalse(torch.equal(param, new_param))

    def test_compute_metrics(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token

        dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference")

        def dummy_compute_metrics(*args, **kwargs):
            return {"test": 0.0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = ORPOConfig(
                output_dir=tmp_dir,
                remove_unused_columns=False,
                per_device_train_batch_size=2,
                do_eval=True,
                eval_strategy="steps",
                eval_steps=1,
                per_device_eval_batch_size=2,
                report_to="none",
            )

            trainer = ORPOTrainer(
                model=model,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                compute_metrics=dummy_compute_metrics,
            )

            trainer.train()

            self.assertEqual(trainer.state.log_history[-2]["eval_test"], 0.0)
