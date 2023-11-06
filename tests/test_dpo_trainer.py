# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from datasets import Dataset
from parameterized import parameterized
from pytest import mark
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments

from trl import DPOTrainer

from .testing_utils import require_no_wandb, require_peft


class DPOTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {
            "gpt2": {
                "model_id": "trl-internal-testing/dummy-GPT2-correct-vocab",
                "model_type": AutoModelForCausalLM,
                "tokenizer_type": AutoTokenizer,
            },
            "t5": {
                "model_id": "trl-internal-testing/tiny-T5ForConditionalGeneration-correct-vocab",
                "model_type": AutoModelForSeq2SeqLM,
                "tokenizer_type": AutoTokenizer,
            },
            # add more models here if needed
        }
        for key, model_info in cls.models.items():
            model_info["model"] = model_info["model_type"].from_pretrained(model_info["model_id"])
            model_info["ref_model"] = model_info["model_type"].from_pretrained(model_info["model_id"])
            model_info["tokenizer"] = model_info["tokenizer_type"].from_pretrained(model_info["model_id"])
        cls.models["gpt2"]["tokenizer"].pad_token = cls.models["gpt2"]["tokenizer"].eos_token

    def _init_dummy_dataset(self):
        # fmt: off
        dummy_dataset_dict = {
            "prompt": [
                "hello\n",
                "how are you\n",
                "What is your name?\n",
                "What is your name?\n",
                "Which is the best programming language?\n",
                "Which is the best programming language?\n",
                "Which is the best programming language?\n",
            ],
            "chosen": [
                "hi nice to meet you",
                "I am fine",
                "My name is Mary",
                "My name is Mary",
                "Python",
                "Python",
                "Python",
            ],
            "rejected": [
                "leave me alone",
                "I am not fine",
                "Whats it to you?",
                "I dont have a name",
                "Javascript",
                "C++",
                "Java",
            ],
        }
        # fmt: on
        return Dataset.from_dict(dummy_dataset_dict)

    def _get_models_by_name(self, name):
        return self.models[name]["model"], self.models[name]["ref_model"], self.models[name]["tokenizer"]

    @parameterized.expand([["gpt2", "sigmoid", True], ["t5", "hinge", False]])
    def test_dpo_trainer(self, name, loss_type, precompute_ref_log_probs):
        model, ref_model, tokenizer = self._get_models_by_name(name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                beta=0.1,
                loss_type=loss_type,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                precompute_ref_log_probs=precompute_ref_log_probs,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    @parameterized.expand([["gpt2", True], ["t5", True]])
    def test_dpo_trainer_without_providing_ref_model(self, name, precompute_ref_log_probs):
        model, _, tokenizer = self._get_models_by_name(name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                beta=0.1,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                precompute_ref_log_probs=precompute_ref_log_probs,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.equal(param, new_param))

    @require_peft
    @mark.peft_test
    @parameterized.expand([["gpt2", True], ["t5", True]])
    def test_dpo_trainer_without_providing_ref_model_with_lora(self, name, precompute_ref_log_probs):
        model, _, tokenizer = self._get_models_by_name(name)
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                beta=0.1,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
                precompute_ref_log_probs=precompute_ref_log_probs,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    # check the params have changed - ignore 0 biases
                    if param.sum() != 0:
                        self.assertFalse(torch.equal(param, new_param))

    @parameterized.expand([["gpt2", False], ["t5", True]])
    @require_no_wandb
    def test_dpo_trainer_generate_during_eval_no_wandb(self, name, precompute_ref_log_probs):
        model, _, tokenizer = self._get_models_by_name(name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            dummy_dataset = self._init_dummy_dataset()

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve.",
            ):
                DPOTrainer(
                    model=model,
                    ref_model=None,
                    beta=0.1,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=dummy_dataset,
                    eval_dataset=dummy_dataset,
                    generate_during_eval=True,
                    precompute_ref_log_probs=precompute_ref_log_probs,
                )

    @require_peft
    @mark.peft_test
    def test_dpo_lora_save(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model_peft = get_peft_model(model, lora_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model_peft,
                ref_model=None,
                beta=0.1,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

            # assert that the model is loaded without giving OSError
            try:
                AutoModelForCausalLM.from_pretrained(tmp_dir)
            except OSError:
                self.fail("Loading the saved peft adapter failed")
