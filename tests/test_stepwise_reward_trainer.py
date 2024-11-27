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
from transformers import AutoModelForTokenClassification, AutoTokenizer, EvalPrediction
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import StepwiseRewardConfig, StepwiseRewardTrainer
from trl.trainer import compute_accuracy


if is_peft_available():
    from peft import LoraConfig, TaskType


class StepwiseRewardTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_token_level_accuracy(self):
        dummy_eval_predictions = EvalPrediction(
            torch.FloatTensor([[[0.1, 0.9], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]),
            torch.LongTensor([[-100, 1], [-100, 1]]),
        )
        accuracy = compute_accuracy(dummy_eval_predictions)
        self.assertEqual(accuracy["accuracy"], 0.5)

    @parameterized.expand([True, False])
    def test_preprocessing(self, train_on_last_step_only):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = StepwiseRewardConfig(
                output_dir=tmp_dir, report_to="none", max_length=512, train_on_last_step=train_on_last_step
            )
            trainer = StepwiseRewardTrainer(
                model=self.model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
            )
            dummy_dataset = dummy_dataset.map(
                trainer.tokenize_row,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "step_separator": "\n",
                    "max_completion_length": 512,
                    "train_on_last_step_only": train_on_last_step_only,
                },
                remove_columns=dummy_dataset.features,
            )
            self.assertDictEqual(trainer.train_dataset[:], dummy_dataset[:])

    @parameterized.expand([True, False])
    def test_train_full(self, train_on_last_step):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = StepwiseRewardConfig(
                output_dir=tmp_dir,
                max_steps=3,
                report_to="none",
                max_length=512,
                train_on_last_step=train_on_last_step,
            )
            trainer = StepwiseRewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))

    @parameterized.expand([True, False])
    def test_train_full_pretokenized(self, train_on_last_step):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            dummy_dataset = dummy_dataset.map(
                _tokenize_fn,
                batched=True,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "max_length": 512,
                    "step_separator": "\n",
                    "train_on_last_step": train_on_last_step,
                },
                remove_columns=dummy_dataset.features,
            )

            training_args = StepwiseRewardConfig(
                output_dir=tmp_dir,
                max_steps=3,
                report_to="none",
                max_length=512,
                train_on_last_step=train_on_last_step,
            )
            trainer = StepwiseRewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))

    @require_peft
    def test_train_lora(self):
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none", max_length=512)
            trainer = StepwiseRewardTrainer(
                model=self.model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
                peft_config=peft_config,
            )
            previous_trainable_params = {}
            previous_non_trainable_params = {}

            # due to a change in the way the modules to save are dealt in PEFT.
            trainable_params_name = ["lora", "modules_to_save"]

            # check gradients are not None
            for n, param in trainer.model.named_parameters():
                if any(t in n for t in trainable_params_name):
                    previous_trainable_params[n] = param.clone()
                else:
                    previous_non_trainable_params[n] = param.clone()

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

            # check the non trainable params have not changed
            for n, param in previous_non_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertTrue(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

    def test_tags(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, report_to="none", max_length=512)
            trainer = StepwiseRewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            self.assertEqual(trainer.model.model_tags, trainer._tag_names)
