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
from unittest.mock import MagicMock

import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForTokenClassification, AutoTokenizer, EvalPrediction, PreTrainedTokenizerBase
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import StepwiseRewardConfig, StepwiseRewardTrainer
from trl.trainer import compute_accuracy


if is_peft_available():
    from peft import LoraConfig, TaskType


class TestTokenizeRow(unittest.TestCase):
    def setUp(self):
        # Set up the mock tokenizer with specific behaviors
        self.tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = 2

        def mock_encode(text, add_special_tokens):
            token_map = {
                "Which number is larger, 9.8 or 9.11?": [465, 6766, 318, 298],
                "11 is greater than 8.": [4, 322, 12],
                "Hence, 9.11 > 9.8.": [4995, 11, 22],
                "\n": [1030],
                "\n\n": [1030, 1030],
            }

            return token_map[text]

        def mock_tokenizer_call(text, add_special_tokens):
            return {"input_ids": mock_encode(text, add_special_tokens)}

        self.tokenizer.encode.side_effect = mock_encode
        self.tokenizer.side_effect = mock_tokenizer_call

    def test_tokenize_row_no_truncation(self):
        # Define the input features
        features = {
            "prompt": "Which number is larger, 9.8 or 9.11?",
            "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
            "labels": [True, False],
        }

        # Call the method with no truncation
        result = StepwiseRewardTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_completion_length=None,
            train_on_last_step_only=False,
            is_eval=False,
        )

        self.assertEqual(
            result,
            {
                "input_ids": [0, 465, 6766, 318, 298, 4, 322, 12, 1030, 4995, 11, 22, 1030],
                "labels": [-100, -100, -100, -100, -100, -100, -100, -100, 1, -100, -100, -100, 0],
            },
        )

    def test_tokenize_row_train_on_last_step_only(self):
        # Define the input features
        features = {
            "prompt": "Which number is larger, 9.8 or 9.11?",
            "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
            "labels": [True, False],
        }

        result = StepwiseRewardTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_completion_length=None,
            train_on_last_step_only=True,
            is_eval=False,
        )

        self.assertEqual(
            result,
            {
                "input_ids": [0, 465, 6766, 318, 298, 4, 322, 12, 1030, 4995, 11, 22, 1030],
                "labels": [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0],
            },
        )

    def test_tokenize_row_completion_truncation(self):
        # Define the input features
        features = {
            "prompt": "Which number is larger, 9.8 or 9.11?",
            "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
            "labels": [True, False],
        }

        # Call the method with truncation on the completion
        result = StepwiseRewardTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_completion_length=6,
            train_on_last_step_only=False,
            is_eval=False,
        )

        self.assertEqual(
            result,
            {
                "input_ids": [0, 465, 6766, 318, 298, 4, 322, 12, 1030, 4995, 11],
                "labels": [-100, -100, -100, -100, -100, -100, -100, -100, 1, -100, -100],
            },
        )

    def test_tokenize_row_prompt_completion_truncation(self):
        # Define the input features
        features = {
            "prompt": "Which number is larger, 9.8 or 9.11?",
            "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
            "labels": [True, False],
        }

        # Call the method with truncation on the prompt and completion
        result = StepwiseRewardTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=9,
            max_completion_length=None,
            train_on_last_step_only=False,
            is_eval=False,
        )

        self.assertEqual(
            result,
            {
                "input_ids": [0, 465, 6766, 318, 298, 4, 322, 12, 1030],
                "labels": [-100, -100, -100, -100, -100, -100, -100, -100, 1],
            },
        )

    def test_tokenize_row_multi_token_separator(self):
        # Define the input features
        features = {
            "prompt": "Which number is larger, 9.8 or 9.11?",
            "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
            "labels": [True, False],
        }

        # Call the method using multiple tokens as step_separator
        result = StepwiseRewardTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n\n",
            max_length=None,
            max_completion_length=None,
            train_on_last_step_only=False,
            is_eval=False,
        )

        self.assertEqual(
            result,
            {
                "input_ids": [0, 465, 6766, 318, 298, 4, 322, 12, 1030, 1030, 4995, 11, 22, 1030, 1030],
                "labels": [-100, -100, -100, -100, -100, -100, -100, -100, -100, 1, -100, -100, -100, -100, 0],
            },
        )


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
    def test_train_full(self, train_on_last_step_only):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = StepwiseRewardConfig(
                output_dir=tmp_dir,
                max_steps=3,
                report_to="none",
                train_on_last_step_only=train_on_last_step_only,
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

    def test_train_full_pretokenized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")

            dummy_dataset = dummy_dataset.map(
                StepwiseRewardTrainer.tokenize_row,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "max_length": None,
                    "max_completion_length": None,
                    "step_separator": "\n",
                    "train_on_last_step_only": False,
                },
                remove_columns=dummy_dataset.features,
            )

            training_args = StepwiseRewardConfig(
                output_dir=tmp_dir,
                max_steps=3,
                report_to="none",
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
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
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
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, report_to="none")
            trainer = StepwiseRewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            self.assertEqual(trainer.model.model_tags, trainer._tag_names)
