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
from unittest.mock import MagicMock

import torch
from datasets import Dataset, load_dataset
from parameterized import parameterized
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerBase
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import PRMConfig, PRMTrainer


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
        result = PRMTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_prompt_length=None,
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

        result = PRMTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_prompt_length=None,
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

    def test_tokenize_row_prompt_truncation(self):
        # Define the input features
        features = {
            "prompt": "Which number is larger, 9.8 or 9.11?",
            "completions": ["11 is greater than 8.", "Hence, 9.11 > 9.8."],
            "labels": [True, False],
        }

        # Call the method with truncation on the completion
        result = PRMTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_prompt_length=3,
            max_completion_length=None,
            train_on_last_step_only=False,
            is_eval=False,
        )

        self.assertEqual(
            result,
            {
                "input_ids": [6766, 318, 298, 4, 322, 12, 1030, 4995, 11, 22, 1030],
                "labels": [-100, -100, -100, -100, -100, -100, 1, -100, -100, -100, 0],
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
        result = PRMTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=None,
            max_prompt_length=None,
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
        result = PRMTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n",
            max_length=9,
            max_prompt_length=None,
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
        result = PRMTrainer.tokenize_row(
            features=features,
            tokenizer=self.tokenizer,
            step_separator="\n\n",
            max_length=None,
            max_prompt_length=None,
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


class PRMTrainerTester(unittest.TestCase):
    def setUp(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @parameterized.expand([True, False])
    def test_train_full(self, train_on_last_step_only):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = PRMConfig(
                output_dir=tmp_dir,
                report_to="none",
                train_on_last_step_only=train_on_last_step_only,
            )
            trainer = PRMTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))

    def test_train_full_pretokenized(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = Dataset.from_dict(
                {
                    "labels": [
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, 0, -100, -100, 1],
                        [-100, -100, -100, -100, -100, -100, -100, -100, 0, -100, -100, 1, -100, -100, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0, -100, -100, 1],
                        [-100, -100, -100, -100, -100, -100, -100, 1, -100, -100, 1],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, 1, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, 1],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, 1, -100, -100, -100, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, -100, -100, 0, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, 0, -100, -100, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 1],
                        [-100, -100, -100, -100, -100, -100, 0],
                        [-100, -100, -100, -100, -100, -100, -100, -100, 1],
                        [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 0],
                    ],
                    "input_ids": [
                        [46518, 374, 2664, 1091, 11, 1077, 752, 1744, 1112, 198, 27261, 13, 198],
                        [98923, 374, 2664, 1091, 11, 315, 3308, 11, 198, 17995, 13, 198, 1576, 31273, 12850, 13, 198],
                        [16374, 374, 2664, 1091, 1112, 1077, 594, 2506, 432, 6770, 11, 198, 6351, 13, 198],
                        [31137, 374, 2664, 1091, 979, 4362, 11, 198, 16965, 13, 198],
                        [31019, 374, 2664, 1091, 304, 3793, 315, 5944, 11, 198, 24034, 13, 198],
                        [98491, 374, 2664, 1091, 1112, 5310, 369, 91494, 13, 198],
                        [4418, 2897, 14579, 5310, 979, 3800, 1349, 432, 13, 198],
                        [20366, 5048, 7629, 944, 3281, 3322, 11, 7241, 1112, 198, 807, 1795, 279, 5601, 13, 198],
                        [15802, 14976, 487, 33327, 1045, 31787, 63443, 11, 198, 52400, 13, 198],
                        [13877, 1265, 2581, 1494, 49394, 11, 198, 7241, 20975, 91681, 13, 198],
                        [641, 279, 3579, 315, 71768, 11, 25066, 279, 61361, 311, 7942, 13, 198],
                        [7039, 374, 2664, 1091, 2937, 13, 198],
                        [26155, 374, 3545, 2664, 1091, 34933, 26537, 13, 198],
                        [2679, 279, 8129, 374, 4135, 311, 10339, 11, 432, 2578, 387, 264, 1661, 2884, 13, 198],
                    ],
                }
            )

            training_args = PRMConfig(output_dir=tmp_dir, report_to="none")
            trainer = PRMTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # ignore 0 biases
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
            training_args = PRMConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
            trainer = PRMTrainer(
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

            # Check that the parameters have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

            # Check that the non trainable parameters have not changed
            for n, param in previous_non_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertTrue(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

    def test_tags(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_stepwise_supervision", split="train")
            training_args = PRMConfig(output_dir=tmp_dir, report_to="none")
            trainer = PRMTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            self.assertEqual(trainer.model.model_tags, trainer._tag_names)
