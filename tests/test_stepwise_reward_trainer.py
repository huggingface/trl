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

from trl import StepwiseRewardConfig, StepwiseRewardTrainer, maybe_apply_chat_template
from trl.trainer import compute_accuracy
from trl.trainer.stepwise_reward_trainer import _tokenize
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


if is_peft_available():
    from peft import LoraConfig, TaskType


class StepwiseRewardTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure the tokenizer has a chat template
        if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def test_token_level_accuracy(self):
        dummy_eval_predictions = EvalPrediction(
            torch.FloatTensor([[[0.1, 0.9], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]),
            torch.LongTensor([[-100, 1], [-100, 1]]),
        )
        accuracy = compute_accuracy(dummy_eval_predictions)
        self.assertEqual(accuracy["accuracy"], 0.5)

    @parameterized.expand(["conversational_step", "standard_step"])
    def test_preprocessing(self, dataset_type):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", dataset_type, split="train")
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, report_to="none", max_length=512)
            trainer = StepwiseRewardTrainer(
                model=self.model,
                args=training_args,
                processing_class=self.tokenizer,
                train_dataset=dummy_dataset,
            )
            dummy_dataset = dummy_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
            dummy_dataset = dummy_dataset.map(
                _tokenize,
                batched=True,
                fn_kwargs={"tokenizer": self.tokenizer, "max_length": 512, "step_separator": "\n"},
            )
            self.assertDictEqual(trainer.train_dataset[:], dummy_dataset[:])

    def test_train_full(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_step", split="train")
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none", max_length=512)
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
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_step", split="train")
            dummy_dataset = dummy_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
            dummy_dataset = dummy_dataset.map(
                _tokenize,
                batched=True,
                fn_kwargs={"tokenizer": self.tokenizer, "max_length": 512, "post_step_separator": "\n"},
            )
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none", max_length=512)
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
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_step", split="train")
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
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_step", split="train")
            training_args = StepwiseRewardConfig(output_dir=tmp_dir, report_to="none", max_length=512)
            trainer = StepwiseRewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            self.assertEqual(trainer.model.model_tags, trainer._tag_names)
