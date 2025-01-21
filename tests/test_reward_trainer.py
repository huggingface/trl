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
from datasets import Dataset, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import RewardConfig, RewardTrainer, maybe_apply_chat_template
from trl.trainer.reward_trainer import _tokenize


if is_peft_available():
    from peft import LoraConfig, TaskType


class RewardTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def test_preprocessing_conversational(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
            training_args = RewardConfig(output_dir=tmp_dir, report_to="none")
            trainer = RewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            dummy_dataset = dummy_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
            dummy_dataset = dummy_dataset.map(_tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            self.assertDictEqual(trainer.train_dataset[:], dummy_dataset[:])

    def test_preprocessing_standard(self):
        # No chat template, so we load a fresh tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
            training_args = RewardConfig(output_dir=tmp_dir, report_to="none")
            trainer = RewardTrainer(
                model=self.model, args=training_args, processing_class=tokenizer, train_dataset=dummy_dataset
            )
            dummy_dataset = dummy_dataset.map(_tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
            self.assertDictEqual(trainer.train_dataset[:], dummy_dataset[:])

    def test_train_full(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
            training_args = RewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
            trainer = RewardTrainer(
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
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
            dummy_dataset = dummy_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
            dummy_dataset = dummy_dataset.map(_tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            training_args = RewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
            trainer = RewardTrainer(
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
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
            training_args = RewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
            trainer = RewardTrainer(
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

    @require_peft
    def test_train_lora_pretokenized(self):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
            dummy_dataset = dummy_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
            dummy_dataset = dummy_dataset.map(_tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer})
            training_args = RewardConfig(output_dir=tmp_dir, max_steps=3, report_to="none")
            trainer = RewardTrainer(
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

    def test_margin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset_dict = {
                "input_ids_chosen": [
                    torch.LongTensor([0, 1, 2]),
                ],
                "attention_mask_chosen": [
                    torch.LongTensor([1, 1, 1]),
                ],
                "input_ids_rejected": [
                    torch.LongTensor([0, 2]),
                ],
                "attention_mask_rejected": [
                    torch.LongTensor([1, 1]),
                ],
                "margin": [
                    torch.FloatTensor([1.0]),
                ],
            }
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
            training_args = RewardConfig(output_dir=tmp_dir, report_to="none")
            trainer = RewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )

            batch = [dummy_dataset[0]]
            batch = trainer.data_collator(batch)
            batch = {k: v.to(trainer.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)

            l_val = -torch.nn.functional.logsigmoid(
                outputs["rewards_chosen"] - outputs["rewards_rejected"] - batch["margin"]
            ).mean()

            self.assertLess(abs(loss - l_val), 1e-6)

    def test_tags(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_preference", split="train")
            training_args = RewardConfig(output_dir=tmp_dir, report_to="none")
            trainer = RewardTrainer(
                model=self.model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )
            self.assertEqual(trainer.model.model_tags, trainer._tag_names)
