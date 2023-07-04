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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, TrainingArguments

from trl import RewardTrainer
from trl.trainer import compute_accuracy

from .testing_utils import require_peft


class RewardTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForSequenceClassification.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def test_accuracy_metrics(self):
        dummy_eval_predictions = EvalPrediction(torch.FloatTensor([[0.1, 0.9], [0.9, 0.1]]), torch.LongTensor([0, 0]))
        accuracy = compute_accuracy(dummy_eval_predictions)
        self.assertEqual(accuracy["accuracy"], 0.5)

    def test_reward_trainer(self):
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

            # fmt: off
            dummy_dataset_dict = {
                "input_ids_chosen": [
                    torch.LongTensor([0, 1, 2,]),
                    torch.LongTensor([1, 2]),
                    torch.LongTensor([0, 1, 2,]),
                    torch.LongTensor([1, 2]),
                ],
                "attention_mask_chosen": [
                    torch.LongTensor([1, 1, 1]),
                    torch.LongTensor([1, 0]),
                    torch.LongTensor([1, 1, 1]),
                    torch.LongTensor([1, 0]),
                ],
                "input_ids_rejected": [
                    torch.LongTensor([0, 2,]),
                    torch.LongTensor([1, 2, 0]),
                    torch.LongTensor([0, 2,]),
                    torch.LongTensor([1, 2, 0]),
                ],
                "attention_mask_rejected": [
                    torch.LongTensor([1, 1]),
                    torch.LongTensor([1, 1, 0]),
                    torch.LongTensor([1, 1]),
                    torch.LongTensor([1, 1, 1]),
                ],
            }
            # fmt: on
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

            trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
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

            preds = trainer.predict(dummy_dataset)
            self.assertEqual(preds.predictions.shape, (4, 2))

    @require_peft
    def test_reward_trainer_peft(self):
        import peft
        from peft import LoraConfig, TaskType

        peft_version = peft.__version__

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=6,
                remove_unused_columns=False,
                gradient_accumulation_steps=2,
                learning_rate=9e-1,
                evaluation_strategy="steps",
            )

            # fmt: off
            dummy_dataset_dict = {
                "input_ids_chosen": [
                    torch.LongTensor([0, 1, 2,]),
                    torch.LongTensor([1, 2]),
                    torch.LongTensor([0, 1, 2,]),
                    torch.LongTensor([1, 2]),
                ],
                "attention_mask_chosen": [
                    torch.LongTensor([1, 1, 1]),
                    torch.LongTensor([1, 0]),
                    torch.LongTensor([1, 1, 1]),
                    torch.LongTensor([1, 0]),
                ],
                "input_ids_rejected": [
                    torch.LongTensor([0, 2,]),
                    torch.LongTensor([1, 2, 0]),
                    torch.LongTensor([0, 2,]),
                    torch.LongTensor([1, 2, 0]),
                ],
                "attention_mask_rejected": [
                    torch.LongTensor([1, 1]),
                    torch.LongTensor([1, 1, 0]),
                    torch.LongTensor([1, 1]),
                    torch.LongTensor([1, 1, 1]),
                ],
            }
            # fmt: on
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

            trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=peft_config,
            )
            previous_trainable_params = {}
            previous_non_trainable_params = {}

            # due to a change in the way the modules to save are dealt in PEFT.
            trainable_params_name = ["lora", "score"] if peft_version < "0.3.0" else ["lora", "modules_to_save"]

            # check gradients are not None
            for n, param in trainer.model.named_parameters():
                if any([t in n for t in trainable_params_name]):
                    previous_trainable_params[n] = param.clone()
                else:
                    previous_non_trainable_params[n] = param.clone()

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

            # check the non trainable params have not changed
            for n, param in previous_non_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertTrue(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

            preds = trainer.predict(dummy_dataset)
            self.assertEqual(preds.predictions.shape, (4, 2))

    def test_reward_trainer_assert_value_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                remove_unused_columns=False,
            )

            dummy_dataset_dict = {
                # fmt: off
                "input_ids_b": [
                    torch.LongTensor([0, 1, 2,]),
                    torch.LongTensor([1, 2]),
                    torch.LongTensor([0, 1, 2,]),
                    torch.LongTensor([1, 2]),
                ],
                "attention_mask_c": [
                    torch.LongTensor([1, 1, 1]),
                    torch.LongTensor([1, 0]),
                    torch.LongTensor([1, 1, 1]),
                    torch.LongTensor([1, 0]),
                ],
                "input_ids_f": [
                    torch.LongTensor([0, 2,]),
                    torch.LongTensor([1, 2, 0]),
                    torch.LongTensor([0, 2,]),
                    torch.LongTensor([1, 2, 0]),
                ],
                "attention_mask_g": [
                    torch.LongTensor([1, 1]),
                    torch.LongTensor([1, 1, 0]),
                    torch.LongTensor([1, 1]),
                    torch.LongTensor([1, 1, 1]),
                ],
                # fmt: on
            }
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

            trainer = RewardTrainer(
                model=self.model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
            )

            with self.assertRaises(ValueError):
                trainer.train()

            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                remove_unused_columns=True,
            )

            with self.assertWarns(UserWarning):
                trainer = RewardTrainer(
                    model=self.model,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset,
                )
