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
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl import RewardConfig, RewardTrainer, maybe_apply_chat_template
from trl.trainer import compute_accuracy
from trl.trainer.reward_trainer import _tokenize


if is_peft_available():
    from peft import LoraConfig, TaskType


class RewardTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def test_accuracy_metrics(self):
        dummy_eval_predictions = EvalPrediction(torch.FloatTensor([[0.1, 0.9], [0.9, 0.1]]), torch.LongTensor([0, 0]))
        accuracy = compute_accuracy(dummy_eval_predictions)
        self.assertEqual(accuracy["accuracy"], 0.5)

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
            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
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
            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
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

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

            # check the non trainable params have not changed
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

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.allclose(param, new_param, atol=1e-12, rtol=1e-12))

            # check the non trainable params have not changed
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

    def test_train_with_feedback(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a dummy dataset with feedback
            dummy_dataset_dict = {
                "prompt": ["What is 2+2?", "How are you?"],
                "chosen": ["The answer is 4.", "I'm doing great!"],
                "rejected": ["The answer is 5.", "Not so good."],
                "chosen_feedback": [["Good explanation", "Clear answer"], ["Nice response", "Polite"]],
                "rejected_feedback": [["Wrong answer", "Incorrect"], ["Too negative", "Unhelpful"]],
            }
            dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

            # Configure training arguments with feedback
            training_args = RewardConfig(
                output_dir=tmp_dir, max_steps=3, report_to="none", feedback_method="teacher", lm_weight=1.0
            )

            # Initialize trainer with feedback
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            trainer = RewardTrainer(
                model=model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
            )

            # Store initial parameters
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            # Train the model
            trainer.train()

            # Verify training occurred
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that parameters changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:  # Ignore zero-initialized parameters
                    self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))

            # Test compute_loss with feedback
            batch = next(iter(trainer.get_train_dataloader()))
            loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)

            # Verify outputs contain both reward and language model logits
            self.assertIn("rewards_chosen", outputs)
            self.assertIn("rewards_rejected", outputs)
            self.assertIsNotNone(outputs["chosen_logits"])
            self.assertIsNotNone(outputs["rejected_logits"])

            # Verify loss is computed correctly (includes both reward and LM components)
            reward_loss = -torch.nn.functional.logsigmoid(
                outputs["rewards_chosen"] - outputs["rewards_rejected"]
            ).mean()
            self.assertGreater(loss, reward_loss)  # Total loss should be larger due to LM component


def test_train_with_feedback_pretokenized(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy dataset with feedback and pre-tokenize it
        dummy_dataset_dict = {
            "prompt": ["What is 2+2?", "How are you?"],
            "chosen": ["The answer is 4.", "I'm doing great!"],
            "rejected": ["The answer is 5.", "Not so good."],
            "chosen_feedback": [["Good explanation", "Clear answer"], ["Nice response", "Polite"]],
            "rejected_feedback": [["Wrong answer", "Incorrect"], ["Too negative", "Unhelpful"]],
        }
        dummy_dataset = Dataset.from_dict(dummy_dataset_dict)

        # Pre-tokenize the dataset
        dummy_dataset = dummy_dataset.map(maybe_apply_chat_template, fn_kwargs={"tokenizer": self.tokenizer})
        dummy_dataset = dummy_dataset.map(
            _tokenize, batched=True, fn_kwargs={"tokenizer": self.tokenizer, "feedback_method": "teacher"}
        )
        # Configure training arguments
        training_args = RewardConfig(
            output_dir=tmp_dir, max_steps=3, report_to="none", feedback_method="teacher", lm_weight=1.0
        )

        # Initialize and train
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        trainer = RewardTrainer(
            model=model, args=training_args, processing_class=self.tokenizer, train_dataset=dummy_dataset
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}
        trainer.train()

        # Verify training occurred
        self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

        # Check parameters changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                self.assertFalse(torch.allclose(param, new_param, rtol=1e-12, atol=1e-12))
