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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft

from trl import CGPOConfig, CGPOTrainer
from trl.trainer.cgpo_trainer import MixtureOfConstraintJudges
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


class CGPOTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=1)
        # to replace one the Mixture of containtPR is merged
        self.moj = MixtureOfConstraintJudges()

        # Ensure the tokenizer has a chat template
        if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    @parameterized.expand(["crraft", "crpg", "codpo"])
    def test_cgpo_trainer(self, rlhf_optimizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer=rlhf_optimizer,
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                mixture_of_judges=self.moj,
                args=training_args,
                tokenizer=self.tokenizer,
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
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @parameterized.expand(["crraft", "crpg", "codpo"])
    def test_cgpo_trainer_no_satisfied_constraints(self, rlhf_optimizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            moj = MixtureOfConstraintJudges(method="all_violated")
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer=rlhf_optimizer,
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                mixture_of_judges=moj,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have not changed if no constraints are satisfied
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:
                    assert torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @parameterized.expand(["crraft", "crpg", "codpo"])
    def test_cgpo_trainer_all_satisfied_constraints(self, rlhf_optimizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            moj = MixtureOfConstraintJudges(method="all_satisfied")
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer=rlhf_optimizer,
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                mixture_of_judges=moj,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if param.sum() != 0:
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_cgpo_trainer_no_moj(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer="crraft",
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`mixture_of_judges` must be provided.",
            ):
                CGPOTrainer(
                    model=self.model,
                    ref_model=self.ref_model,
                    reward_model=self.reward_model,
                    mixture_of_judges=None,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

    def test_cgpo_trainer_no_reward_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer="crraft",
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`reward_model` must be provided.",
            ):
                CGPOTrainer(
                    model=self.model,
                    ref_model=self.ref_model,
                    reward_model=None,
                    mixture_of_judges=self.moj,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

    def test_cgpo_trainer_wrong_rlhf_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrong_rlhf_optimizer = "crraftss"
            with self.assertRaisesRegex(
                ValueError,
                expected_regex=f"Invalid value for rlhf_optimizer: {wrong_rlhf_optimizer}. Must be one of 'crraft', 'codpo', or 'crpg'.",
            ):
                CGPOConfig(
                    output_dir=tmp_dir,
                    rlhf_optimizer=wrong_rlhf_optimizer,
                    k=4,
                    kl_threshold=5.0,
                    temperature=0.9,
                    max_new_tokens=4,
                    per_device_train_batch_size=4,
                    max_steps=3,
                    remove_unused_columns=False,
                    gradient_accumulation_steps=1,
                    learning_rate=9e-1,
                    eval_strategy="steps",
                    report_to="none",
                )

    @parameterized.expand(["crraft", "crpg", "codpo"])
    def test_cgpo_trainer_with_missing_eos_penalty(self, rlhf_optimizer):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer=rlhf_optimizer,
                k=4,
                missing_eos_penalty=1.0,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                mixture_of_judges=self.moj,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_cgpo_trainer_without_providing_ref_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer="crraft",
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                reward_model=self.reward_model,
                mixture_of_judges=self.moj,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_cgpo_trainer_with_ref_model_is_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer="crraft",
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            with self.assertRaises(ValueError):
                CGPOTrainer(
                    model=self.model,
                    ref_model=self.model,
                    reward_model=self.reward_model,
                    mixture_of_judges=self.moj,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

    @require_peft
    def test_cgpo_trainer_without_providing_ref_model_with_lora(self):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer="crraft",
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                reward_model=self.reward_model,
                mixture_of_judges=self.moj,
                args=training_args,
                tokenizer=self.tokenizer,
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
                        assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_cgpo_trainer_padding_token_id_is_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer="crraft",
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = None

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.",
            ):
                trainer = CGPOTrainer(
                    model=self.model,
                    ref_model=self.ref_model,
                    reward_model=self.reward_model,
                    mixture_of_judges=self.moj,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=dummy_dataset["train"],
                    eval_dataset=dummy_dataset["test"],
                )

                trainer.train()

    @parameterized.expand(["crraft", "crpg", "codpo"])
    def test_cgpo_tags(self, optimizer_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = CGPOConfig(
                output_dir=tmp_dir,
                rlhf_optimizer=optimizer_name,
                k=4,
                kl_threshold=5.0,
                temperature=0.9,
                max_new_tokens=4,
                per_device_train_batch_size=4,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = CGPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                mixture_of_judges=self.moj,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
            )

            for tag in ["cgpo", "trl", optimizer_name]:
                self.assertIn(tag, trainer.model.model_tags)
