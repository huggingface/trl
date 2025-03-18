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

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl import GKDConfig, GKDTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


class TestGKDTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(model_id)
        cls.generation_config = GenerationConfig(
            max_new_tokens=20,
            num_return_sequences=1,
            pad_token_id=cls.tokenizer.pad_token_id,
            eos_token_id=cls.tokenizer.eos_token_id,
        )

    def test_generate_on_policy_outputs_deterministic(self):
        prompts = ["Hello, how are you?", "What's the weather like today?"]
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)

        inputs = {
            "prompts": tokenized_prompts["input_ids"],
            "prompt_attention_mask": tokenized_prompts["attention_mask"],
        }

        # Set temperature to 0 for deterministic output
        deterministic_generation_config = GenerationConfig(
            max_new_tokens=30,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.0,
        )

        outputs = GKDTrainer.generate_on_policy_outputs(
            self.model, inputs, deterministic_generation_config, self.tokenizer.pad_token_id
        )

        new_input_ids, new_attention_mask, new_labels = outputs

        # Decode the generated outputs
        generated_texts = self.tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)

        # Check if the generated texts start with the original prompts
        for prompt, generated_text in zip(prompts, generated_texts):
            self.assertTrue(
                generated_text.startswith(prompt),
                f"Generated text '{generated_text}' does not start with prompt '{prompt}'",
            )

        # Run the generation twice and check if the outputs are identical
        outputs2 = GKDTrainer.generate_on_policy_outputs(
            self.model, inputs, deterministic_generation_config, self.tokenizer.pad_token_id
        )

        new_input_ids2, new_attention_mask2, new_labels2 = outputs2

        # Check if the two generations are identical
        self.assertTrue(torch.all(new_input_ids.eq(new_input_ids2)), "Deterministic generations are not identical")
        self.assertTrue(
            torch.all(new_attention_mask.eq(new_attention_mask2)),
            "Attention masks for deterministic generations are not identical",
        )
        self.assertTrue(
            torch.all(new_labels.eq(new_labels2)),
            "Labels for deterministic generations are not identical",
        )

    def test_generate_on_policy_outputs(self):
        prompts = ["Hello, how are you?", "What's the weather like today?"]
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)

        inputs = {
            "prompts": tokenized_prompts["input_ids"],
            "attention_mask": tokenized_prompts["attention_mask"],
        }

        outputs = GKDTrainer.generate_on_policy_outputs(
            self.model, inputs, self.generation_config, self.tokenizer.pad_token_id
        )

        # Check that outputs is a tuple of three tensors
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 3)

        new_input_ids, new_attention_mask, new_labels = outputs

        # Check shapes
        batch_size = len(prompts)
        self.assertEqual(new_input_ids.shape[0], batch_size)
        self.assertEqual(new_attention_mask.shape[0], batch_size)
        self.assertEqual(new_labels.shape[0], batch_size)

        # Check types
        self.assertIsInstance(new_input_ids, torch.Tensor)
        self.assertIsInstance(new_attention_mask, torch.Tensor)
        self.assertIsInstance(new_labels, torch.Tensor)

        # Check that new_input_ids and new_attention_mask have the same shape
        self.assertEqual(new_input_ids.shape, new_attention_mask.shape)
        self.assertEqual(new_labels.shape, new_attention_mask.shape)


class TestGeneralizedJSDLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 3
        self.vocab_size = 5
        self.student_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.teacher_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)

    def test_uniform_distribution(self):
        logits = torch.ones(1, 1, self.vocab_size)
        loss = GKDTrainer.generalized_jsd_loss(logits, logits)
        self.assertAlmostEqual(loss.item(), 0, places=5)

    def test_generalized_jsd_loss_edge_cases(self):
        # Setup
        student_logits = torch.log(torch.tensor([[0.1, 0.9]])).unsqueeze(0)
        teacher_logits = torch.log(torch.tensor([[0.9, 0.1]])).unsqueeze(0)

        # Case 1: beta = 1 (should be equivalent to KL(student || teacher))
        loss_beta_1 = GKDTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=1)
        expected_loss_beta_1 = F.kl_div(
            F.log_softmax(teacher_logits, dim=-1), F.softmax(student_logits, dim=-1), reduction="batchmean"
        )
        self.assertAlmostEqual(loss_beta_1.item(), expected_loss_beta_1.item(), places=5)

        # Case 2: beta = 0 (should be equivalent to KL(teacher || student))
        loss_beta_0 = GKDTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=0)
        expected_loss_beta_0 = F.kl_div(
            F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean"
        )
        self.assertAlmostEqual(loss_beta_0.item(), expected_loss_beta_0.item(), places=5)

    def test_output_shape(self):
        loss = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_beta_values(self):
        loss_beta_0 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0)
        loss_beta_1 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=1)
        self.assertNotEqual(loss_beta_0, loss_beta_1)

    def test_temperature_scaling(self):
        loss_temp_1 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=1)
        loss_temp_2 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=2)
        self.assertNotEqual(loss_temp_1, loss_temp_2)

    def test_reduction_methods(self):
        loss_batchmean = GKDTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, reduction="batchmean"
        )
        loss_sum = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="sum")
        loss_mean = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="mean")
        loss_none = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="none")

        self.assertEqual(loss_batchmean.shape, torch.Size([]))
        self.assertEqual(loss_sum.shape, torch.Size([]))
        self.assertEqual(loss_mean.shape, torch.Size([]))
        self.assertEqual(loss_none.shape, self.student_logits.shape)

    def test_symmetry(self):
        student_teacher = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.1)
        teacher_student = GKDTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.1)
        self.assertNotEqual(student_teacher, teacher_student)

        student_teacher = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.5)
        teacher_student = GKDTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.5)
        self.assertEqual(student_teacher, teacher_student)

    def test_zero_loss_for_identical_inputs(self):
        identical_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        loss = GKDTrainer.generalized_jsd_loss(identical_logits, identical_logits)
        self.assertAlmostEqual(loss.item(), 0, places=6)


class GKDTrainerTester(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure the tokenizer has a chat template
        if not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def test_gkd_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GKDConfig(
                output_dir=tmp_dir,
                dataloader_drop_last=True,
                eval_strategy="steps",
                max_steps=4,
                eval_steps=2,
                save_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = GKDTrainer(
                model=self.model_id,
                teacher_model=self.model_id,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
            )

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[(-1)]["train_loss"])
            self.assertIsNotNone(trainer.state.log_history[0]["eval_loss"])
            self.assertIn("model.safetensors", os.listdir(tmp_dir + "/checkpoint-2"))

    def test_generation_config_init(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GKDConfig(output_dir=tmp_dir)
            dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

            trainer = GKDTrainer(
                model=self.model_id,
                teacher_model=self.model_id,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
            )

            self.assertEqual(trainer.generation_config.pad_token_id, self.tokenizer.eos_token_id)
            self.assertEqual(trainer.generation_config.eos_token_id, self.model.generation_config.eos_token_id)
            self.assertEqual(trainer.generation_config.max_new_tokens, training_args.max_new_tokens)
            self.assertEqual(trainer.generation_config.temperature, training_args.temperature)
            self.assertEqual(trainer.generation_config.top_k, 0)
