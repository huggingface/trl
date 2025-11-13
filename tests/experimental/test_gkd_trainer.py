# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import pytest
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl.experimental.gkd import GKDConfig, GKDTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel


class TestGKDTrainerGenerateOnPolicy(TrlTestCase):
    @classmethod
    def setup_class(cls):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(model_id).to(cls.device)
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
            "prompts": tokenized_prompts["input_ids"].to(self.device),
            "prompt_attention_mask": tokenized_prompts["attention_mask"].to(self.device),
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
        for prompt, generated_text in zip(prompts, generated_texts, strict=True):
            assert generated_text.startswith(prompt), (
                f"Generated text '{generated_text}' does not start with prompt '{prompt}'"
            )

        # Run the generation twice and check if the outputs are identical
        outputs2 = GKDTrainer.generate_on_policy_outputs(
            self.model, inputs, deterministic_generation_config, self.tokenizer.pad_token_id
        )

        new_input_ids2, new_attention_mask2, new_labels2 = outputs2

        # Check if the two generations are identical
        assert torch.all(new_input_ids.eq(new_input_ids2)), "Deterministic generations are not identical"
        assert torch.all(new_attention_mask.eq(new_attention_mask2)), (
            "Attention masks for deterministic generations are not identical"
        )
        assert torch.all(new_labels.eq(new_labels2)), "Labels for deterministic generations are not identical"

    def test_generate_on_policy_outputs(self):
        prompts = ["Hello, how are you?", "What's the weather like today?"]
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)

        inputs = {
            "prompts": tokenized_prompts["input_ids"].to(self.device),
            "attention_mask": tokenized_prompts["attention_mask"].to(self.device),
        }

        outputs = GKDTrainer.generate_on_policy_outputs(
            self.model, inputs, self.generation_config, self.tokenizer.pad_token_id
        )

        # Check that outputs is a tuple of three tensors
        assert isinstance(outputs, tuple)
        assert len(outputs) == 3

        new_input_ids, new_attention_mask, new_labels = outputs

        # Check shapes
        batch_size = len(prompts)
        assert new_input_ids.shape[0] == batch_size
        assert new_attention_mask.shape[0] == batch_size
        assert new_labels.shape[0] == batch_size

        # Check types
        assert isinstance(new_input_ids, torch.Tensor)
        assert isinstance(new_attention_mask, torch.Tensor)
        assert isinstance(new_labels, torch.Tensor)

        # Check that new_input_ids and new_attention_mask have the same shape
        assert new_input_ids.shape == new_attention_mask.shape
        assert new_labels.shape == new_attention_mask.shape


class TestGeneralizedJSDLoss(TrlTestCase):
    def setup_method(self):
        self.batch_size = 2
        self.seq_length = 3
        self.vocab_size = 5
        self.student_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.teacher_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)

    def test_uniform_distribution(self):
        logits = torch.ones(1, 1, self.vocab_size)
        loss = GKDTrainer.generalized_jsd_loss(logits, logits)
        assert round(abs(loss.item() - 0), 5) == 0

    def test_generalized_jsd_loss_edge_cases(self):
        # Setup
        student_logits = torch.log(torch.tensor([[0.1, 0.9]])).unsqueeze(0)
        teacher_logits = torch.log(torch.tensor([[0.9, 0.1]])).unsqueeze(0)

        # Case 1: beta = 1 (should be equivalent to KL(student || teacher))
        loss_beta_1 = GKDTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=1)
        expected_loss_beta_1 = F.kl_div(
            F.log_softmax(teacher_logits, dim=-1), F.softmax(student_logits, dim=-1), reduction="batchmean"
        )
        assert round(abs(loss_beta_1.item() - expected_loss_beta_1.item()), 5) == 0

        # Case 2: beta = 0 (should be equivalent to KL(teacher || student))
        loss_beta_0 = GKDTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=0)
        expected_loss_beta_0 = F.kl_div(
            F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean"
        )
        assert round(abs(loss_beta_0.item() - expected_loss_beta_0.item()), 5) == 0

    def test_output_shape(self):
        loss = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits)
        assert torch.is_tensor(loss)
        assert loss.shape == torch.Size([])

    def test_beta_values(self):
        loss_beta_0 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0)
        loss_beta_1 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=1)
        assert loss_beta_0 != loss_beta_1

    def test_temperature_scaling(self):
        loss_temp_1 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=1)
        loss_temp_2 = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=2)
        assert loss_temp_1 != loss_temp_2

    def test_reduction_methods(self):
        loss_batchmean = GKDTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, reduction="batchmean"
        )
        loss_sum = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="sum")
        loss_mean = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="mean")
        loss_none = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="none")

        assert loss_batchmean.shape == torch.Size([])
        assert loss_sum.shape == torch.Size([])
        assert loss_mean.shape == torch.Size([])
        assert loss_none.shape == self.student_logits.shape

    def test_symmetry(self):
        student_teacher = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.1)
        teacher_student = GKDTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.1)
        assert student_teacher != teacher_student

        student_teacher = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.5)
        teacher_student = GKDTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.5)
        assert student_teacher == teacher_student

    def test_zero_loss_for_identical_inputs(self):
        identical_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        loss = GKDTrainer.generalized_jsd_loss(identical_logits, identical_logits)
        assert round(abs(loss.item() - 0), 6) == 0


class TestGKDTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_gkd_trainer(self):
        training_args = GKDConfig(
            output_dir=self.tmp_dir,
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

        assert trainer.state.log_history[(-1)]["train_loss"] is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None
        assert "model.safetensors" in os.listdir(self.tmp_dir + "/checkpoint-2")

    @require_liger_kernel
    @pytest.mark.xfail(reason="Computing the Liger loss spikes GPU memory usage, causing the test to run OOM.")
    def test_gkd_trainer_with_liger(self):
        training_args = GKDConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            use_liger_kernel=True,
        )
        dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            processing_class=self.tokenizer,
        )

        # Ensure liger fused JSD path is enabled; if not, skip (runtime may lack system libs)
        if not getattr(trainer, "use_liger_gkd_loss", False):
            pytest.skip("Liger fused JSD not enabled at runtime; skipping fused-loss assertion")

        trainer.train()

        # Check we logged a train loss
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_generation_config_init(self):
        training_args = GKDConfig(output_dir=self.tmp_dir)
        dummy_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dummy_dataset["train"],
            eval_dataset=dummy_dataset["test"],
            processing_class=self.tokenizer,
        )

        assert trainer.generation_config.pad_token_id == self.tokenizer.eos_token_id
        assert trainer.generation_config.eos_token_id == self.model.generation_config.eos_token_id
        assert trainer.generation_config.max_new_tokens == training_args.max_new_tokens
        assert trainer.generation_config.temperature == training_args.temperature
        assert trainer.generation_config.top_k == 0
