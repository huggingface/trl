# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.trainer.sft_trainer import SFTTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel


class TestGKDTrainerGenerateOnPolicy(TrlTestCase):
    @classmethod
    def setup_class(cls):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="float32").to(cls.device)
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
            do_sample=False,
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

    def test_accumulated_loss_matches_combined_batch_with_global_denominator(self):
        torch.manual_seed(0)
        student_logits_1 = torch.randn(1, 4, self.vocab_size)
        teacher_logits_1 = torch.randn(1, 4, self.vocab_size)
        labels_1 = torch.tensor([[0, -100, -100, -100]])

        student_logits_2 = torch.randn(1, 4, self.vocab_size)
        teacher_logits_2 = torch.randn(1, 4, self.vocab_size)
        labels_2 = torch.tensor([[0, 1, 2, -100]])

        total_tokens = labels_1.ne(-100).sum() + labels_2.ne(-100).sum()
        accumulated = GKDTrainer.generalized_jsd_loss(
            student_logits_1, teacher_logits_1, labels=labels_1, num_items_in_batch=total_tokens
        ) + GKDTrainer.generalized_jsd_loss(
            student_logits_2, teacher_logits_2, labels=labels_2, num_items_in_batch=total_tokens
        )
        combined = GKDTrainer.generalized_jsd_loss(
            torch.cat([student_logits_1, student_logits_2]),
            torch.cat([teacher_logits_1, teacher_logits_2]),
            labels=torch.cat([labels_1, labels_2]),
        )
        old_style = GKDTrainer.generalized_jsd_loss(
            student_logits_1, teacher_logits_1, labels=labels_1
        ) + GKDTrainer.generalized_jsd_loss(student_logits_2, teacher_logits_2, labels=labels_2)

        assert torch.allclose(accumulated, combined)
        assert not torch.allclose(old_style, combined)

    def test_batchmean_with_global_denominator_handles_zero_tokens(self):
        student_logits = torch.randn(1, 3, self.vocab_size)
        teacher_logits = torch.randn(1, 3, self.vocab_size)
        labels = torch.full((1, 3), -100)

        loss = GKDTrainer.generalized_jsd_loss(
            student_logits, teacher_logits, labels=labels, num_items_in_batch=torch.tensor(0)
        )

        assert loss == 0

    def test_generate_on_policy_outputs_masks_prompt_labels(self):
        class DummyModel:
            def generate(self, **kwargs):
                return SimpleNamespace(sequences=torch.tensor([[0, 5, 6, 7, 0], [0, 8, 9, 10, 11]]))

        inputs = {
            "prompts": torch.tensor([[0, 5, 6], [0, 8, 9]]),
            "prompt_attention_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
        }

        _, new_attention_mask, new_labels = GKDTrainer.generate_on_policy_outputs(
            DummyModel(), inputs, GenerationConfig(), pad_token_id=0
        )

        assert torch.all(new_labels[:, : inputs["prompts"].shape[1]] == -100)
        assert new_labels[0, 3] == 7
        assert new_labels[0, 4] == -100
        assert new_attention_mask[0, 0] == 0
        assert new_attention_mask[0, 4] == 0


class TestGKDTrainerLossAccounting(TrlTestCase):
    def test_compute_loss_forwards_num_items_in_batch(self, monkeypatch):
        class DummyModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits

            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(logits=self.logits)

        captured_kwargs = {}

        def fake_generalized_jsd_loss(**kwargs):
            captured_kwargs.update(kwargs)
            return torch.tensor(1.0)

        monkeypatch.setattr(GKDTrainer, "generalized_jsd_loss", staticmethod(fake_generalized_jsd_loss))

        vocab_size = 11
        trainer = GKDTrainer.__new__(GKDTrainer)
        trainer.use_liger_gkd_loss = False
        trainer.teacher_model = DummyModel(torch.randn(1, 5, vocab_size))
        trainer.beta = 0.5
        trainer.args = SimpleNamespace(average_tokens_across_devices=False, n_gpu=0)
        trainer.model_accepts_loss_kwargs = True
        trainer.accelerator = SimpleNamespace(num_processes=1)

        denominator = torch.tensor(3)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.ones(1, 5),
            "prompts": torch.tensor([[1, 2]]),
            "labels": torch.tensor([[-100, -100, 3, 4, 5]]),
        }

        loss = trainer.compute_loss(
            DummyModel(torch.randn(1, 5, vocab_size)),
            inputs,
            num_items_in_batch=denominator,
        )

        assert loss == 1.0
        assert captured_kwargs["num_items_in_batch"] is denominator
        assert captured_kwargs["labels"].shape == torch.Size([1, 3])

    def test_get_batch_samples_counts_prepared_labels(self):
        trainer = GKDTrainer.__new__(GKDTrainer)
        trainer.model_accepts_loss_kwargs = True
        trainer.compute_loss_func = None
        trainer.args = SimpleNamespace(average_tokens_across_devices=False, n_gpu=0)
        trainer.accelerator = SimpleNamespace(parallelism_config=None)
        trainer.model_wrapped = object()
        prepared_models = []

        def prepare_inputs(model, inputs):
            prepared_models.append(model)
            inputs["labels"] = inputs["prepared_labels"]
            inputs["_gkd_inputs_prepared"] = True

        trainer._prepare_gkd_inputs_for_loss = prepare_inputs
        original_labels = torch.tensor([[1, 2, 3, 4]])
        batch_1 = {
            "labels": original_labels.clone(),
            "prepared_labels": torch.tensor([[1, -100, -100, -100]]),
        }
        batch_2 = {
            "labels": original_labels.clone(),
            "prepared_labels": torch.tensor([[1, 2, 3, -100]]),
        }

        batch_samples, num_items_in_batch = trainer.get_batch_samples(
            iter([batch_1, batch_2]), num_batches=2, device=torch.device("cpu")
        )

        assert num_items_in_batch == 4
        assert prepared_models == [trainer.model_wrapped, trainer.model_wrapped]
        assert all(batch["_gkd_inputs_prepared"] for batch in batch_samples)

    def test_training_step_skips_already_prepared_inputs(self, monkeypatch):
        trainer = GKDTrainer.__new__(GKDTrainer)
        prepare_calls = []

        def prepare_inputs(model, inputs):
            prepare_calls.append((model, inputs))

        def fake_training_step(self, model, inputs, num_items_in_batch=None):
            assert "_gkd_inputs_prepared" not in inputs
            return torch.tensor(2.0)

        trainer._prepare_gkd_inputs_for_loss = prepare_inputs
        monkeypatch.setattr(SFTTrainer, "training_step", fake_training_step, raising=False)

        inputs = {"_gkd_inputs_prepared": True}
        loss = trainer.training_step(object(), inputs, num_items_in_batch=torch.tensor(4))

        assert loss == 2.0
        assert prepare_calls == []
        assert inputs == {}


class TestGKDTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
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
