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

import importlib
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl.experimental.gkd import GKDConfig, GKDTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel, require_torch_accelerator


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

        outputs = GKDTrainer.generate_on_policy_outputs(self.model, inputs, deterministic_generation_config)

        new_input_ids, new_attention_mask, new_labels = outputs

        # Decode the generated outputs
        generated_texts = self.tokenizer.batch_decode(new_input_ids, skip_special_tokens=True)

        # Check if the generated texts start with the original prompts
        for prompt, generated_text in zip(prompts, generated_texts, strict=True):
            assert generated_text.startswith(prompt), (
                f"Generated text '{generated_text}' does not start with prompt '{prompt}'"
            )

        # Run the generation twice and check if the outputs are identical
        outputs2 = GKDTrainer.generate_on_policy_outputs(self.model, inputs, deterministic_generation_config)

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

        outputs = GKDTrainer.generate_on_policy_outputs(self.model, inputs, self.generation_config)

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

    def test_generate_on_policy_outputs_masks_prompt(self):
        # The on-policy / seq-KD labels must mask the prompt with -100, matching the collator
        # convention (labels[:len(prompt)] = -100). Otherwise `compute_loss`, which relies on the
        # -100 mask to skip prompts, would apply the JSD loss to prompt positions too.
        # Prompts have different lengths so batched left-padding is exercised (the bug condition).
        prompts = ["Hello, how are you doing today?", "Hi"]
        self.tokenizer.padding_side = "left"
        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)
        prompt_width = tokenized_prompts["input_ids"].shape[1]

        inputs = {
            "prompts": tokenized_prompts["input_ids"].to(self.device),
            "prompt_attention_mask": tokenized_prompts["attention_mask"].to(self.device),
        }

        # Force a non-trivial completion so the completion region is not all padding (which would
        # let the prompt-masking assertion pass for the wrong reason).
        generation_config = GenerationConfig(
            max_new_tokens=16,
            min_new_tokens=8,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )

        _, _, new_labels = GKDTrainer.generate_on_policy_outputs(self.model, inputs, generation_config)

        # Every prompt position (the first `prompt_width` columns) must be masked.
        assert (new_labels[:, :prompt_width] == -100).all(), "Prompt positions are not fully masked"
        # The completion region must still carry signal (not entirely masked away).
        assert (new_labels[:, prompt_width:] != -100).any(), "Completion tokens were unexpectedly all masked"

    def test_generate_on_policy_outputs_pad_equals_eos_keeps_eos(self):
        # pad == eos here (the setup ties them): identity-based pad masking used to erase the
        # terminating EOS label on every row and zero attention on pad-id tokens inside the prompt.
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        assert pad_id == eos_id

        # Row 0 stops on EOS and is right-padded; row 1 runs to max_new_tokens (no padding).
        prompts = torch.tensor([[pad_id, 11, eos_id, 13], [pad_id, 11, eos_id, 13]], device=self.device)
        prompt_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], device=self.device)
        completions = torch.tensor([[21, 22, eos_id, pad_id, pad_id], [21, 22, 23, 24, 25]], device=self.device)
        generated_sequence = torch.cat([prompts, completions], dim=1)

        class DummyModel:
            def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
                return SimpleNamespace(sequences=generated_sequence)

        inputs = {"prompts": prompts, "prompt_attention_mask": prompt_mask}
        _, new_attention_mask, new_labels = GKDTrainer.generate_on_policy_outputs(
            DummyModel(), inputs, self.generation_config
        )

        prompt_width = prompts.shape[1]
        eos_pos = prompt_width + 2
        # Pad-id tokens inside the prompt keep attention, only real prompt padding is masked
        assert torch.equal(new_attention_mask[:, :prompt_width], prompt_mask)
        # The terminating EOS stays attended and supervised, only the padding after it is masked
        assert new_attention_mask[0, eos_pos] == 1
        assert new_labels[0, eos_pos] == eos_id
        assert torch.all(new_attention_mask[0, eos_pos + 1 :] == 0)
        assert torch.all(new_labels[0, eos_pos + 1 :] == -100)
        # A row that hits max_new_tokens has no padding and stays fully supervised
        assert torch.all(new_attention_mask[1, prompt_width:] == 1)
        assert torch.equal(new_labels[1, prompt_width:], completions[1])
        # Prompt positions never contribute to the loss
        assert torch.all(new_labels[:, :prompt_width] == -100)

    def test_generate_on_policy_outputs_without_eos_id_keeps_full_completion(self):
        # GenerationConfig defaults eos_token_id to None; torch.tensor(None) would raise, so the
        # None guard must keep the whole completion instead of trying to find a stop token.
        prompts = torch.tensor([[11, 12, 13], [14, 15, 16]], device=self.device)
        prompt_mask = torch.ones_like(prompts)
        completions = torch.tensor([[21, 22, 23, 24], [25, 26, 27, 28]], device=self.device)
        generated_sequence = torch.cat([prompts, completions], dim=1)

        class DummyModel:
            def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
                return SimpleNamespace(sequences=generated_sequence)

        generation_config = SimpleNamespace(eos_token_id=None)
        inputs = {"prompts": prompts, "prompt_attention_mask": prompt_mask}
        _, new_attention_mask, new_labels = GKDTrainer.generate_on_policy_outputs(
            DummyModel(), inputs, generation_config
        )

        prompt_width = prompts.shape[1]
        # With no stop token nothing is masked in the completion region
        assert torch.all(new_attention_mask[:, prompt_width:] == 1)
        assert torch.all(new_labels[:, prompt_width:] != -100)

    def test_generate_on_policy_outputs_masks_after_any_stop_token(self):
        # eos_token_id can be a list of stop tokens; each row must stop on the first occurrence of
        # any of them. A scalar == would only catch one id and miss the other row's terminator.
        prompts = torch.tensor([[11, 12, 13], [14, 15, 16]], device=self.device)
        prompt_mask = torch.ones_like(prompts)
        # Row 0 terminates on 7, row 1 terminates on 9, each at a different position
        completions = torch.tensor([[21, 7, 23, 24], [25, 26, 9, 28]], device=self.device)
        generated_sequence = torch.cat([prompts, completions], dim=1)

        class DummyModel:
            def generate(self, input_ids, attention_mask, generation_config, return_dict_in_generate):
                return SimpleNamespace(sequences=generated_sequence)

        generation_config = SimpleNamespace(eos_token_id=[7, 9])
        inputs = {"prompts": prompts, "prompt_attention_mask": prompt_mask}
        _, new_attention_mask, new_labels = GKDTrainer.generate_on_policy_outputs(
            DummyModel(), inputs, generation_config
        )

        prompt_width = prompts.shape[1]
        completion_mask = new_attention_mask[:, prompt_width:]
        # Row 0 stops on 7 at index 1, row 1 stops on 9 at index 2: keep up to and including it
        assert torch.equal(completion_mask[0], torch.tensor([1, 1, 0, 0], device=self.device))
        assert torch.equal(completion_mask[1], torch.tensor([1, 1, 1, 0], device=self.device))
        assert new_labels[0, prompt_width + 1] == 7
        assert new_labels[1, prompt_width + 2] == 9
        assert torch.all(new_labels[0, prompt_width + 2 :] == -100)
        assert torch.all(new_labels[1, prompt_width + 3 :] == -100)


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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="float32")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def teardown_method(self):
        if hasattr(self, "_liger_module"):
            importlib.reload(importlib.import_module(self._liger_module))

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
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=self.tokenizer,
        )

        trainer.train()

        assert trainer.state.log_history[(-1)]["train_loss"] is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None
        assert "model.safetensors" in os.listdir(self.tmp_dir + "/checkpoint-2")

    @require_liger_kernel
    def test_gkd_trainer_with_liger(self):
        training_args = GKDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            max_length=64,
            report_to="none",
            use_liger_kernel=True,
        )
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        self._liger_module = trainer.model.__module__

        # Ensure liger fused JSD path is enabled; if not, skip (runtime may lack system libs)
        if not getattr(trainer, "use_liger_gkd_loss", False):
            pytest.skip("Liger fused JSD not enabled at runtime; skipping fused-loss assertion")

        trainer.train()

        # Check we logged a train loss
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_generation_config_init(self):
        training_args = GKDConfig(output_dir=self.tmp_dir)
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=self.tokenizer,
        )

        assert trainer.generation_config.pad_token_id == self.tokenizer.eos_token_id
        assert trainer.generation_config.eos_token_id == self.model.generation_config.eos_token_id
        assert trainer.generation_config.max_new_tokens == training_args.max_new_tokens
        assert trainer.generation_config.temperature == training_args.temperature
        assert trainer.generation_config.top_k == 0
        assert trainer.generation_config.top_p == 1.0

    def test_init_multimodal_model(self):
        """Multimodal configs keep vocab_size in their text_config; the vocab check must handle them."""
        model_id = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        training_args = GKDConfig(output_dir=self.tmp_dir, report_to="none")
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")

        trainer = GKDTrainer(
            model=model_id,
            teacher_model=model_id,
            args=training_args,
            train_dataset=dataset["train"],
            processing_class=tokenizer,
        )

        student_vocab_size = trainer.model.config.get_text_config().vocab_size
        assert student_vocab_size == trainer.teacher_model.config.get_text_config().vocab_size

    @require_liger_kernel
    def test_compute_loss_return_outputs_with_liger(self):
        """Test that return_outputs=True works correctly with Liger kernel path."""
        training_args = GKDConfig(
            output_dir=self.tmp_dir,
            report_to="none",
            use_liger_kernel=True,
            max_steps=2,
            eval_strategy="steps",
            eval_steps=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
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
        self._liger_module = trainer.model.__module__

        # evaluate() calls compute_loss with return_outputs=True; must not raise UnboundLocalError
        eval_results = trainer.evaluate()
        assert "eval_loss" in eval_results
        assert eval_results["eval_loss"] is not None

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_matches_non_liger_loss(self):
        # The Liger fused JSD path must compute the same loss as the non-Liger path
        # (LigerFusedLinearJSDLoss defaults mix 0.5 * CE + 0.5 * JSD, but GKD wants pure JSD).
        common = dict(output_dir=self.tmp_dir, report_to="none", per_device_train_batch_size=2, max_length=64)
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train").select(
            range(2)
        )

        ref_trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=GKDConfig(use_liger_kernel=False, **common),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        liger_trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=GKDConfig(use_liger_kernel=True, **common),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        self._liger_module = liger_trainer.model.__module__

        # Force student/teacher weights identical between trainers, then diverge teacher
        # so JSD is well above fp noise.
        liger_trainer.model.load_state_dict(ref_trainer.model.state_dict())
        torch.manual_seed(0)
        with torch.no_grad():
            for p in ref_trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))
        liger_trainer.teacher_model.load_state_dict(ref_trainer.teacher_model.state_dict())

        device = next(ref_trainer.model.parameters()).device
        batch = ref_trainer.data_collator([ref_trainer.train_dataset[i] for i in range(2)])
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        ref_trainer.model.eval()
        liger_trainer.model.eval()
        with torch.no_grad():
            ref_loss = ref_trainer.compute_loss(ref_trainer.model, batch).item()
            liger_loss = liger_trainer.compute_loss(liger_trainer.model, batch).item()

        torch.testing.assert_close(
            torch.tensor(liger_loss),
            torch.tensor(ref_loss),
            rtol=2e-2,
            atol=1e-6,
        )

    @require_torch_accelerator
    def test_loss_normalizes_by_num_items_in_batch(self):
        # When `num_items_in_batch` is passed (as under gradient accumulation), the loss must be the JSD summed over
        # valid tokens divided by that global count, rather than the local per-microbatch mean. See issue #4719.
        # GPU-gated like `test_liger_loss_matches_non_liger_loss`: GKD's loss path is accelerator-affine, so the model
        # runs on the device rather than being forced to CPU.
        common = dict(
            output_dir=self.tmp_dir,
            report_to="none",
            per_device_train_batch_size=2,
            max_length=64,
        )
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train").select(
            range(2)
        )
        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=GKDConfig(use_liger_kernel=False, **common),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        # Diverge the teacher from the student so JSD is well above fp noise (else the loss is identically 0).
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        device = next(trainer.model.parameters()).device
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Number of valid (non-ignored) tokens in the local batch, sliced the same way `compute_loss` does.
        prompt_lengths = batch["prompts"].shape[1]
        num_valid = (batch["labels"][:, prompt_lengths:] != -100).sum()

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

        # With num_items_in_batch equal to the local valid-token count, sum/N equals the local mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)

        # Passing a different global count rescales the loss exactly by num_valid / num_items_in_batch. This is the
        # gradient-accumulation-correct behavior: a microbatch contributes its token-sum divided by the *global* count.
        loss_double = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid * 2)
        torch.testing.assert_close(loss_double, loss_mean / 2, rtol=1e-4, atol=1e-6)

    @require_torch_accelerator
    def test_loss_covers_all_completion_tokens_with_variable_length_prompts(self):
        # The loss must be computed over EVERY valid completion token, even when prompts have different lengths.
        # A previous implementation sliced logits/labels by `inputs["prompts"].shape[1]` (the batch-max prompt
        # width); because `labels` is padded to the full-sequence width independently of `prompts`, that slice
        # dropped completion tokens for samples whose prompt was shorter than the batch maximum, mis-scaling the
        # loss when normalized by `num_items_in_batch`. See issue #4719.
        common = dict(output_dir=self.tmp_dir, report_to="none", per_device_train_batch_size=2, max_length=64)
        # Two conversations with deliberately different prompt lengths.
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there, how are you?"}],
                    [
                        {"role": "user", "content": "Please explain in detail the theory of general relativity"},
                        {"role": "assistant", "content": "OK"},
                    ],
                ]
            }
        )
        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=GKDConfig(use_liger_kernel=False, **common),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        device = next(trainer.model.parameters()).device
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # The prompts are different lengths, so the batch-max prompt width exceeds at least one sample's prompt.
        assert batch["prompts"].shape[1] > (batch["labels"][0] != -100).nonzero()[0].item()

        # All valid completion tokens across the batch — what num_items_in_batch counts.
        num_valid = (batch["labels"] != -100).sum()

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # None -> local mean over the tokens it summed
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

        # If the loss covers every valid completion token, the global-count reduction (sum / num_valid) equals the
        # local mean. The old prompt-width slice summed FEWER tokens than num_valid, so loss_global != loss_mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_normalizes_by_num_items_in_batch(self):
        # The Liger fused JSD path normalizes by the local valid-token count internally; passing num_items_in_batch
        # must rescale it to the global count (see issue #4719). Mirrors the non-Liger test on the Liger path.
        common = dict(output_dir=self.tmp_dir, report_to="none", per_device_train_batch_size=2, max_length=64)
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train").select(
            range(2)
        )
        trainer = GKDTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=GKDConfig(use_liger_kernel=True, **common),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        self._liger_module = trainer.model.__module__
        if not getattr(trainer, "use_liger_gkd_loss", False):
            pytest.skip("Liger fused JSD not enabled at runtime; skipping fused-loss assertion")

        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        device = next(trainer.model.parameters()).device
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_k = trainer.compute_loss(trainer.model, batch, num_items_in_batch=100)
            loss_2k = trainer.compute_loss(trainer.model, batch, num_items_in_batch=200)

        # Doubling the global count exactly halves the loss; the rescaled loss differs from the local mean.
        torch.testing.assert_close(loss_2k, loss_k / 2, rtol=1e-4, atol=1e-6)
        assert not torch.allclose(loss_k, loss_mean)
