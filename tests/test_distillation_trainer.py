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

import pytest
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers.utils import is_peft_available

from trl import DistillationConfig, DistillationTrainer
from trl.experimental.gkd.gkd_trainer import GKDTrainer
from trl.trainer.distillation_trainer import _chunked_divergence_loss

from .testing_utils import TrlTestCase, require_peft


if is_peft_available():
    from peft import LoraConfig


def _perturbed_teacher(model_id, scale=0.05, seed=42):
    """A teacher that differs from the student, so that the divergence is non-zero."""
    teacher = AutoModelForCausalLM.from_pretrained(model_id)
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in teacher.parameters():
            param.add_(torch.randn_like(param) * scale)
    return teacher


class TestChunkedDivergenceLoss(TrlTestCase):
    """The chunked loss must equal the full-vocab divergence, whatever the chunking."""

    def _tensors(self):
        torch.manual_seed(0)
        student_hidden, teacher_hidden = torch.randn(3, 7, 16), torch.randn(3, 7, 16)
        student_weight, teacher_weight = torch.randn(29, 16), torch.randn(29, 16)
        mask = torch.ones(3, 7)
        mask[0, 5:] = 0  # ragged: some completions are shorter than others
        mask[2, 3:] = 0
        return student_hidden, teacher_hidden, student_weight, teacher_weight, mask

    def _reference(self, student_logits, teacher_logits, mask, beta):
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        if beta == 0.0:
            divergence = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1.0:
            divergence = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            beta_t = torch.tensor(beta)
            mixture = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]), dim=0
            )
            divergence = beta_t * F.kl_div(mixture, teacher_log_probs, reduction="none", log_target=True) + (
                1 - beta_t
            ) * F.kl_div(mixture, student_log_probs, reduction="none", log_target=True)
        return (divergence.sum(-1) * mask).sum() / mask.sum()

    # chunk_size that divides, doesn't divide, and exceeds the number of valid tokens
    @pytest.mark.parametrize("chunk_size", [4, 5, 64])
    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_matches_full_vocab_divergence(self, beta, chunk_size):
        student_hidden, teacher_hidden, student_weight, teacher_weight, mask = self._tensors()
        loss, _, num_valid = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_weight,
            teacher_hidden=teacher_hidden,
            teacher_lm_head_weight=teacher_weight,
            mask=mask,
            beta=beta,
            chunk_size=chunk_size,
            num_items_in_batch=mask.sum(),
        )
        expected = self._reference(
            student_hidden @ student_weight.t(), teacher_hidden @ teacher_weight.t(), mask, beta
        )
        assert torch.allclose(loss, expected, atol=1e-5)
        assert num_valid == mask.sum()

    def test_beta_one_is_reverse_kl(self):
        # The 2026 frontier objective: KL(student || teacher) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
        student_hidden, teacher_hidden, student_weight, teacher_weight, mask = self._tensors()
        student_logits = student_hidden @ student_weight.t()
        teacher_logits = teacher_hidden @ teacher_weight.t()
        per_token = (
            F.softmax(student_logits, dim=-1)
            * (F.log_softmax(student_logits, dim=-1) - F.log_softmax(teacher_logits, dim=-1))
        ).sum(-1)
        expected = (per_token * mask).sum() / mask.sum()
        loss, _, _ = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_weight,
            teacher_hidden=teacher_hidden,
            teacher_lm_head_weight=teacher_weight,
            mask=mask,
            beta=1.0,
            chunk_size=4,
            num_items_in_batch=mask.sum(),
        )
        assert torch.allclose(loss, expected, atol=1e-5)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_matches_gkd_loss(self, beta):
        # The objective must be identical to the one GKD has been shipping.
        student_hidden, teacher_hidden, student_weight, teacher_weight, mask = self._tensors()
        labels = torch.where(
            mask.bool(), torch.zeros_like(mask, dtype=torch.long), torch.full_like(mask, -100, dtype=torch.long)
        )
        expected = GKDTrainer.generalized_jsd_loss(
            student_hidden @ student_weight.t(),
            teacher_hidden @ teacher_weight.t(),
            labels=labels,
            beta=beta,
            num_items_in_batch=mask.sum(),
        )
        loss, _, _ = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_weight,
            teacher_hidden=teacher_hidden,
            teacher_lm_head_weight=teacher_weight,
            mask=mask,
            beta=beta,
            chunk_size=4,
            num_items_in_batch=mask.sum(),
        )
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_masked_positions_are_ignored(self):
        # The padded tail of a chunk holds masked positions; they must not contribute.
        student_hidden, teacher_hidden, student_weight, teacher_weight, mask = self._tensors()
        kwargs = {
            "student_lm_head_weight": student_weight,
            "teacher_hidden": teacher_hidden,
            "teacher_lm_head_weight": teacher_weight,
            "mask": mask,
            "beta": 1.0,
            "chunk_size": 4,
            "num_items_in_batch": mask.sum(),
        }
        loss, _, _ = _chunked_divergence_loss(student_hidden=student_hidden, **kwargs)
        garbled = student_hidden.clone()
        garbled[0, 5:] = 999.0  # masked positions
        garbled_loss, _, _ = _chunked_divergence_loss(student_hidden=garbled, **kwargs)
        assert torch.allclose(loss, garbled_loss, atol=1e-6)

    def test_identical_models_give_zero_divergence(self):
        student_hidden, _, student_weight, _, mask = self._tensors()
        loss, _, _ = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_weight,
            teacher_hidden=student_hidden,
            teacher_lm_head_weight=student_weight,
            mask=mask,
            beta=1.0,
            chunk_size=4,
            num_items_in_batch=mask.sum(),
        )
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

    def test_gradients_flow_to_student(self):
        student_hidden, teacher_hidden, student_weight, teacher_weight, mask = self._tensors()
        student_hidden = student_hidden.clone().requires_grad_(True)
        loss, _, _ = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_weight.clone().requires_grad_(True),
            teacher_hidden=teacher_hidden,
            teacher_lm_head_weight=teacher_weight,
            mask=mask,
            beta=1.0,
            chunk_size=4,
            num_items_in_batch=mask.sum(),
        )
        loss.backward()
        assert student_hidden.grad.abs().sum() > 0
        # Masked positions must receive no gradient
        assert torch.equal(student_hidden.grad[0, 5:], torch.zeros_like(student_hidden.grad[0, 5:]))

    def test_fully_masked_batch_keeps_graph(self):
        # Every completion truncated: the loss must stay attached so backward() doesn't hang DDP/FSDP.
        student_hidden, teacher_hidden, student_weight, teacher_weight, _ = self._tensors()
        student_hidden = student_hidden.clone().requires_grad_(True)
        loss, _, num_valid = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_weight.clone().requires_grad_(True),
            teacher_hidden=teacher_hidden,
            teacher_lm_head_weight=teacher_weight,
            mask=torch.zeros(3, 7),
            beta=1.0,
            chunk_size=4,
            num_items_in_batch=1,
        )
        assert num_valid == 0
        assert loss.requires_grad
        loss.backward()  # must not raise


class TestDistillationTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=_perturbed_teacher(model_id),
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_dataset_format(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=_perturbed_teacher(model_id),
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_train_beta(self, beta):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            max_completion_length=8,
            beta=beta,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=_perturbed_teacher(model_id),
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_teacher_from_model_id(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id, teacher_model=model_id, args=training_args, train_dataset=dataset
        )
        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_identical_teacher_gives_zero_loss(self):
        # A student distilling from itself has nothing to learn: the divergence is exactly zero.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            max_completion_length=8,
            max_steps=1,
            logging_steps=1,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id, teacher_model=model_id, args=training_args, train_dataset=dataset
        )
        trainer.train()

        losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        assert losses[0] == pytest.approx(0.0, abs=1e-6)

    def test_teacher_vocab_size_mismatch_raises(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = DistillationConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="vocab_size"):
            DistillationTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                teacher_model="trl-internal-testing/tiny-LlamaForCausalLM-3.2",
                args=training_args,
                train_dataset=dataset,
            )

    def test_teacher_model_init_kwargs_with_instantiated_teacher_raises(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        training_args = DistillationConfig(
            output_dir=self.tmp_dir, teacher_model_init_kwargs={"dtype": "float32"}, report_to="none"
        )
        with pytest.raises(ValueError, match="teacher_model_init_kwargs"):
            DistillationTrainer(
                model=model_id,
                teacher_model=AutoModelForCausalLM.from_pretrained(model_id),
                args=training_args,
                train_dataset=dataset,
            )

    @require_peft
    def test_train_peft(self):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        training_args = DistillationConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = DistillationTrainer(
            model=model_id,
            teacher_model=_perturbed_teacher(model_id),
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(),
        )

        base_params = {n: param.clone() for n, param in trainer.model.named_parameters() if "lora" not in n.lower()}
        lora_params = {n: param.clone() for n, param in trainer.model.named_parameters() if "lora" in n.lower()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Base parameters must be frozen, only the adapter trains
        for n, param in base_params.items():
            new_param = trainer.model.get_parameter(n)
            assert torch.equal(param, new_param), f"Base parameter {n} has changed."
        for n, param in lora_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"LoRA parameter {n} has not changed."
