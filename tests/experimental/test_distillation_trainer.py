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

import pytest
import torch
import torch.nn.functional as F
from datasets import DatasetDict, IterableDatasetDict, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.distillation import DistillationConfig, DistillationTrainer
from trl.experimental.distillation.distillation_trainer import _chunked_divergence_loss
from trl.experimental.gkd.gkd_trainer import GKDTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel, require_peft, require_torch_accelerator


def _reference_chunked_divergence(
    student_hidden,
    teacher_hidden,
    student_w,
    teacher_w,
    completion_mask,
    beta,
    num_items_in_batch=None,
    s_scale=1.0,
    t_scale=1.0,
    s_softcap=None,
    t_softcap=None,
    temperature=1.0,
):
    """Naive full-vocab reference for `_chunked_divergence_loss`: project the whole batch at once (no chunking) and
    build the JSD straight from the definition, so it shares neither the chunking nor `F.kl_div`'s argument order."""
    student_logits = student_hidden.float() @ student_w.float().t()
    teacher_logits = teacher_hidden.float() @ teacher_w.float().t()
    if s_scale != 1.0:
        student_logits = student_logits * s_scale
    if s_softcap is not None:
        student_logits = s_softcap * torch.tanh(student_logits / s_softcap)
    if t_scale != 1.0:
        teacher_logits = teacher_logits * t_scale
    if t_softcap is not None:
        teacher_logits = t_softcap * torch.tanh(teacher_logits / t_softcap)
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_probs, teacher_probs = student_log_probs.exp(), teacher_log_probs.exp()

    if beta == 0.0:  # forward KL: KL(teacher || student)
        per_element = teacher_probs * (teacher_log_probs - student_log_probs)
    elif beta == 1.0:  # reverse KL: KL(student || teacher)
        per_element = student_probs * (student_log_probs - teacher_log_probs)
    else:  # generalized JSD against the mixture M = (1 - beta) * student + beta * teacher
        mixture_log_probs = ((1 - beta) * student_probs + beta * teacher_probs).log()
        per_element = beta * (teacher_probs * (teacher_log_probs - mixture_log_probs)) + (1 - beta) * (
            student_probs * (student_log_probs - mixture_log_probs)
        )

    per_token = per_element.sum(dim=-1) * completion_mask  # (B, K)
    denom = completion_mask.sum() if num_items_in_batch is None else num_items_in_batch
    return per_token.sum() / denom


class TestChunkedDivergenceLoss(TrlTestCase):
    """Unit tests for the memory-efficient chunked JSD loss (`_chunked_divergence_loss`)."""

    def _inputs(self, B=2, K=6, H=8, V=17, n_masked=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        student_hidden = torch.randn(B, K, H, generator=g)
        teacher_hidden = torch.randn(B, K, H, generator=g)
        student_w = torch.randn(V, H, generator=g)
        teacher_w = torch.randn(V, H, generator=g)
        completion_mask = torch.ones(B, K)
        # Mask a few scattered positions so the packed masked tail is exercised.
        completion_mask.reshape(-1)[torch.randperm(B * K, generator=g)[:n_masked]] = 0
        return student_hidden, teacher_hidden, student_w, teacher_w, completion_mask

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("chunk_size", [3, 4, 100])  # divides / doesn't divide / exceeds n_valid (= 9)
    def test_matches_naive_full_vocab(self, beta, chunk_size):
        sh, th, sw, tw, mask = self._inputs()
        loss, _, n_valid = _chunked_divergence_loss(sh, th, sw, tw, mask, beta, chunk_size)
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta)
        torch.testing.assert_close(loss, expected)
        assert n_valid.item() == int(mask.sum().item())

    def test_different_teacher_student_hidden_sizes(self):
        # Teacher and student may have different hidden widths; only the vocabulary must match.
        g = torch.Generator().manual_seed(2)
        B, K, V = 2, 5, 13
        sh, th = torch.randn(B, K, 8, generator=g), torch.randn(B, K, 12, generator=g)
        sw, tw = torch.randn(V, 8, generator=g), torch.randn(V, 12, generator=g)
        mask = torch.ones(B, K)
        mask[1, -1] = 0
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta=0.5)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_applies_logit_scale_and_softcapping(self, beta):
        # Per-model `logit_scale` (Cohere) / `final_logit_softcapping` (Gemma) must be applied before the softmax.
        sh, th, sw, tw, mask = self._inputs()
        loss, _, _ = _chunked_divergence_loss(
            sh,
            th,
            sw,
            tw,
            mask,
            beta,
            chunk_size=4,
            student_logit_scale=0.7,
            teacher_logit_scale=1.3,
            student_final_logit_softcapping=50.0,
            teacher_final_logit_softcapping=30.0,
        )
        expected = _reference_chunked_divergence(
            sh, th, sw, tw, mask, beta, s_scale=0.7, t_scale=1.3, s_softcap=50.0, t_softcap=30.0
        )
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_applies_temperature(self, beta):
        # Softmax temperature softens both distributions before the divergence.
        sh, th, sw, tw, mask = self._inputs()
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta, chunk_size=4, temperature=2.0)
        expected = _reference_chunked_divergence(sh, th, sw, tw, mask, beta, temperature=2.0)
        torch.testing.assert_close(loss, expected)

    def test_beta_1_is_reverse_kl(self):
        sh, th, sw, tw, mask = self._inputs()
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=1.0, chunk_size=4)
        # Hand-rolled reverse KL: sum_x p_s * (log p_s - log p_t) over valid positions, normalized by n_valid.
        student_log_probs = torch.log_softmax(sh @ sw.t(), dim=-1)
        teacher_log_probs = torch.log_softmax(th @ tw.t(), dim=-1)
        per_token = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(dim=-1) * mask
        expected = per_token.sum() / mask.sum()
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    def test_parity_with_gkd(self, beta):
        # Identity lm_head so hidden states are the logits, then compare against GKD's full-vocab JSD (sum reduction).
        B, K, V = 2, 5, 11
        g = torch.Generator().manual_seed(1)
        student_logits = torch.randn(B, K, V, generator=g)
        teacher_logits = torch.randn(B, K, V, generator=g)
        eye = torch.eye(V)
        mask = torch.ones(B, K)
        mask[0, -1] = 0
        labels = torch.where(mask.bool(), torch.ones_like(mask, dtype=torch.long), torch.full_like(mask, -100).long())
        loss, _, _ = _chunked_divergence_loss(
            student_logits, teacher_logits, eye, eye, mask, beta, chunk_size=4, num_items_in_batch=1
        )
        gkd = GKDTrainer.generalized_jsd_loss(
            student_logits, teacher_logits, labels=labels, beta=beta, reduction="sum"
        )
        torch.testing.assert_close(loss, gkd)

    def test_masked_positions_ignored(self):
        sh, th, sw, tw, mask = self._inputs()
        loss_a, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        # Perturbing the hidden states at masked positions only must not change the loss.
        masked = mask.reshape(-1) == 0
        sh2, th2 = sh.clone().reshape(-1, sh.size(-1)), th.clone().reshape(-1, th.size(-1))
        sh2[masked] += 5.0
        th2[masked] += 5.0
        loss_b, _, _ = _chunked_divergence_loss(sh2.view_as(sh), th2.view_as(th), sw, tw, mask, beta=0.5, chunk_size=4)
        torch.testing.assert_close(loss_a, loss_b)

    def test_grads_flow_and_zero_at_masked(self):
        sh, th, sw, tw, mask = self._inputs()
        sh = sh.clone().requires_grad_(True)
        loss, _, _ = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        loss.backward()
        grad = sh.grad.reshape(-1, sh.size(-1))
        valid = mask.reshape(-1) != 0
        assert (grad[valid].abs().sum(dim=-1) > 0).all()  # valid positions receive gradient
        assert torch.equal(grad[~valid], torch.zeros_like(grad[~valid]))  # masked positions get none

    def test_fully_masked_batch_keeps_graph(self):
        sh, th, sw, tw, _ = self._inputs()
        sh = sh.clone().requires_grad_(True)
        mask = torch.zeros(sh.size(0), sh.size(1))
        loss, _, n_valid = _chunked_divergence_loss(sh, th, sw, tw, mask, beta=0.5, chunk_size=4)
        assert n_valid.item() == 0
        assert torch.isfinite(loss)
        loss.backward()  # must not raise: the graph stays connected through the student hidden states + lm_head
        assert sh.grad is not None


class TestDistillationTrainer(TrlTestCase):
    def setup_method(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _make_args(self, **kwargs):
        args = {
            "output_dir": self.tmp_dir,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "disable_tqdm": True,
            "use_cpu": True,
            "bf16": False,
            "max_completion_length": 32,
            "model_init_kwargs": {"dtype": "float32", "device_map": None},
            "teacher_model_init_kwargs": {"dtype": "float32", "device_map": None},
        }
        args.update(kwargs)
        return DistillationConfig(**args)

    def _make_local_trainer(self, **kwargs):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        return DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(**kwargs),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

    def test_distillation_trainer_train_runs_with_local_teacher(self):
        training_args = self._make_args(
            dataloader_drop_last=True,
            eval_strategy="steps",
            max_steps=4,
            eval_steps=2,
            save_strategy="steps",
            save_steps=2,
            per_device_eval_batch_size=2,
        )
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only")
        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=self.tokenizer,
        )

        train_result = trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert trainer.state.log_history[0]["eval_loss"] is not None
        # Self-distillation (teacher == student), so the divergence is ~0; allow tiny floating-point noise below zero
        # while still catching a genuinely negative loss.
        assert train_result.metrics["train_loss"] >= -1e-4
        assert "model.safetensors" in os.listdir(self.tmp_dir + "/checkpoint-2")

    def test_train_updates_params(self):
        """Training is always on-policy: the student generates completions, the teacher scores them, params move."""
        # Higher lr than the default: gradients are tiny on this model and the default lr can stall the update, which
        # would make the assertion below vacuous.
        trainer = self._make_local_trainer(max_steps=2, learning_rate=0.1)

        # Diverge the teacher from the student so the divergence (and thus the gradient) is well above fp noise; with
        # matched weights it would be ~0 and the update below could pass on noise alone.
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        previous_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for name, param in previous_params.items():
            assert not torch.equal(param, trainer.model.get_parameter(name)), f"Parameter {name} has not changed."

    def test_train_runs_with_prompt_only_dataset(self):
        """The forward-looking prompt-only format trains end to end: the student generates, the teacher scores."""
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(max_steps=1, learning_rate=0.1),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

        assert all(torch.isfinite(param).all() for param in trainer.model.parameters())

    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        train_dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "conversational_prompt_only", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DistillationConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        # The distillation collator consumes raw examples, so eval datasets are stored as-is (not tokenized).
        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset

    def test_loss_normalizes_by_num_items_in_batch(self):
        # When `num_items_in_batch` is passed (as under gradient accumulation), the divergence loss must be reduced as
        # sum / num_items_in_batch rather than the local per-microbatch mean. See issue #4719. The chunked JSD path
        # must honor `num_items_in_batch`.
        trainer = self._make_local_trainer(beta=0.5)

        # Diverge the teacher from the student so the divergence is well above fp noise (else the loss is ~0).
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        # The collator is prompt-only (completions come from on-policy generation); build a batch with completion
        # tokens directly, in GRPO's key layout, to exercise the loss reduction.
        device = trainer.accelerator.device
        prompt_length, completion_length = 4, 3
        vocab_size = trainer.model.config.vocab_size
        completion_mask = torch.ones(2, completion_length, dtype=torch.long, device=device)
        batch = {
            "prompt_ids": torch.randint(0, vocab_size, (2, prompt_length), device=device),
            "prompt_mask": torch.ones(2, prompt_length, dtype=torch.long, device=device),
            "completion_ids": torch.randint(0, vocab_size, (2, completion_length), device=device),
            "completion_mask": completion_mask,
        }

        # Number of valid (non-masked) completion tokens in the local batch.
        num_valid = completion_mask.sum()

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)
            loss_double = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid * 2)

        # With num_items_in_batch equal to the local valid-token count, sum/N equals the local mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)
        # Doubling the global count exactly halves the loss (sum / num_items is linear in 1/num_items).
        torch.testing.assert_close(loss_double, loss_mean / 2, rtol=1e-4, atol=1e-6)

    def test_compute_loss_matches_full_logit_reference(self):
        # The chunked JSD path must produce the same loss as the full-logit JSD it replaced (the handover). This also
        # checks the wiring: the student stays the student, the teacher the teacher, and only completion positions score.
        # A non-symmetric `beta` (0.25, not 0.5) is used deliberately so a student/teacher swap would change the loss.
        trainer = self._make_local_trainer(beta=0.25)
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))  # diverge the teacher so the JSD is well above fp noise

        device = trainer.accelerator.device
        vocab_size = trainer.model.config.vocab_size
        prompt_length, completion_length = 4, 3
        batch = {
            "prompt_ids": torch.randint(0, vocab_size, (2, prompt_length), device=device),
            "prompt_mask": torch.ones(2, prompt_length, dtype=torch.long, device=device),
            "completion_ids": torch.randint(0, vocab_size, (2, completion_length), device=device),
            "completion_mask": torch.ones(2, completion_length, dtype=torch.long, device=device),
        }
        num_valid = batch["completion_mask"].sum()

        trainer.model.eval()
        with torch.no_grad():
            loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

            # Full-logit reference (the pre-chunking loss): run both models' full forward, take the completion logits,
            # and compute the generalized JSD over the completion positions, normalized by num_items.
            input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
            attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
            keep = slice(-completion_length - 1, -1)
            s = trainer.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, keep, :]
            t = trainer.teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, keep, :]
            slp = torch.log_softmax(s.float(), dim=-1)
            tlp = torch.log_softmax(t.float(), dim=-1)
            beta_t = torch.tensor(0.25)
            mixture = torch.logsumexp(torch.stack([slp + torch.log1p(-beta_t), tlp + torch.log(beta_t)]), dim=0)
            jsd = 0.25 * F.kl_div(mixture, tlp, reduction="none", log_target=True) + 0.75 * F.kl_div(
                mixture, slp, reduction="none", log_target=True
            )
            reference = jsd.sum() / num_valid

        torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-6)

    def test_generated_batch_emits_completion_mask(self, monkeypatch):
        """The generated batch emits a region-shaped `completion_mask` that the loss consumes via GRPO's key layout."""
        trainer = self._make_local_trainer()
        captured = {}
        original = DistillationTrainer.compute_loss

        def _capturing(self, model, inputs, *args, **kwargs):
            captured.setdefault("inputs", {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()})
            return original(self, model, inputs, *args, **kwargs)

        monkeypatch.setattr(DistillationTrainer, "compute_loss", _capturing)
        trainer.train()

        inputs = captured["inputs"]
        assert "completion_mask" in inputs
        # Region-shaped (B, C), aligned with `completion_ids`.
        assert inputs["completion_mask"].shape == inputs["completion_ids"].shape

    def test_generated_batch_emits_prompt_and_completion_ids(self, monkeypatch):
        """The generated batch emits the GRPO-style keys the loss consumes."""
        trainer = self._make_local_trainer()
        captured = {}
        original = DistillationTrainer.compute_loss

        def _capturing(self, model, inputs, *args, **kwargs):
            captured.setdefault("inputs", {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()})
            return original(self, model, inputs, *args, **kwargs)

        monkeypatch.setattr(DistillationTrainer, "compute_loss", _capturing)
        trainer.train()

        inputs = captured["inputs"]
        for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "num_items_in_batch"):
            assert key in inputs

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_agrees_with_chunked(self):
        # The Liger fused path and the default chunked path compute the same JSD objective (they share the hidden-state
        # extraction and differ only in the final loss call), so they must agree on a fixed batch. Toggling
        # `use_liger_loss` on one trainer keeps the same (un-patched) model, so only the loss kernel differs.
        from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(beta=0.5, use_cpu=False),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        # Diverge the teacher so the JSD is well above fp noise.
        torch.manual_seed(0)
        with torch.no_grad():
            for p in trainer.teacher_model.parameters():
                p.add_(0.5 * torch.randn_like(p))

        device = trainer.accelerator.device
        vocab_size = trainer.model.config.vocab_size
        gen = torch.Generator().manual_seed(1)
        prompt_length, completion_length = 4, 3
        batch = {
            "prompt_ids": torch.randint(0, vocab_size, (2, prompt_length), generator=gen).to(device),
            "prompt_mask": torch.ones(2, prompt_length, dtype=torch.long, device=device),
            "completion_ids": torch.randint(0, vocab_size, (2, completion_length), generator=gen).to(device),
            "completion_mask": torch.ones(2, completion_length, dtype=torch.long, device=device),
        }
        num_valid = batch["completion_mask"].sum()

        trainer.model.eval()
        with torch.no_grad():
            chunked_loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)
            trainer.use_liger_loss = True
            trainer.liger_loss = LigerFusedLinearJSDLoss(
                beta=trainer.beta,
                ignore_index=-100,
                temperature=trainer.temperature,
                compiled=False,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            liger_loss = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)

        torch.testing.assert_close(liger_loss, chunked_loss, rtol=1e-3, atol=1e-4)

    def test_teacher_vocab_size_mismatch_raises(self):
        # The local-teacher loss compares full next-token distributions, so student and teacher must share a
        # vocabulary. A teacher with a different vocab_size is rejected (use GOLD for cross-tokenizer distillation).
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        with pytest.raises(ValueError, match="vocab_size"):
            DistillationTrainer(
                model=self.model_id,
                teacher_model="trl-internal-testing/tiny-LlamaForCausalLM-3.2",
                args=self._make_args(),
                train_dataset=dataset,
                processing_class=self.tokenizer,
            )

    def test_teacher_model_init_kwargs_with_instantiated_teacher_raises(self):
        # `teacher_model_init_kwargs` only applies when the teacher is a model id; passing it alongside an already
        # instantiated teacher is a mistake worth surfacing.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        with pytest.raises(ValueError, match="teacher_model_init_kwargs"):
            DistillationTrainer(
                model=self.model_id,
                teacher_model=AutoModelForCausalLM.from_pretrained(self.model_id),
                args=self._make_args(),
                train_dataset=dataset,
                processing_class=self.tokenizer,
            )

    @require_peft
    def test_peft_lm_head_adapter_raises(self):
        # Both loss paths read `lm_head.weight` directly, so a PEFT adapter on the head is silently ignored. Reject it.
        from peft import LoraConfig

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        with pytest.raises(ValueError, match="lm_head"):
            DistillationTrainer(
                model=self.model_id,
                teacher_model=self.model_id,
                args=self._make_args(),
                train_dataset=dataset,
                processing_class=self.tokenizer,
                peft_config=LoraConfig(target_modules=["q_proj", "lm_head"]),
            )

    @require_peft
    def test_peft_prompt_learning_raises(self):
        # Prompt-learning injects virtual tokens via `PeftModel.forward()`, which the backbone-direct loss bypasses.
        from peft import PrefixTuningConfig

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        with pytest.raises(ValueError, match="Prompt-learning"):
            DistillationTrainer(
                model=self.model_id,
                teacher_model=self.model_id,
                args=self._make_args(),
                train_dataset=dataset,
                processing_class=self.tokenizer,
                peft_config=PrefixTuningConfig(num_virtual_tokens=4, task_type="CAUSAL_LM"),
            )
