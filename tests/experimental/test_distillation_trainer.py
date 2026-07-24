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
from trl.experimental.distillation.distillation_trainer import _RepeatBatchDataLoader
from trl.experimental.gkd.gkd_trainer import GKDTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel, require_torch_accelerator


def _reference_generalized_jsd(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0):
    """Naive reference for the generalized JSD, written straight from the definition.

    Deliberately independent of the implementation: probabilities are formed explicitly and the mixture is built in
    probability space, so this does not share `F.kl_div`'s inverted argument order nor the `logsumexp` trick. That is
    what makes it able to catch an argument-order or mixture-weight regression.
    """
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits / temperature, dim=-1)
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

    if labels is None:  # "batchmean" without labels divides by the batch size
        return per_element.sum() / max(per_element.size(0), 1)
    mask = labels != -100
    return per_element[mask].sum() / mask.sum().clamp(min=1)


class TestGeneralizedJSDLossIsPinned(TrlTestCase):
    """Pins the distillation objective while the trainer is refactored.

    The implementation is expected to change (top-k support removal, then the switch to a chunked loss); the value it
    computes is not. Any diff that moves these numbers is changing the objective and must say so.
    """

    def setup_method(self):
        generator = torch.Generator().manual_seed(42)  # seeded: an unseeded fixture cannot pin anything
        self.student_logits = torch.randn(2, 3, 5, generator=generator)
        self.teacher_logits = torch.randn(2, 3, 5, generator=generator)
        self.labels = torch.tensor([[-100, 1, 2], [-100, -100, 3]])

    @pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
    @pytest.mark.parametrize("use_labels", [False, True])
    def test_matches_reference_implementation(self, beta, use_labels):
        labels = self.labels if use_labels else None
        loss = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, labels=labels, beta=beta
        )
        expected = _reference_generalized_jsd(self.student_logits, self.teacher_logits, labels=labels, beta=beta)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
    @pytest.mark.parametrize("use_labels", [False, True])
    def test_matches_gkd(self, beta, use_labels):
        # GKD implements the same objective. Keeping the two in lockstep is the cross-trainer contract: if this breaks,
        # either the promotion changed the objective or GKD drifted.
        labels = self.labels if use_labels else None
        loss = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, labels=labels, beta=beta
        )
        gkd_loss = GKDTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, labels=labels, beta=beta)
        torch.testing.assert_close(loss, gkd_loss)

    @pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
    def test_temperature_matches_reference(self, beta):
        # `temperature` is applied to the loss today. It is scheduled to become sampling-only, so pin it explicitly:
        # that change must be a visible diff here, not a silent drift.
        loss = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, labels=self.labels, beta=beta, temperature=2.0
        )
        expected = _reference_generalized_jsd(
            self.student_logits, self.teacher_logits, labels=self.labels, beta=beta, temperature=2.0
        )
        torch.testing.assert_close(loss, expected)


class TestGeneralizedJSDLoss(TrlTestCase):
    def setup_method(self):
        self.batch_size = 2
        self.seq_length = 3
        self.vocab_size = 5
        self.student_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        self.teacher_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)

    def test_uniform_distribution(self):
        logits = torch.ones(1, 1, self.vocab_size)
        loss = DistillationTrainer.generalized_jsd_loss(logits, logits)
        assert round(abs(loss.item() - 0), 5) == 0

    def test_generalized_jsd_loss_edge_cases(self):
        # Setup
        student_logits = torch.log(torch.tensor([[0.1, 0.9]])).unsqueeze(0)
        teacher_logits = torch.log(torch.tensor([[0.9, 0.1]])).unsqueeze(0)

        # Case 1: beta = 1 (should be equivalent to KL(student || teacher))
        loss_beta_1 = DistillationTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=1)
        expected_loss_beta_1 = F.kl_div(
            F.log_softmax(teacher_logits, dim=-1), F.softmax(student_logits, dim=-1), reduction="batchmean"
        )
        assert round(abs(loss_beta_1.item() - expected_loss_beta_1.item()), 5) == 0

        # Case 2: beta = 0 (should be equivalent to KL(teacher || student))
        loss_beta_0 = DistillationTrainer.generalized_jsd_loss(student_logits, teacher_logits, beta=0)
        expected_loss_beta_0 = F.kl_div(
            F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean"
        )
        assert round(abs(loss_beta_0.item() - expected_loss_beta_0.item()), 5) == 0

    def test_output_shape(self):
        loss = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits)
        assert torch.is_tensor(loss)
        assert loss.shape == torch.Size([])

    def test_beta_values(self):
        loss_beta_0 = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0)
        loss_beta_1 = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=1)
        assert loss_beta_0 != loss_beta_1

    def test_temperature_scaling(self):
        loss_temp_1 = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=1)
        loss_temp_2 = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, temperature=2)
        assert loss_temp_1 != loss_temp_2

    def test_reduction_methods(self):
        loss_batchmean = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, reduction="batchmean"
        )
        loss_sum = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, reduction="sum")
        loss_mean = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, reduction="mean"
        )
        loss_none = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, reduction="none"
        )

        assert loss_batchmean.shape == torch.Size([])
        assert loss_sum.shape == torch.Size([])
        assert loss_mean.shape == torch.Size([])
        assert loss_none.shape == self.student_logits.shape

    def test_symmetry(self):
        student_teacher = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.1)
        teacher_student = DistillationTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.1)
        assert student_teacher != teacher_student

        student_teacher = DistillationTrainer.generalized_jsd_loss(self.student_logits, self.teacher_logits, beta=0.5)
        teacher_student = DistillationTrainer.generalized_jsd_loss(self.teacher_logits, self.student_logits, beta=0.5)
        assert student_teacher == teacher_student

    def test_zero_loss_for_identical_inputs(self):
        identical_logits = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        loss = DistillationTrainer.generalized_jsd_loss(identical_logits, identical_logits)
        assert round(abs(loss.item() - 0), 6) == 0


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

    def test_num_items_in_batch_counts_the_tokens_trained_on(self, monkeypatch):
        """`num_items_in_batch` is the loss denominator, so it must count the completion tokens actually trained on.

        Capture the value where it is applied (`_reduce_divergence_loss`) rather than the argument transformers passes
        to `compute_loss`: the GRPO-style fix computes the count during generation and the loss reads it from there, so
        asserting on the applied denominator keeps the test valid — and able to turn green — across that move.
        """
        recorded = []  # (denominator applied, completion tokens in this microbatch)
        original = DistillationTrainer._reduce_divergence_loss

        def _recording(jsd, completion_mask=None, reduction="batchmean", num_items_in_batch=None):
            # `generalized_jsd_loss(reduction="none")` also routes through here with `completion_mask=None`; only the
            # loss-reducing call (with a mask) carries the denominator under test.
            if completion_mask is not None:
                recorded.append((num_items_in_batch, int(completion_mask.sum())))
            return original(
                jsd, completion_mask=completion_mask, reduction=reduction, num_items_in_batch=num_items_in_batch
            )

        monkeypatch.setattr(DistillationTrainer, "_reduce_divergence_loss", staticmethod(_recording))

        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(gradient_accumulation_steps=2, max_steps=1),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

        assert len(recorded) == 2, "expected one loss reduction per accumulation step"
        denominator = recorded[0][0]
        assert denominator is not None, "the loss was not reduced by a token count"
        # The denominator must be the completion tokens summed over the whole accumulation window.
        assert int(denominator) == sum(tokens for _, tokens in recorded)

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
        # sum / num_items_in_batch rather than the local per-microbatch mean. See issue #4719. The full-vocabulary JSD
        # path routes through `_reduce_divergence_loss`, which must honor `num_items_in_batch`.
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
        # Reconstruction invariant: cat(prompt_mask, completion_mask) == attention_mask.
        assert torch.equal(
            torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1), inputs["attention_mask"]
        )
        # Same valid-token count as the (retained-for-metrics) labels.
        assert int(inputs["completion_mask"].sum()) == int((inputs["labels"] != -100).sum())

    def test_generated_batch_emits_prompt_and_completion_ids(self, monkeypatch):
        """The generated batch emits GRPO-style `prompt_ids`/`prompt_mask`/`completion_ids` alongside the old keys."""
        trainer = self._make_local_trainer()
        captured = {}
        original = DistillationTrainer.compute_loss

        def _capturing(self, model, inputs, *args, **kwargs):
            captured.setdefault("inputs", {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()})
            return original(self, model, inputs, *args, **kwargs)

        monkeypatch.setattr(DistillationTrainer, "compute_loss", _capturing)
        trainer.train()

        inputs = captured["inputs"]
        for key in ("prompt_ids", "prompt_mask", "completion_ids"):
            assert key in inputs
        # cat(prompt_ids, completion_ids) reconstructs input_ids; the new keys mirror the existing prompt tensors.
        assert torch.equal(torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1), inputs["input_ids"])
        assert torch.equal(inputs["prompt_ids"], inputs["prompts"])
        assert torch.equal(inputs["prompt_mask"], inputs["prompt_attention_mask"])

    @require_liger_kernel
    @require_torch_accelerator
    def test_distillation_trainer_with_liger(self):
        import importlib

        training_args = self._make_args(use_liger_kernel=True, use_cpu=False)
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=self.tokenizer,
        )

        try:
            assert trainer.use_liger_loss is True
            trainer.train()
            assert trainer.state.log_history[-1]["train_loss"] is not None
            assert trainer.evaluate()["eval_loss"] is not None
        finally:
            importlib.reload(importlib.import_module(trainer.model.__module__))

    def test_prediction_step_gathers_liger_zero3_lm_head_like_training_step(self, monkeypatch):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        training_args = self._make_args(
            eval_strategy="no",
            per_device_eval_batch_size=2,
        )
        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=self.tokenizer,
        )

        call_count = 0
        original_gather_ctx = trainer._get_liger_zero3_lm_head_gather_ctx

        def counting_gather_ctx(model):
            nonlocal call_count
            call_count += 1
            return original_gather_ctx(model)

        monkeypatch.setattr(trainer, "_get_liger_zero3_lm_head_gather_ctx", counting_gather_ctx)

        trainer.evaluate()
        assert call_count > 0

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


def test_repeat_batch_dataloader_delegates_set_epoch_via_getattr():
    class DummyDataLoader:
        def __init__(self):
            self.epoch = None

        def __iter__(self):
            yield {"x": 1}

        def __len__(self):
            return 1

        def set_epoch(self, epoch):
            self.epoch = epoch

    dataloader = DummyDataLoader()
    wrapper = _RepeatBatchDataLoader(dataloader, repeat_count=2)

    wrapper.set_epoch(7)

    assert dataloader.epoch == 7
