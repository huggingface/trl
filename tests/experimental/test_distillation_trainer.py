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
from transformers import AutoTokenizer

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

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("use_labels", [False, True])
    def test_matches_reference_implementation(self, beta, use_labels):
        labels = self.labels if use_labels else None
        loss = DistillationTrainer.generalized_jsd_loss(
            self.student_logits, self.teacher_logits, labels=labels, beta=beta
        )
        expected = _reference_generalized_jsd(self.student_logits, self.teacher_logits, labels=labels, beta=beta)
        torch.testing.assert_close(loss, expected)

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
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

    @pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
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
            "lmbda": 0.0,
            "max_length": 128,
            "max_completion_length": 32,
            "model_init_kwargs": {"dtype": "float32", "device_map": None},
            "teacher_model_init_kwargs": {"dtype": "float32", "device_map": None},
        }
        args.update(kwargs)
        return DistillationConfig(**args)

    def _make_local_trainer(self, **kwargs):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        return DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(**kwargs),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

    def _make_batch(self, trainer):
        examples = [trainer.train_dataset[i] for i in range(2)]
        return trainer.data_collator(examples)

    @staticmethod
    def _move_batch_to_device(batch, device):
        return {key: value.to(device) for key, value in batch.items()}

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
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling")
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

    @pytest.mark.parametrize("lmbda", [0.0, 1.0])
    def test_train_updates_params_on_and_off_policy(self, lmbda):
        """Pin both policy modes end to end before `lmbda` is removed.

        `lmbda=0.0` trains on the dataset's own completions, `lmbda=1.0` on completions the student generates. The
        trainer is scheduled to become always-on-policy, so the off-policy case is pinned here to make its removal a
        deliberate deletion rather than a silent one.
        """
        # Higher lr than the default: gradients are tiny on this model and the default lr can stall the update, which
        # would make the assertion below vacuous.
        trainer = self._make_local_trainer(lmbda=lmbda, max_steps=2, learning_rate=0.1)
        previous_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        for name, param in previous_params.items():
            assert not torch.equal(param, trainer.model.get_parameter(name)), f"Parameter {name} has not changed."

    @pytest.mark.xfail(
        reason="num_items_in_batch is computed from the raw dataloader batches, before _prepare_inputs runs. "
        "_RepeatBatchDataLoader yields the same generation batch once per accumulation step, so the count is "
        "gradient_accumulation_steps times too large, and it is derived from the dataset labels even on steps whose "
        "completions are replaced by generation. Fixed when the buffer moves to the GRPO-style _prepare_inputs."
    )
    def test_num_items_in_batch_counts_the_tokens_trained_on(self):
        """`num_items_in_batch` is the loss denominator, so it must count the tokens actually trained on.

        The loss is reduced as `sum / num_items_in_batch`, so an inflated count silently scales the loss — and the
        gradient — down. Documented here as a known failure; un-xfail it in the PR that fixes the denominator.
        """
        recorded = []

        class _RecordingTrainer(DistillationTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # Prompt positions are already -100, so this is the completion-token count either way.
                recorded.append((num_items_in_batch, int((inputs["labels"] != -100).sum())))
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        trainer = _RecordingTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=self._make_args(gradient_accumulation_steps=2, max_steps=1),
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()

        assert len(recorded) == 2, "expected one compute_loss call per accumulation step"
        reported = recorded[0][0]
        assert reported is not None, "transformers did not pass num_items_in_batch"
        # The denominator should be the number of tokens summed over the accumulation window.
        assert int(reported) == sum(trained_on for _, trained_on in recorded)

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
        train_dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "conversational_language_modeling", split="test", streaming=streaming
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

        batch = self._move_batch_to_device(self._make_batch(trainer), trainer.accelerator.device)

        # Number of valid (non-ignored) tokens in the local batch, sliced the same way `compute_loss` does.
        prompt_length = trainer._compute_prompt_length(batch)
        num_valid = (batch["labels"][:, prompt_length:] != -100).sum()

        trainer.model.eval()
        with torch.no_grad():
            loss_mean = trainer.compute_loss(trainer.model, batch)  # num_items_in_batch=None -> local mean
            loss_global = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid)
            loss_double = trainer.compute_loss(trainer.model, batch, num_items_in_batch=num_valid * 2)

        # With num_items_in_batch equal to the local valid-token count, sum/N equals the local mean.
        torch.testing.assert_close(loss_global, loss_mean, rtol=1e-4, atol=1e-6)
        # Doubling the global count exactly halves the loss (sum / num_items is linear in 1/num_items).
        torch.testing.assert_close(loss_double, loss_mean / 2, rtol=1e-4, atol=1e-6)

    @require_liger_kernel
    @require_torch_accelerator
    def test_distillation_trainer_with_liger(self):
        import importlib

        training_args = self._make_args(use_liger_kernel=True, use_cpu=False)
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")

        trainer = DistillationTrainer(
            model=self.model_id,
            teacher_model=self.model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        try:
            assert trainer.use_liger_loss is True
            trainer.train()
            assert trainer.state.log_history[-1]["train_loss"] is not None
        finally:
            importlib.reload(importlib.import_module(trainer.model.__module__))


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
