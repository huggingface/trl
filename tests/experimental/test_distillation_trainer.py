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

import math
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.distillation import DistillationConfig, DistillationTrainer
from trl.experimental.distillation.distillation_trainer import (
    _add_tail_bucket,
    _jsd_divergence,
    _RepeatBatchDataLoader,
    build_teacher_request_inputs,
)

from ..testing_utils import TrlTestCase, require_liger_kernel, require_torch_accelerator


def _make_distillation_config_kwargs(tmp_path):
    return {"output_dir": str(tmp_path), "report_to": "none", "use_cpu": True, "bf16": False}


def _build_server_result(teacher_logits, inputs, temperature=1.0):
    """Simulate a vLLM server response with variable-length per-sample completions."""
    _, _, completion_lengths = build_teacher_request_inputs(
        inputs["input_ids"],
        inputs["attention_mask"],
        prompt_attention_mask=inputs.get("prompt_attention_mask"),
        labels=inputs.get("labels"),
    )

    label_mask = inputs["labels"] != -100
    actual_logprobs = []
    logprobs = []
    logprob_token_ids = []

    for i, comp_len in enumerate(completion_lengths):
        if comp_len == 0:
            actual_logprobs.append([])
            logprobs.append([])
            logprob_token_ids.append([])
            continue

        comp_start = int(torch.nonzero(label_mask[i], as_tuple=False)[0].item())
        sample_logits = teacher_logits[i, comp_start - 1 : comp_start - 1 + comp_len, :]
        sample_log_probs = F.log_softmax(sample_logits / temperature, dim=-1)
        comp_tokens = inputs["input_ids"][i, comp_start : comp_start + comp_len]

        top1_ids = sample_logits.argmax(dim=-1, keepdim=True)
        top1_lps = sample_log_probs.gather(dim=-1, index=top1_ids)
        actual_lps = sample_log_probs.gather(dim=-1, index=comp_tokens.unsqueeze(-1))

        actual_logprobs.append(actual_lps.tolist())
        logprobs.append(top1_lps.tolist())
        logprob_token_ids.append(top1_ids.tolist())

    return {
        "actual_logprobs": actual_logprobs,
        "logprobs": logprobs,
        "logprob_token_ids": logprob_token_ids,
    }


class RecordingTeacherClient:
    def __init__(self):
        self.calls = []
        self.result = None

    def get_sequence_logprobs(self, sequences, prompt_lengths, top_logprobs, temperature):
        self.calls.append(
            {
                "sequences": sequences,
                "prompt_lengths": prompt_lengths,
                "top_logprobs": top_logprobs,
                "temperature": temperature,
            }
        )
        return self.result


def _ragged_server_response():
    # Two samples with completion lengths 1 and 3 respectively; matches the wire format
    # of VLLMClient.get_sequence_logprobs (per-sample shape (comp_len, top_k=1)).
    return {
        "logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
        "logprob_token_ids": [[[90]], [[90], [9217], [100]]],
        "actual_logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
    }


def _canned_teacher_logprobs(**kwargs):
    # Fabricate ragged per-sample logprobs matching the requested sequence shapes.
    sequences = kwargs["sequences"]
    prompt_lengths = kwargs["prompt_lengths"]
    top_k = kwargs.get("top_logprobs", 1)
    logprobs, token_ids, actual = [], [], []
    for seq, plen in zip(sequences, prompt_lengths, strict=True):
        comp_len = len(seq) - plen
        logprobs.append([[-1.0 - 0.05 * i] * top_k for i in range(comp_len)])
        token_ids.append([[int(seq[plen + i])] * top_k for i in range(comp_len)])
        actual.append([[-1.0 - 0.05 * i] for i in range(comp_len)])
    return {"logprobs": logprobs, "logprob_token_ids": token_ids, "actual_logprobs": actual}


def _variable_length_dataset():
    return Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4."}]},
            {
                "messages": [
                    {"role": "user", "content": "Name three primary colors."},
                    {
                        "role": "assistant",
                        "content": "Red, green, and blue are the three primary colors commonly used in additive color mixing.",
                    },
                ]
            },
        ]
    )


def test_distillation_config_rejects_liger_with_teacher_server(tmp_path):
    with pytest.raises(ValueError, match="use_liger_kernel=True is not supported with use_teacher_server=True"):
        DistillationConfig(
            **_make_distillation_config_kwargs(tmp_path),
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            use_liger_kernel=True,
        )


def test_distillation_config_rejects_invalid_reverse_kl_top_1_mode(tmp_path):
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode must be one of"):
        DistillationConfig(**_make_distillation_config_kwargs(tmp_path), reverse_kl_top_1_mode="invalid")


def test_distillation_config_rejects_teacher_server_with_reverse_kl_argmax(tmp_path):
    with pytest.raises(ValueError, match="reverse_kl_top_1_mode='argmax' is not supported"):
        DistillationConfig(
            **_make_distillation_config_kwargs(tmp_path),
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            reverse_kl_top_1_mode="argmax",
        )


def test_distillation_config_rejects_teacher_server_mixed_loss_without_top_1(tmp_path):
    with pytest.raises(ValueError, match="loss_top_k must be 1 when using use_teacher_server=True with beta>0"):
        DistillationConfig(
            **_make_distillation_config_kwargs(tmp_path),
            use_teacher_server=True,
            teacher_model_server_url="http://localhost:8000",
            beta=0.5,
            loss_top_k=2,
        )


def test_distillation_config_requires_teacher_server_url(tmp_path):
    with pytest.raises(ValueError, match="teacher_model_server_url must be set when use_teacher_server=True"):
        DistillationConfig(
            **_make_distillation_config_kwargs(tmp_path),
            use_teacher_server=True,
            beta=0.5,
            loss_top_k=1,
        )


@pytest.mark.parametrize(
    (
        "input_ids",
        "attention_mask",
        "prompt_attention_mask",
        "labels",
        "expected_sequences",
        "expected_prompt_lengths",
        "expected_completion_lengths",
    ),
    [
        pytest.param(
            torch.tensor([[0, 0, 10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]], dtype=torch.long),
            torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            None,
            torch.tensor(
                [[-100, -100, -100, -100, -100, 20, 21], [-100, -100, -100, -100, -100, 40, 41]],
                dtype=torch.long,
            ),
            [[10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]],
            [3, 5],
            [2, 2],
            id="variable_prompt_lengths",
        ),
        pytest.param(
            torch.tensor(
                [[0, 0, 10, 11, 12, 20, 21, 0, 0], [30, 31, 32, 33, 34, 40, 41, 0, 0]],
                dtype=torch.long,
            ),
            torch.tensor(
                [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]],
                dtype=torch.long,
            ),
            torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.long),
            torch.tensor(
                [
                    [-100, -100, -100, -100, -100, 20, 21, -100, -100],
                    [-100, -100, -100, -100, -100, 40, 41, -100, -100],
                ],
                dtype=torch.long,
            ),
            [[10, 11, 12, 20, 21], [30, 31, 32, 33, 34, 40, 41]],
            [3, 5],
            [2, 2],
            id="padded_on_policy_completions",
        ),
        pytest.param(
            torch.tensor([[10, 11, 12]], dtype=torch.long),
            torch.tensor([[1, 1, 1]], dtype=torch.long),
            None,
            torch.tensor([[-100, -100, -100]], dtype=torch.long),
            [[10, 11, 12]],
            [3],
            [0],
            id="empty_completion",
        ),
    ],
)
def test_build_teacher_request_inputs(
    input_ids,
    attention_mask,
    prompt_attention_mask,
    labels,
    expected_sequences,
    expected_prompt_lengths,
    expected_completion_lengths,
):
    sequences, prompt_lengths, completion_lengths = build_teacher_request_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_attention_mask=prompt_attention_mask,
        labels=labels,
    )

    assert sequences == expected_sequences
    assert prompt_lengths == expected_prompt_lengths
    assert completion_lengths == expected_completion_lengths


class TestGetTeacherTokenLogprobsFromServer(TrlTestCase):
    def test_variable_lengths_use_neg_inf_sentinel_at_padding(self):
        mock_self = MagicMock()
        mock_self.teacher_client.get_sequence_logprobs = MagicMock(return_value=_ragged_server_response())
        mock_self.loss_top_k = 1
        mock_self.temperature = 1.0

        inputs = {
            "input_ids": torch.tensor([[10, 11, 90, 0, 0], [10, 11, 90, 9217, 100]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 90, -100, -100], [-100, -100, 90, 9217, 100]]),
        }

        out = DistillationTrainer._get_teacher_token_logprobs_from_server(mock_self, inputs, aligned_prompt_length=2)

        assert out["actual_logprobs"].shape == (2, 3)
        assert out["topk_logprobs"].shape == (2, 3, 1)

        # Real completion positions preserved.
        assert out["actual_logprobs"][0, 0].item() == pytest.approx(-2.3, rel=1e-5)
        assert out["actual_logprobs"][1, 0].item() == pytest.approx(-1.1, rel=1e-5)
        assert out["actual_logprobs"][1, 2].item() == pytest.approx(-3.0, rel=1e-5)

        # Sample 0 is 1 token long; positions 1 and 2 are padded with the -inf sentinel.
        assert out["actual_logprobs"][0, 1].item() == float("-inf")
        assert out["actual_logprobs"][0, 2].item() == float("-inf")
        assert out["topk_logprobs"][0, 1, 0].item() == float("-inf")

        # Sample 1 is full-length and fully finite.
        assert torch.isfinite(out["actual_logprobs"][1, :]).all()


class TestServerReverseKLPaddingMask(TrlTestCase):
    def test_mask_keeps_forward_and_backward_finite(self):
        # Simulates the getter's output: sample 0 has completion length 1 (positions 1-2
        # padded with -inf), sample 1 is full-length.
        teacher_topk = torch.tensor(
            [[[-2.3], [float("-inf")], [float("-inf")]], [[-1.1], [-0.4], [-3.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[90, -100, -100], [90, 9217, 100]])

        # Strategy B: neutralise -inf at labels == -100 before the divergence math.
        pad_mask = (labels == -100).unsqueeze(-1)
        zero = torch.zeros((), dtype=teacher_topk.dtype)
        teacher_topk = torch.where(pad_mask, zero, teacher_topk)

        valid_mask = torch.ones_like(teacher_topk, dtype=torch.bool)
        teacher_with_tail, support_mask = _add_tail_bucket(teacher_topk, valid_mask)
        assert torch.isfinite(teacher_with_tail).all()

        raw_student = torch.randn(2, 3, 2, requires_grad=True)
        student_log_probs = F.log_softmax(raw_student, dim=-1)
        loss = _jsd_divergence(student_log_probs, teacher_with_tail, beta=1.0, support_mask=support_mask)
        assert torch.isfinite(loss).all()

        loss.sum().backward()
        assert torch.isfinite(raw_student.grad).all()


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

    def _make_server_trainer(self, **kwargs):
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        return DistillationTrainer(
            model=self.model_id,
            teacher_model=None,
            args=self._make_args(use_teacher_server=True, teacher_model_server_url="http://localhost:8000", **kwargs),
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
        assert train_result.metrics["train_loss"] >= 0.0
        assert "model.safetensors" in os.listdir(self.tmp_dir + "/checkpoint-2")

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

    def test_sampled_mode_matches_between_local_and_external_teachers(self, monkeypatch):
        import trl.generation.vllm_client as vllm_client_module

        teacher_client = RecordingTeacherClient()
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *args, **kwargs: teacher_client)

        local_trainer = self._make_local_trainer(beta=0.5, loss_top_k=1, reverse_kl_top_1_mode="sampled")
        server_trainer = self._make_server_trainer(beta=0.5, loss_top_k=1)

        cpu_inputs = self._make_batch(local_trainer)
        expected_sequences, expected_prompt_lengths, _ = build_teacher_request_inputs(
            cpu_inputs["input_ids"],
            cpu_inputs["attention_mask"],
            prompt_attention_mask=cpu_inputs["prompt_attention_mask"],
            labels=cpu_inputs["labels"],
        )

        local_inputs = self._move_batch_to_device(cpu_inputs, local_trainer.accelerator.device)
        server_inputs = self._move_batch_to_device(cpu_inputs, server_trainer.accelerator.device)

        local_trainer.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = local_trainer.teacher_model(
                input_ids=local_inputs["input_ids"],
                attention_mask=local_inputs["attention_mask"],
            ).logits
            teacher_client.result = _build_server_result(
                teacher_logits,
                local_inputs,
                temperature=local_trainer.temperature,
            )
            local_loss = local_trainer.compute_loss(local_trainer.model, local_inputs)
            server_loss = server_trainer.compute_loss(server_trainer.model, server_inputs)

        assert teacher_client.calls[0]["sequences"] == expected_sequences
        assert teacher_client.calls[0]["prompt_lengths"] == expected_prompt_lengths
        assert teacher_client.calls[0]["top_logprobs"] == 1
        torch.testing.assert_close(local_loss, server_loss)


class TestDistillationTrainerServerPath(TrlTestCase):
    @classmethod
    def setup_class(cls):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model_id = model_id

    def _run_one_step(self, bs, ga, monkeypatch):
        from trl.generation import vllm_client as vllm_client_module

        fake_client = MagicMock()
        fake_client.get_sequence_logprobs.side_effect = _canned_teacher_logprobs
        monkeypatch.setattr(vllm_client_module, "VLLMClient", lambda *a, **kw: fake_client)

        config = DistillationConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
            learning_rate=1e-4,
            max_length=64,
            max_prompt_length=32,
            max_completion_length=32,
            use_teacher_server=True,
            teacher_model_server_url="http://fake-teacher.invalid:8000",
            loss_top_k=1,
            beta=1.0,
            lmbda=0.0,
            loss_add_tail=True,
            save_strategy="no",
            report_to="none",
            logging_steps=1,
            use_cpu=not torch.cuda.is_available(),
            bf16=False,
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32).to(self.device)
        trainer = DistillationTrainer(
            model=model,
            args=config,
            train_dataset=_variable_length_dataset(),
            processing_class=self.tokenizer,
        )
        trainer.teacher_client = fake_client
        trainer.train()
        return [rec for rec in trainer.state.log_history if "grad_norm" in rec]

    @pytest.mark.parametrize(("bs", "ga"), [(1, 2), (2, 1)])
    def test_reverse_kl_finite_grad_with_ragged_batch(self, bs, ga, monkeypatch):
        records = self._run_one_step(bs=bs, ga=ga, monkeypatch=monkeypatch)
        assert records, "Expected at least one grad_norm log entry during training"
        for record in records:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])


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
