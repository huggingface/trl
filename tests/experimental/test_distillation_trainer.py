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
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.distillation import DistillationConfig, DistillationTrainer
from trl.experimental.distillation.distillation_trainer import _add_tail_bucket, _jsd_divergence

from ..testing_utils import TrlTestCase


def _ragged_server_response():
    # Two samples with completion lengths 1 and 3 respectively; matches the wire format
    # of VLLMClient.get_sequence_logprobs (per-sample shape (comp_len, top_k=1)).
    return {
        "logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
        "logprob_token_ids": [[[90]], [[90], [9217], [100]]],
        "actual_logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
    }


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

    @pytest.mark.slow
    @pytest.mark.parametrize(("bs", "ga"), [(1, 2), (2, 1)])
    def test_reverse_kl_finite_grad_with_ragged_batch(self, bs, ga, monkeypatch):
        records = self._run_one_step(bs=bs, ga=ga, monkeypatch=monkeypatch)
        assert records, "Expected at least one grad_norm log entry during training"
        for record in records:
            assert math.isfinite(record["grad_norm"]), f"grad_norm={record['grad_norm']} leaked -inf into backward"
            assert math.isfinite(record["loss"])
