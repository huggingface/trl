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

"""Tests for trl.experimental.distillation.DistillationTrainer.

Regression guards for the server-backed path (``use_teacher_server=True``). Historically,
``_get_teacher_token_logprobs_from_server`` filled padding positions for shorter samples
in a batch with ``float('-inf')``. That sentinel flows through ``_add_tail_bucket`` into
teacher distributions ``[-inf, 0]``, and through ``_jsd_divergence`` produces ``+inf`` in
the forward pass (``nan_to_num`` clamps to ``torch.finfo(dtype).max``) but leaks NaN
into the backward pass. The symptom observed in practice is finite loss values with
``grad_norm == nan`` whenever ``per_device_train_batch_size * gradient_accumulation_steps
> 1`` coincides with per-sample completion lengths that differ within a batch.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.distillation import DistillationConfig, DistillationTrainer
from trl.experimental.distillation.distillation_trainer import (
    _add_tail_bucket,
    _jsd_divergence,
)

from ..testing_utils import TrlTestCase


TINY_MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


# ---------------------------------------------------------------------------
# T1 — Unit: _get_teacher_token_logprobs_from_server padding uses `-inf` sentinel.
# ---------------------------------------------------------------------------


def _teacher_response_ragged():
    """Canned response: sample 0 has 1 completion token, sample 1 has 3 (ragged).

    Per-sample shape convention matches the real ``VLLMClient.get_sequence_logprobs``
    contract: ``logprobs[i]`` and ``actual_logprobs[i]`` are ``(comp_len, top_k)``
    2-D nested lists (with ``top_k == 1`` for ``actual_logprobs``).
    """
    return {
        "logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
        "logprob_token_ids": [[[90]], [[90], [9217], [100]]],
        "actual_logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
    }


def _ragged_inputs():
    """Two samples. Sample 0: completion length 1 (pad positions 1-2).

    Sample 1: completion length 3. Prompts padded to length 2. Labels use ``-100``
    for prompt and padding positions so ``build_teacher_request_inputs`` can derive
    per-sample completion lengths.
    """
    return {
        "input_ids": torch.tensor([[10, 11, 90, 0, 0], [10, 11, 90, 9217, 100]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        "labels": torch.tensor(
            [[-100, -100, 90, -100, -100], [-100, -100, 90, 9217, 100]]
        ),
    }


def _bind_method_self(teacher_response, loss_top_k=1, temperature=1.0):
    """Return a bare object exposing the attributes the method under test reads."""
    obj = MagicMock()
    client = MagicMock()
    client.get_sequence_logprobs = MagicMock(return_value=teacher_response)
    obj.teacher_client = client
    obj.loss_top_k = loss_top_k
    obj.temperature = temperature
    return obj


def test_server_logprobs_variable_lengths_place_neg_inf_sentinel_at_padding():
    """The server-path getter keeps the ``-inf`` sentinel at intra-batch padding.

    Shorter samples in a variable-length batch pad the tail with ``-inf`` — the TRL
    house sentinel for "no teacher data at this position". The defensive masking that
    keeps the loss backward finite lives downstream in
    ``_compute_server_sparse_top_1_divergence_loss``; this test pins the sentinel
    contract at the getter's output so a future refactor does not silently change it
    without updating the downstream consumers.
    """
    mock_self = _bind_method_self(_teacher_response_ragged())
    inputs = _ragged_inputs()

    out = DistillationTrainer._get_teacher_token_logprobs_from_server(
        mock_self, inputs, aligned_prompt_length=2
    )

    assert out["actual_logprobs"].shape == (2, 3)
    assert out["topk_logprobs"].shape == (2, 3, 1)
    assert out["topk_token_ids"].shape == (2, 3, 1)

    actual = out["actual_logprobs"]
    topk = out["topk_logprobs"]

    # Real completion positions preserved.
    assert actual[0, 0].item() == pytest.approx(-2.3, rel=1e-5)
    assert actual[1, 0].item() == pytest.approx(-1.1, rel=1e-5)
    assert actual[1, 1].item() == pytest.approx(-0.4, rel=1e-5)
    assert actual[1, 2].item() == pytest.approx(-3.0, rel=1e-5)

    # Sample 0 is 1 token long; positions 1 and 2 are the padded tail — sentinel expected.
    assert actual[0, 1].item() == float("-inf"), "padding must carry the -inf sentinel"
    assert actual[0, 2].item() == float("-inf"), "padding must carry the -inf sentinel"
    assert topk[0, 1, 0].item() == float("-inf")
    assert topk[0, 2, 0].item() == float("-inf")

    # Sample 1 is full-length; no padding positions — every value finite.
    assert torch.isfinite(actual[1, :]).all()
    assert torch.isfinite(topk[1, :, :]).all()


def test_reverse_kl_padding_mask_keeps_forward_and_backward_finite():
    """Pins the Strategy-B masking pattern used in ``_compute_server_sparse_top_1_divergence_loss``.

    Given ``-inf``-padded teacher tensors at known padding positions (``labels == -100``),
    replacing the sentinel with finite zeros before ``_add_tail_bucket`` and
    ``_jsd_divergence`` must produce finite forward AND finite gradients. Without this
    mask the reverse-KL server path produces ``grad_norm=nan`` whenever
    ``bs * grad_accum > 1`` coincides with per-sample completion lengths that differ.
    """
    # Simulate the getter's output: sample 0 has completion length 1 (indices 1-2 padded
    # with -inf), sample 1 has completion length 3 (full).
    teacher_topk = torch.tensor(
        [[[-2.3], [float("-inf")], [float("-inf")]], [[-1.1], [-0.4], [-3.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[90, -100, -100], [90, 9217, 100]])

    # Apply the defensive mask pattern (mirror of _compute_server_sparse_top_1_divergence_loss).
    required = labels != -100
    pad_mask_2d = ~required
    pad_mask_3d = pad_mask_2d.unsqueeze(-1)
    zero = torch.zeros((), dtype=teacher_topk.dtype)
    teacher_topk = torch.where(pad_mask_3d, zero, teacher_topk)

    valid_mask = torch.ones_like(teacher_topk, dtype=torch.bool)
    teacher_with_tail, mask_with_tail = _add_tail_bucket(teacher_topk, valid_mask)
    assert torch.isfinite(teacher_with_tail).all(), (
        f"_add_tail_bucket must be finite post-mask; got {teacher_with_tail}"
    )

    raw_student = torch.randn(2, 3, 2, requires_grad=True)
    student_log_probs = torch.nn.functional.log_softmax(raw_student, dim=-1)

    loss_elems = _jsd_divergence(
        student_log_probs,
        teacher_with_tail,
        beta=1.0,
        support_mask=mask_with_tail,
    )
    assert torch.isfinite(loss_elems).all(), (
        f"Forward produced non-finite values: {loss_elems}"
    )

    loss_elems.sum().backward()
    assert raw_student.grad is not None
    assert torch.isfinite(raw_student.grad).all(), (
        f"Backward produced non-finite gradients: {raw_student.grad}"
    )


# ---------------------------------------------------------------------------
# T2/T3 — Functional: end-to-end .train() with server-backed path and
#   ``per_device_train_batch_size * gradient_accumulation_steps == 2`` against
#   per-sample completion lengths that differ. ``lmbda=0.0`` keeps the run
#   off-policy so no student-side vLLM is required.
# ---------------------------------------------------------------------------


def _variable_length_dataset():
    """Two samples whose assistant turns tokenise to clearly different lengths."""
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "4."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Name three primary colors."},
                    {
                        "role": "assistant",
                        "content": (
                            "Red, green, and blue are the three primary colors "
                            "commonly used in additive color mixing."
                        ),
                    },
                ]
            },
        ]
    )


class _CannedTeacherLogprobs:
    """Side effect that returns fabricated ragged logprobs matching the request shape."""

    def __call__(self, **kwargs):
        sequences = kwargs["sequences"]
        prompt_lengths = kwargs["prompt_lengths"]
        top_k = kwargs.get("top_logprobs", 1)

        logprobs: list[list[list[float]]] = []
        logprob_token_ids: list[list[list[int]]] = []
        actual_logprobs: list[list[list[float]]] = []
        for seq, plen in zip(sequences, prompt_lengths, strict=True):
            comp_len = len(seq) - plen
            row_lp = [[-1.0 - 0.05 * i] * top_k for i in range(comp_len)]
            row_tids = [[int(seq[plen + i])] * top_k for i in range(comp_len)]
            row_actual = [[-1.0 - 0.05 * i] for i in range(comp_len)]
            logprobs.append(row_lp)
            logprob_token_ids.append(row_tids)
            actual_logprobs.append(row_actual)

        return {
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
            "actual_logprobs": actual_logprobs,
        }


class TestDistillationTrainerServerPathVariableCompletion(TrlTestCase):
    """End-to-end regression: server-backed training under bs*ga>1 with ragged batches."""

    @classmethod
    def setup_class(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def _run_one_training_step(self, beta, monkeypatch):
        """Construct trainer with patched VLLMClient, run ``.train()``, return step logs.

        Returns the list of log-history records that contain ``grad_norm`` (logged by
        HF Trainer at each optim step, before the post-step ``zero_grad`` nulls the
        per-parameter ``.grad`` tensors).
        """
        from trl.generation import vllm_client as vllm_client_module

        fake_client = MagicMock()
        fake_client.get_sequence_logprobs.side_effect = _CannedTeacherLogprobs()

        def _fake_vllm_client_ctor(*args, **kwargs):
            return fake_client

        monkeypatch.setattr(vllm_client_module, "VLLMClient", _fake_vllm_client_ctor)

        config = DistillationConfig(
            output_dir=self.tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            max_length=64,
            max_prompt_length=32,
            max_completion_length=32,
            use_teacher_server=True,
            teacher_model_server_url="http://fake-teacher.invalid:8000",
            loss_top_k=1,
            beta=beta,
            lmbda=0.0,
            loss_add_tail=True,
            bf16=False,
            save_strategy="no",
            report_to="none",
            logging_steps=1,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID, dtype=torch.float32).to(device)

        trainer = DistillationTrainer(
            model=model,
            args=config,
            train_dataset=_variable_length_dataset(),
            processing_class=self.tokenizer,
        )
        trainer.teacher_client = fake_client
        trainer.train()

        return [rec for rec in trainer.state.log_history if "grad_norm" in rec]

    def test_reverse_kl_finite_grad_under_ga2_with_ragged_batch(self, monkeypatch):
        """``beta=1`` (reverse KL): exactly the path used by server-backed GKD recipes."""
        step_logs = self._run_one_training_step(beta=1.0, monkeypatch=monkeypatch)
        assert step_logs, "expected at least one training step to have been logged"
        for record in step_logs:
            grad_norm = record["grad_norm"]
            loss = record["loss"]
            assert math.isfinite(grad_norm), (
                f"grad_norm={grad_norm} is not finite (loss={loss}); the -inf sentinel "
                f"leaked into the backward pass."
            )
            assert math.isfinite(loss), f"loss={loss} is not finite"

    def test_jsd_finite_grad_under_ga2_with_ragged_batch(self, monkeypatch):
        """``beta=0.5`` (JSD): touches the mixture ``clamp_min(tiny)`` path."""
        step_logs = self._run_one_training_step(beta=0.5, monkeypatch=monkeypatch)
        assert step_logs, "expected at least one training step to have been logged"
        for record in step_logs:
            grad_norm = record["grad_norm"]
            loss = record["loss"]
            assert math.isfinite(grad_norm), (
                f"grad_norm={grad_norm} is not finite (loss={loss}); the -inf sentinel "
                f"leaked into the backward pass."
            )
            assert math.isfinite(loss), f"loss={loss} is not finite"
