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
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers.testing_utils import torch_device
from transformers.utils import is_triton_available

from .testing_utils import require_torch_accelerator, require_triton


if is_triton_available():
    from trl.kernels import ChunkedLogProbFunction, selective_log_softmax_and_entropy
    from trl.kernels.chunked_divergence import ChunkedDivergenceFunction
    from trl.kernels.chunked_logprob import _addmm_fp32


def reference(logits, index, temperature, row_mask):
    logits = logits.float() / temperature
    logprobs = logits.log_softmax(-1)
    selected_logprobs = logprobs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    entropy = -(logprobs.exp() * logprobs).sum(-1)
    if row_mask is not None:
        selected_logprobs = selected_logprobs.masked_fill(~row_mask, 0.0)
        entropy = entropy.masked_fill(~row_mask, 0.0)
    return selected_logprobs, entropy


@require_torch_accelerator
@require_triton
class TestLogProbEntropy:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        ("logprob_weight", "entropy_weight", "use_mask"),
        [(2, None, True), (None, 0.5, True), (2, 0.5, True), (2, 0.5, False)],
    )
    def test_forward_and_backward(self, dtype, logprob_weight, entropy_weight, use_mask):
        # Cross two full Triton blocks and leave a partial final block.
        vocab_size = 2053
        base_logits = torch.randn(2, 5, vocab_size, device=torch_device, dtype=dtype)
        base_logits[..., :1024].add_(20)
        # Exercise online normalization across blocks with sharply different maxima.
        base_logits[..., 1024:2048].sub_(20)
        base_logits.requires_grad_()
        # TRL slices sequence logits and token IDs without making them contiguous.
        logits = base_logits[:, 1:4]
        index = torch.randint(vocab_size, (2, 5), device=torch_device)[:, 1:4]
        index[0, 0] = vocab_size - 1  # select a target from the partial block
        temperature = 0.7
        # Slice the mask too, matching the non-contiguous views passed by TRL.
        sliced_mask = torch.tensor([[1, 1, 0, 1, 1], [1, 0, 1, 1, 1]], device=torch_device, dtype=torch.bool)[:, 1:4]
        row_mask = sliced_mask if use_mask else None

        logprobs, entropy = selective_log_softmax_and_entropy(
            logits, index, temperature=temperature, row_mask=row_mask
        )
        # Mutating exposed outputs must not corrupt the private statistics saved for backward.
        logprobs[0, 2] = 0.0
        loss = 0.0
        if logprob_weight is not None:
            loss = loss + logprob_weight * logprobs.sum()
        if entropy_weight is not None:
            loss = loss + entropy_weight * entropy.sum()
        loss.backward()
        actual_grad = base_logits.grad.clone()

        reference_logits = base_logits.detach().clone().requires_grad_()
        reference_logprobs, reference_entropy = reference(reference_logits[:, 1:4], index, temperature, row_mask)
        reference_logprobs[0, 2] = 0.0
        reference_loss = 0.0
        if logprob_weight is not None:
            reference_loss = reference_loss + logprob_weight * reference_logprobs.sum()
        if entropy_weight is not None:
            reference_loss = reference_loss + entropy_weight * reference_entropy.sum()
        reference_loss.backward()

        torch.testing.assert_close(logprobs, reference_logprobs, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(entropy, reference_entropy, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_grad, reference_logits.grad, rtol=1e-3, atol=1e-3)

    def test_torch_compile_fullgraph(self):
        logits = torch.randn(2, 3, 257, device=torch_device, dtype=torch.bfloat16, requires_grad=True)
        index = torch.randint(257, (2, 3), device=torch_device)
        row_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], device=torch_device, dtype=torch.bool)

        def loss(logits, index, row_mask):
            logprobs, entropy = selective_log_softmax_and_entropy(logits, index, temperature=0.8, row_mask=row_mask)
            return (logprobs + 0.1 * entropy).sum()

        torch.compile(loss, fullgraph=True)(logits, index, row_mask).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad[~row_mask]) == 0


@require_torch_accelerator
@require_triton
class TestChunkedLogProbFunction:
    N, H, V = 64, 32, 128
    CHUNK_SIZE = 32

    def test_torch_compile(self):
        # Inductor passes the float arguments (temperature, logit scale, softcapping) as fp64
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device, requires_grad=True)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        def loss(hidden):
            logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
                hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, 30.0, 0.5
            )
            return logprobs.sum() + entropy.sum()

        loss(hidden).backward()
        expected_grad, hidden.grad = hidden.grad, None
        torch.compile(loss)(hidden).backward()

        torch.testing.assert_close(hidden.grad, expected_grad)

    def _reference_logprobs_and_entropy(
        self, hidden, weight, labels, temperature, bias=None, logit_scale=1.0, final_logit_softcapping=None
    ):
        logits = hidden @ weight.t()
        if bias is not None:
            logits = logits + bias
        logits = logits.to(torch.float32) * logit_scale
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        logits = logits / temperature  # [N, V]
        log_p = F.log_softmax(logits, dim=-1)
        logprobs = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        p = torch.softmax(logits, dim=-1)
        entropy = -(p * log_p).sum(dim=-1)
        return logprobs, entropy

    # CPU tensors stand in for an accelerator without Triton (e.g. Trainium): TRL does not target CPU training
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_forward(self, device, temperature):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=device)
        weight = torch.randn(self.V, self.H, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)
        chunk_rows = []
        torch_mm = torch.mm

        def record_chunk(input, mat2, *, out=None):
            chunk_rows.append(input.size(0))
            return torch_mm(input, mat2, out=out)

        with (
            patch("trl.kernels.chunked_logprob.TOKEN_CHUNK_SIZE", 17),
            patch("trl.kernels.chunked_logprob.torch.mm", side_effect=record_chunk),
        ):
            logprobs_chunked, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
                hidden, weight, None, labels, temperature, self.CHUNK_SIZE
            )
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)

        torch.testing.assert_close(logprobs_chunked, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy_chunked, entropy_ref, atol=1e-5, rtol=1e-5)
        assert max(chunk_rows) <= 17
        assert chunk_rows[-1] == 13

    def test_log_sum_sq_probs(self):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        _, _, log_sum_sq_probs, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)
        logits = (hidden @ weight.t()) / 0.7

        expected = torch.logsumexp(2 * logits, dim=-1) - 2 * torch.logsumexp(logits, dim=-1)
        torch.testing.assert_close(log_sum_sq_probs, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("outputs", [(), ("entropy",), ("is_top1", "mean_logits")])
    def test_outputs_subset(self, outputs):
        # Outputs that are not requested are `None`, and the others, with the gradients, are unchanged
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device, requires_grad=True)
        weight = torch.randn(self.V, self.H, device=torch_device, requires_grad=True)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        full = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)
        full[0].sum().backward()
        grads = hidden.grad.clone(), weight.grad.clone()
        hidden.grad = weight.grad = None
        subset = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, None, 1.0, outputs)
        subset[0].sum().backward()

        names = ("log_probs", "entropy", "log_sum_sq_probs", "mean_logits", "is_top1", "score_log_probs")
        for name, x, y in zip(names, full, subset, strict=True):
            if name == "log_probs" or name in outputs:
                torch.testing.assert_close(y, x)
            else:
                assert y is None
        torch.testing.assert_close(hidden.grad, grads[0])
        torch.testing.assert_close(weight.grad, grads[1])

    def test_mean_logits(self):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        _, _, _, mean_logits, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)

        torch.testing.assert_close(mean_logits, (hidden @ weight.t()).mean(-1) / 0.7, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize(
        ("temperature", "logit_scale", "final_logit_softcapping"),
        [(1.0, 1.0, None), (0.7, 0.5, None), (0.7, 1.0, 3.0)],
    )
    @pytest.mark.parametrize("entropy_weight", [0.0, 0.5])
    def test_backward_frozen_head(self, temperature, logit_scale, final_logit_softcapping, entropy_weight):
        # Only the hidden states need a gradient, as with a PEFT adapter
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device, requires_grad=True)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, temperature, self.CHUNK_SIZE, final_logit_softcapping, logit_scale
        )
        (2.0 * logprobs + entropy_weight * entropy).sum().backward()
        grad = hidden.grad.clone()

        hidden.grad = None
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(
            hidden,
            weight,
            labels,
            temperature,
            logit_scale=logit_scale,
            final_logit_softcapping=final_logit_softcapping,
        )
        (2.0 * logprobs_ref + entropy_weight * entropy_ref).sum().backward()

        torch.testing.assert_close(grad, hidden.grad, atol=1e-4, rtol=1e-4)

    def test_is_top1_matches_argmax_with_ties(self):
        # Duplicated head rows make exact ties; `argmax` picks the first one, and so must `is_top1`
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        weight[self.V // 2 :] = weight[: self.V - self.V // 2]
        argmax = (hidden @ weight.t()).argmax(-1)
        # Half the labels are the argmax, half its duplicate further down the vocabulary
        labels = torch.where(torch.arange(self.N, device=torch_device) % 2 == 0, argmax, argmax + self.V // 2)
        labels = labels.clamp(max=self.V - 1)

        *_, is_top1, _ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)

        torch.testing.assert_close(is_top1, argmax == labels)

    # CPU tensors stand in for an accelerator without Triton (e.g. Trainium): TRL does not target CPU training
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize(
        ("logit_scale", "final_logit_softcapping"),
        [
            (0.5, None),  # models that scale but don't softcap, e.g. MPT
            (1.0, 30.0),  # models that softcap but don't scale, e.g. Gemma 2
            (0.5, 30.0),  # both, applied in that order
        ],
    )
    def test_logit_scale_and_softcapping(self, device, logit_scale, final_logit_softcapping):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=device)
        weight = torch.randn(self.V, self.H, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)

        logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, final_logit_softcapping, logit_scale
        )
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(
            hidden, weight, labels, 0.7, logit_scale=logit_scale, final_logit_softcapping=final_logit_softcapping
        )

        torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy, entropy_ref, atol=1e-5, rtol=1e-5)

    # CPU tensors stand in for an accelerator without Triton (e.g. Trainium): TRL does not target CPU training
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward(self, device, temperature):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)

        # Chunked backward
        logprobs_chunked, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, temperature, self.CHUNK_SIZE)
        logprobs_chunked.sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        logprobs_ref, _ = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        logprobs_ref.sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize(("logit_scale", "final_logit_softcapping"), [(1.0, None), (0.5, 30.0)])
    def test_support_ids(self, device, logit_scale, final_logit_softcapping):
        # Log-probs and their gradient are normalized over each row's support; the entropy stays full-vocabulary
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)
        sizes = torch.randint(1, 8, (self.N,), device=device)
        support_ids = torch.full((self.N, 8), -1, device=device, dtype=torch.long)
        for row in range(self.N):
            ids = torch.randperm(self.V, device=device)[: sizes[row] - 1]
            support_ids[row, : sizes[row]] = torch.cat([labels[row : row + 1], ids])

        logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
            hidden,
            weight,
            None,
            labels,
            0.7,
            self.CHUNK_SIZE,
            final_logit_softcapping,
            logit_scale,
            ("entropy",),
            support_ids,
        )
        (logprobs.sum() + entropy.sum()).backward()
        grad_hidden, grad_weight = hidden.grad.clone(), weight.grad.clone()
        hidden.grad = weight.grad = None

        logits = hidden @ weight.t() * logit_scale
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        logits = logits / 0.7
        keep = F.one_hot(support_ids + 1, self.V + 1)[..., 1:].any(1)
        logprobs_ref = logits.masked_fill(~keep, float("-inf")).log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        log_p = logits.log_softmax(-1)
        entropy_ref = -(log_p.exp() * log_p).sum(-1)
        (logprobs_ref.sum() + entropy_ref.sum()).backward()

        torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy, entropy_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grad_hidden, hidden.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grad_weight, weight.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    @pytest.mark.parametrize("with_support", [False, True])
    @pytest.mark.parametrize(("logit_scale", "final_logit_softcapping"), [(1.0, None), (0.5, 30.0)])
    def test_score_ids(self, device, with_support, logit_scale, final_logit_softcapping):
        # The scored tokens' log-probs are normalized like the target's, with the gradient of a weighted sum of them
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=device)
        labels = torch.randint(0, self.V, (self.N,), device=device)
        support_ids = torch.full((self.N, 8), -1, device=device, dtype=torch.long)
        for row in range(self.N):
            others = torch.randperm(self.V, device=device)
            others = others[others != labels[row]][: row % 7]
            support_ids[row, : 1 + row % 7] = torch.cat([labels[row : row + 1], others])
        # Half the rows score the label among their tokens, the other half do not
        odd = (torch.arange(self.N, device=device) % 2 == 1).unsqueeze(-1)
        score_ids = torch.where(odd, support_ids[:, 1:5], support_ids[:, :4])
        coef = torch.randn(self.N, 4, device=device)

        logprobs, entropy, *_, score_log_probs = ChunkedLogProbFunction.apply(
            hidden,
            weight,
            None,
            labels,
            0.7,
            self.CHUNK_SIZE,
            final_logit_softcapping,
            logit_scale,
            ("entropy",),
            support_ids if with_support else None,
            score_ids,
        )
        (logprobs.sum() + entropy.sum() + (coef * score_log_probs).sum()).backward()
        grad_hidden, grad_weight = hidden.grad.clone(), weight.grad.clone()
        hidden.grad = weight.grad = None

        logits = hidden @ weight.t() * logit_scale
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
        logits = logits / 0.7
        full_log_p = logits.log_softmax(-1)
        entropy_ref = -(full_log_p.exp() * full_log_p).sum(-1)
        if with_support:
            keep = F.one_hot(support_ids + 1, self.V + 1)[..., 1:].any(1)
            logits = logits.masked_fill(~keep, float("-inf"))
        log_p = logits.log_softmax(-1)
        logprobs_ref = log_p.gather(-1, labels[:, None]).squeeze(-1)
        score_log_probs_ref = log_p.gather(-1, score_ids.clamp(min=0)).masked_fill(score_ids < 0, 0.0)
        (logprobs_ref.sum() + entropy_ref.sum() + (coef * score_log_probs_ref).sum()).backward()

        torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(score_log_probs, score_log_probs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(grad_hidden, hidden.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grad_weight, weight.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_bfloat16(self, temperature):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        logprobs_chunked, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, temperature, self.CHUNK_SIZE)
        logprobs_chunked.sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        logprobs_ref, _ = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        logprobs_ref.sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-2, rtol=1e-2)

    def test_backward_bfloat16_hidden_float32_weight(self):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=torch.float32, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)
        logprobs.sum().backward()
        grad_hidden = hidden.grad.clone()
        grad_weight = weight.grad.clone()

        hidden.grad = None
        weight.grad = None
        reference, _ = self._reference_logprobs_and_entropy(hidden, weight.to(hidden.dtype), labels, 1.0)
        reference.sum().backward()

        torch.testing.assert_close(logprobs, reference, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(grad_hidden, hidden.grad, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(grad_weight, weight.grad, atol=1e-2, rtol=1e-2)

    def test_backward_uses_autocast_hidden_for_weight_gradient(self):
        torch.manual_seed(42)
        hidden = (
            torch.linspace(1.0, 2.0, self.N * self.H, device=torch_device)
            .reshape(self.N, self.H)
            .to(torch.bfloat16)
            .float()
        )
        perturbed_hidden = hidden + 1e-4
        assert torch.equal(hidden.to(torch.bfloat16), perturbed_hidden.to(torch.bfloat16))
        hidden.requires_grad_()
        perturbed_hidden.requires_grad_()
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        perturbed_weight = weight.detach().clone().requires_grad_()
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Autocast gives both inputs identical projected values. Their weight gradients must therefore also match;
        # using the original fp32 hidden states in backward would make the gradients depend on the discarded bits.
        with torch.autocast(torch_device, dtype=torch.bfloat16):
            logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)
            perturbed_logprobs, *_ = ChunkedLogProbFunction.apply(
                perturbed_hidden, perturbed_weight, None, labels, 1.0, self.CHUNK_SIZE
            )
        logprobs.sum().backward()
        perturbed_logprobs.sum().backward()

        torch.testing.assert_close(logprobs, perturbed_logprobs, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, perturbed_weight.grad, rtol=0, atol=0)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_entropy(self, temperature):
        """Backprop through the `entropy` output alone (as opposed to `logprobs`, covered above)."""
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        _, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, temperature, self.CHUNK_SIZE
        )
        entropy_chunked.sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        _, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        entropy_ref.sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_combined(self, temperature):
        """Backprop through `logprobs` and `entropy` together, to catch the gradients overwriting each other
        instead of accumulating."""
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        with patch("trl.kernels.chunked_logprob.TOKEN_CHUNK_SIZE", 17):
            logprobs_chunked, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
                hidden, weight, None, labels, temperature, self.CHUNK_SIZE
            )
            (2.0 * logprobs_chunked + 0.5 * entropy_chunked).sum().backward()
        grad_hidden_chunked = hidden.grad.clone()
        grad_weight_chunked = weight.grad.clone()

        hidden.grad = None
        weight.grad = None

        # Reference backward
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, temperature)
        (2.0 * logprobs_ref + 0.5 * entropy_ref).sum().backward()

        torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bias(self, dtype):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=dtype, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=dtype, requires_grad=True, device=torch_device)
        bias = torch.randn(self.V, dtype=dtype, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs_chunked, entropy_chunked, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, bias, labels, 0.7, self.CHUNK_SIZE
        )
        (2.0 * logprobs_chunked + 0.5 * entropy_chunked).sum().backward()
        chunked_grads = hidden.grad.clone(), weight.grad.clone(), bias.grad.clone()

        hidden.grad = weight.grad = bias.grad = None
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(hidden, weight, labels, 0.7, bias)
        (2.0 * logprobs_ref + 0.5 * entropy_ref).sum().backward()

        atol, rtol = (5e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        torch.testing.assert_close(logprobs_chunked, logprobs_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(entropy_chunked, entropy_ref, atol=atol, rtol=rtol)
        for actual, expected in zip(chunked_grads, (hidden.grad, weight.grad, bias.grad), strict=True):
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_backward_skips_frozen_parameter_gradients(self):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        bias = torch.randn(self.V, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, bias, labels, 1.0, self.CHUNK_SIZE)
        with patch.object(torch, "zeros", wraps=torch.zeros) as mock_zeros:
            logprobs.sum().backward()

        # A frozen LM head should not allocate full-vocabulary gradient buffers. These shapes are distinct from the
        # per-token accumulators and the hidden-state gradient allocated by the same backward pass.
        allocated_shapes = [call.args[0] for call in mock_zeros.call_args_list]
        assert weight.shape not in allocated_shapes
        assert bias.shape not in allocated_shapes
        assert hidden.grad is not None

    @pytest.mark.parametrize("requires_grad", [(True, False, False), (False, True, False), (False, False, True)])
    def test_backward_with_partially_frozen_inputs(self, requires_grad):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=requires_grad[0], device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=requires_grad[1], device=torch_device)
        bias = torch.randn(self.V, requires_grad=requires_grad[2], device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, *_ = ChunkedLogProbFunction.apply(hidden, weight, bias, labels, 1.0, self.CHUNK_SIZE)
        logprobs.sum().backward()
        chunked_grads = hidden.grad, weight.grad, bias.grad

        hidden_ref = hidden.detach().clone().requires_grad_(requires_grad[0])
        weight_ref = weight.detach().clone().requires_grad_(requires_grad[1])
        bias_ref = bias.detach().clone().requires_grad_(requires_grad[2])
        logprobs_ref, _ = self._reference_logprobs_and_entropy(hidden_ref, weight_ref, labels, 1.0, bias_ref)
        logprobs_ref.sum().backward()

        for actual, expected in zip(chunked_grads, (hidden_ref.grad, weight_ref.grad, bias_ref.grad), strict=True):
            if expected is not None:
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_addmm_fp32(self, dtype):
        # Accumulates the product in fp32, without rounding it to `dtype` first
        torch.manual_seed(42)
        a = torch.randn(64, 256, device=torch_device, dtype=dtype)
        b = torch.randn(256, 32, device=torch_device, dtype=dtype)
        acc = torch.randn(64, 32, device=torch_device)
        expected = acc.double() + a.double() @ b.double()

        _addmm_fp32(acc, a, b)

        torch.testing.assert_close(acc.double(), expected, atol=1e-3, rtol=1e-4)


@require_torch_accelerator
@require_triton
class TestChunkedDivergenceFunction:
    @pytest.mark.parametrize("beta", [0.0, 0.7, 1.0])
    def test_torch_compile(self, beta):
        # Inductor passes the float arguments (beta, temperature, logit scales, softcapping) as fp64
        torch.manual_seed(42)
        student_hidden = torch.randn(64, 32, device=torch_device, requires_grad=True)
        student_weight = torch.randn(128, 32, device=torch_device)
        teacher_hidden = torch.randn(64, 48, device=torch_device)
        teacher_weight = torch.randn(128, 48, device=torch_device)

        def loss(student_hidden):
            divergence, _ = ChunkedDivergenceFunction.apply(
                student_hidden, student_weight, None, teacher_hidden, teacher_weight, None, beta, 0.7, 0.5, 30.0
            )
            return divergence.sum()

        loss(student_hidden).backward()
        expected_grad, student_hidden.grad = student_hidden.grad, None
        torch.compile(loss)(student_hidden).backward()

        torch.testing.assert_close(student_hidden.grad, expected_grad)


class TestHeadLossNumerics:
    @pytest.mark.parametrize("device", ["cpu", torch_device])
    def test_confident_target_across_triton_blocks(self, device):
        from trl.kernels.chunked_logprob import ChunkedLogProbFunction

        targets = torch.tensor([0, 1024, 2052], device=device)
        hidden = torch.eye(3, device=device, requires_grad=True)
        weight = torch.full((2053, 3), -9.0, device=device)
        weight[targets, torch.arange(3, device=device)] = 20.0
        weight.requires_grad_()
        *_, score_logps = ChunkedLogProbFunction.apply(
            hidden, weight, None, targets.roll(1), 1.0, 2053, None, 1.0, (), None, targets.unsqueeze(-1)
        )
        (-score_logps.sum()).backward()
        # Even fp64 cross-entropy loses relative precision at this margin; use the analytic loss and gradient.
        tail_mass = 2052 * math.exp(-29.0)
        expected_loss = torch.tensor(3 * math.log1p(tail_mass), device=device, dtype=torch.float64)
        expected_grad = torch.full((2053, 3), math.exp(-29.0) / (1 + tail_mass), device=device, dtype=torch.float64)
        expected_grad[targets, torch.arange(3, device=device)] = -tail_mass / (1 + tail_mass)
        torch.testing.assert_close(-score_logps.double().sum(), expected_loss, atol=0, rtol=1e-5)
        torch.testing.assert_close(weight.grad.double(), expected_grad, atol=0, rtol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("device", ["cpu", torch_device])
    @pytest.mark.parametrize("chunk_size", [2, 4])
    @pytest.mark.parametrize("use_support", [False, True])
    def test_confident_target(self, device, chunk_size, use_support, dtype):
        # The target can occur before or after the other logits in the streaming reduction.
        from trl.kernels.chunked_logprob import ChunkedLogProbFunction

        hidden = torch.eye(3, device=device, dtype=dtype, requires_grad=True)
        weight = torch.tensor(
            [[9.0, -9.0, -9.0], [-9.0, 9.0, -9.0], [-9.0, -9.0, 9.0]], device=device, dtype=dtype, requires_grad=True
        )
        targets = torch.arange(3, device=device)
        support = torch.tensor([[0, 1, -1], [1, 2, -1], [0, 2, -1]], device=device) if use_support else None
        score_ids = torch.stack((targets, torch.full_like(targets, -1)), -1)
        logps, *_, score_logps = ChunkedLogProbFunction.apply(
            hidden, weight, None, targets, 1.0, chunk_size, None, 1.0, (), support, score_ids
        )
        advantages = torch.tensor([0.5, -1.0, 2.0], device=device)
        torch.testing.assert_close(score_logps[:, 0], logps, atol=0, rtol=1e-6)
        loss = -((logps + score_logps[:, 0]) * advantages).sum()
        loss.backward()
        logits = (hidden.detach().double() @ weight.detach().double().t()).requires_grad_()
        if use_support:
            keep = F.one_hot(support + 1, 4)[..., 1:].any(1)
            ref_logps = logits.masked_fill(~keep, -torch.inf).log_softmax(-1)
        else:
            ref_logps = logits.log_softmax(-1)
        ref_loss = -2 * (ref_logps.diagonal() * advantages.double()).sum()
        ref_loss.backward()
        torch.testing.assert_close(loss.double(), ref_loss, atol=0, rtol=1e-6)
        torch.testing.assert_close(
            weight.grad.double(), logits.grad.t(), atol=0, rtol=1e-2 if dtype == torch.bfloat16 else 1e-6
        )

    @pytest.mark.parametrize("chunk_size", [2, 4])
    @pytest.mark.parametrize("device", ["cpu", torch_device])
    @pytest.mark.parametrize("beta", [0.0, 0.7, 1.0])
    @pytest.mark.parametrize("perturbation", [0.0, 1e-7])
    def test_near_teacher(self, device, beta, perturbation, chunk_size):
        from trl.kernels.chunked_divergence import ChunkedDivergenceFunction

        hidden = torch.eye(2, device=device, requires_grad=True)
        teacher = torch.tensor([[2.0, 1.0, 0.0], [2.0, 0.01, 0.0]], device=device).t()
        delta = torch.tensor([[-perturbation, perturbation, 0.0], [0.0, perturbation / 10, 0.0]], device=device).t()
        student = (teacher + delta).requires_grad_()
        with patch("trl.kernels.chunked_divergence.VOCAB_CHUNK_SIZE", chunk_size):
            divergence, _ = ChunkedDivergenceFunction.apply(
                hidden, student, None, hidden.detach(), teacher, None, beta
            )
            divergence.sum().backward()
        logits = student.detach().double().t().requires_grad_()
        log_q = logits.log_softmax(-1)
        log_p = teacher.double().t().log_softmax(-1)
        if beta == 0:
            ref = (log_p.exp() * (log_p - log_q)).sum()
        elif beta == 1:
            ref = (log_q.exp() * (log_q - log_p)).sum()
        else:
            log_m = torch.logaddexp(log_q + math.log(1 - beta), log_p + math.log(beta))
            ref = ((1 - beta) * log_q.exp() * (log_q - log_m) + beta * log_p.exp() * (log_p - log_m)).sum()
        ref.backward()
        if perturbation == 0:
            assert torch.count_nonzero(student.grad) == 0
        else:
            torch.testing.assert_close(student.grad.double().t(), logits.grad, atol=1e-15, rtol=1e-5)

    @pytest.mark.parametrize("chunk_size", [2, 4])
    @pytest.mark.parametrize("device", ["cpu", torch_device])
    @pytest.mark.parametrize("beta", [0.0, 0.7, 1.0])
    def test_concentrated_teacher(self, device, beta, chunk_size):
        from trl.kernels.chunked_divergence import ChunkedDivergenceFunction

        hidden = torch.eye(3, device=device)
        student = (
            torch.tensor([[9.0, -9.0, -9.0], [9.0, -9.0, -9.0], [20.0, -20.0, -20.0]], device=device)
            .t()
            .requires_grad_()
        )
        teacher = torch.tensor([[40.0, -40.0, -40.0], [-9.0, 9.0, -9.0], [-20.0, 20.0, -20.0]], device=device).t()
        with patch("trl.kernels.chunked_divergence.VOCAB_CHUNK_SIZE", chunk_size):
            divergence, _ = ChunkedDivergenceFunction.apply(hidden, student, None, hidden, teacher, None, beta)
            divergence.sum().backward()
        # Use the paper's normalization even for the float64 reference.
        logits = student.detach().double().t().requires_grad_()
        is_max = torch.arange(3, device=device) == logits.argmax(-1, keepdim=True)
        shifted = (logits - logits.max(-1, keepdim=True).values).masked_fill(is_max, 0.0)
        log_q = shifted - F.softplus(shifted.masked_fill(is_max, -torch.inf).logsumexp(-1, keepdim=True))
        log_p = teacher.double().t().log_softmax(-1)
        if beta == 0:
            ref = (log_p.exp() * (log_p - log_q)).sum()
        elif beta == 1:
            ref = (log_q.exp() * (log_q - log_p)).sum()
        else:
            log_m = torch.logaddexp(log_q + math.log(1 - beta), log_p + math.log(beta))
            ref = ((1 - beta) * log_q.exp() * (log_q - log_m) + beta * log_p.exp() * (log_p - log_m)).sum()
        ref.backward()
        torch.testing.assert_close(student.grad.double().t(), logits.grad, atol=1e-22, rtol=1e-5)
        torch.testing.assert_close(divergence.double().sum(), ref, atol=0, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", torch_device])
    @pytest.mark.parametrize("chunk_size", [2, 4])
    @pytest.mark.parametrize("beta", [0.0, 0.7, 1.0])
    def test_independent_logit_shifts(self, device, chunk_size, beta):
        from trl.kernels.chunked_divergence import ChunkedDivergenceFunction

        epsilon = 2**-20
        hidden = torch.eye(2, device=device)
        student = torch.tensor([[32.0, -epsilon], [32.0, 0.0], [32.0, epsilon]], device=device, requires_grad=True)
        teacher = torch.tensor([[-epsilon, 32.0], [0.0, 32.0], [epsilon, 32.0]], device=device)
        with patch("trl.kernels.chunked_divergence.VOCAB_CHUNK_SIZE", chunk_size):
            loss, _ = ChunkedDivergenceFunction.apply(hidden, student, None, hidden, teacher, None, beta)
            loss.sum().backward()
        logits = student.detach().double().t().requires_grad_()
        log_q = logits.log_softmax(-1)
        log_p = teacher.double().t().log_softmax(-1)
        if beta == 0:
            reference = (log_p.exp() * (log_p - log_q)).sum()
        elif beta == 1:
            reference = (log_q.exp() * (log_q - log_p)).sum()
        else:
            log_m = torch.logaddexp(log_q + math.log1p(-beta), log_p + math.log(beta))
            reference = ((1 - beta) * log_q.exp() * (log_q - log_m) + beta * log_p.exp() * (log_p - log_m)).sum()
        reference.backward()
        torch.testing.assert_close(student.grad.double().t(), logits.grad, atol=1e-13, rtol=1e-5)

    @pytest.mark.parametrize("device", ["cpu", torch_device])
    @pytest.mark.parametrize("chunk_size", [1, 2])
    @pytest.mark.parametrize("beta", [1e-6, 1 - 1e-6])
    def test_jsd_near_endpoint(self, device, chunk_size, beta):
        from trl.kernels.chunked_divergence import ChunkedDivergenceFunction

        beta = float(torch.tensor(beta, dtype=torch.float32))
        hidden = torch.ones(1, 1, device=device)
        student = torch.tensor([[-40.0], [40.0]], device=device, requires_grad=True)
        teacher = -student.detach()
        with patch("trl.kernels.chunked_divergence.VOCAB_CHUNK_SIZE", chunk_size):
            loss, _ = ChunkedDivergenceFunction.apply(hidden, student, None, hidden, teacher, None, beta)
            loss.sum().backward()
        # Disjoint distributions approach the binary entropy of the mixture weight.
        expected = torch.tensor(-beta * math.log(beta) - (1 - beta) * math.log1p(-beta), device=device)
        torch.testing.assert_close(loss.sum(), expected, atol=0, rtol=1e-5)
