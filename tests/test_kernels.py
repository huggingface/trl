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

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers.testing_utils import torch_device

from trl.kernels import ChunkedLogProbFunction, selective_log_softmax_and_entropy

from .testing_utils import require_torch_accelerator


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
class TestChunkedLogProbFunction:
    N, H, V = 64, 32, 128
    CHUNK_SIZE = 32

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

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_forward(self, temperature):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)
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

        _, _, log_sum_sq_probs, _, _ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)
        logits = (hidden @ weight.t()) / 0.7

        expected = torch.logsumexp(2 * logits, dim=-1) - 2 * torch.logsumexp(logits, dim=-1)
        torch.testing.assert_close(log_sum_sq_probs, expected, atol=1e-5, rtol=1e-5)

    def test_mean_logits(self):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        _, _, _, mean_logits, _ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 0.7, self.CHUNK_SIZE)

        torch.testing.assert_close(mean_logits, (hidden @ weight.t()).mean(-1) / 0.7, atol=1e-5, rtol=1e-5)

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

        *_, is_top1 = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)

        torch.testing.assert_close(is_top1, argmax == labels)

    @pytest.mark.parametrize(
        ("logit_scale", "final_logit_softcapping"),
        [
            (0.5, None),  # models that scale but don't softcap, e.g. MPT
            (1.0, 30.0),  # models that softcap but don't scale, e.g. Gemma 2
            (0.5, 30.0),  # both, applied in that order
        ],
    )
    def test_logit_scale_and_softcapping(self, logit_scale, final_logit_softcapping):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, device=torch_device)
        weight = torch.randn(self.V, self.H, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        logprobs, entropy, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, 0.7, self.CHUNK_SIZE, final_logit_softcapping, logit_scale
        )
        logprobs_ref, entropy_ref = self._reference_logprobs_and_entropy(
            hidden, weight, labels, 0.7, logit_scale=logit_scale, final_logit_softcapping=final_logit_softcapping
        )

        torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(entropy, entropy_ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward(self, temperature):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        logprobs_chunked, _, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, temperature, self.CHUNK_SIZE
        )
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

    @pytest.mark.parametrize("temperature", [1.0, 0.7])
    def test_backward_bfloat16(self, temperature):
        torch.manual_seed(42)
        hidden = torch.randn(self.N, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        weight = torch.randn(self.V, self.H, dtype=torch.bfloat16, requires_grad=True, device=torch_device)
        labels = torch.randint(0, self.V, (self.N,), device=torch_device)

        # Chunked backward
        logprobs_chunked, _, *_ = ChunkedLogProbFunction.apply(
            hidden, weight, None, labels, temperature, self.CHUNK_SIZE
        )
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

        logprobs, _, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)
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
            logprobs, _, *_ = ChunkedLogProbFunction.apply(hidden, weight, None, labels, 1.0, self.CHUNK_SIZE)
            perturbed_logprobs, _, *_ = ChunkedLogProbFunction.apply(
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

        logprobs, _, *_ = ChunkedLogProbFunction.apply(hidden, weight, bias, labels, 1.0, self.CHUNK_SIZE)
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

        logprobs, _, *_ = ChunkedLogProbFunction.apply(hidden, weight, bias, labels, 1.0, self.CHUNK_SIZE)
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
