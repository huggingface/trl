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
from transformers.testing_utils import torch_device

from .testing_utils import require_torch_accelerator


@pytest.fixture(scope="module")
def trl_losses():
    return pytest.importorskip("trl.kernels", reason="test requires triton")


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
    def test_forward_and_backward(self, trl_losses, dtype, logprob_weight, entropy_weight, use_mask):
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

        logprobs, entropy = trl_losses.selective_log_softmax_and_entropy(
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

    def test_torch_compile_fullgraph(self, trl_losses):
        logits = torch.randn(2, 3, 257, device=torch_device, dtype=torch.bfloat16, requires_grad=True)
        index = torch.randint(257, (2, 3), device=torch_device)
        row_mask = torch.tensor([[1, 0, 1], [0, 1, 1]], device=torch_device, dtype=torch.bool)

        def loss(logits, index, row_mask):
            logprobs, entropy = trl_losses.selective_log_softmax_and_entropy(
                logits, index, temperature=0.8, row_mask=row_mask
            )
            return (logprobs + 0.1 * entropy).sum()

        torch.compile(loss, fullgraph=True)(logits, index, row_mask).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.count_nonzero(logits.grad[~row_mask]) == 0
