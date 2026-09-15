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

import pytest
import torch
import torch.nn.functional as F
from transformers import set_seed
from transformers.testing_utils import torch_device

from trl.losses import FusedLinearJSDLoss
from trl.losses.jsd_loss import FusedLinearJSDFunction

from .testing_utils import HFDistillationLoss, assert_verbose_allclose


# These reference-vs-fused parity suites compile ~1000 variants and compare in fp32 at tight tolerances; on the
# CI lanes single elements land 1e-5 outside tolerance depending on which variants ran before on the worker
# (deterministic per lane, passes in isolation). Nightly only.
pytestmark = pytest.mark.slow


device = torch_device

# set random seed globally
set_seed(42)


class HFJSDLoss(HFDistillationLoss):
    """
    Naive implementation of a distillation loss using Jensen-Shannon Divergence (JSD).
    """

    def __init__(
        self,
        temperature: float = 1.0,
        ignore_index: int = -100,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
    ):
        super().__init__(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        )

    def distillation_loss(self, student_logits, teacher_logits, target=None, ignore_index=-100, beta=0.5):
        """
        Compute JSD loss (Jensen-Shannon Divergence Loss).

        Args:
            student_logits (torch.Tensor): Logits of student tokens. Shape: (batch_size * seq_len, vocab_size).
            teacher_logits (torch.Tensor): Logits of teacher tokens. Shape: (batch_size * seq_len, vocab_size).
            target (torch.Tensor): Target labels for masking. Shape: (batch_size * seq_len,).
            ignore_index (int): Index to ignore in loss computation.
            beta (float): Coefficient beta of generalized JSD in the interval [0, 1]. Default: `0.5`.
        Returns:
            torch.Tensor: Jensen-Shannon Divergence loss
        """
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            log_mean_probs = torch.logsumexp(
                torch.stack([student_log_probs + math.log(1 - beta), teacher_log_probs + math.log(beta)], dim=0), dim=0
            )
            student_kl = F.kl_div(log_mean_probs, student_log_probs, reduction="none", log_target=True)
            teacher_kl = F.kl_div(log_mean_probs, teacher_log_probs, reduction="none", log_target=True)
            jsd_loss = beta * teacher_kl + (1 - beta) * student_kl

        # Sum over vocab dimension
        jsd_loss = jsd_loss.sum(dim=-1)

        # Apply ignore_index mask
        if target is not None:
            mask = target != ignore_index
            jsd_loss = jsd_loss * mask.float()
            num_valid_tokens = mask.sum().clamp_min(1)
            return jsd_loss.sum() / num_valid_tokens

        return jsd_loss.sum()


class TorchLMHeadJSD(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based jsd loss.
    :param H: hidden size :param V: vocab size :param temperature: softmax temperature :param weight_hard_loss:
    weight_hard_loss :param weight_soft_loss: weight_soft_loss
    """

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool,
        device: torch.device,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        # smaller student model weights
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype, device=device)
        self.beta = beta
        self.jsd = HFJSDLoss(
            ignore_index=ignore_index,
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            temperature=temperature,
        ).get_batch_loss_metrics

    def forward(self, student_input, teacher_input, target):
        jsd_loss = self.jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
            beta=self.beta,
        )
        return jsd_loss

    def backward_with_grad_and_value(self, student_input, teacher_input, target):
        """
        Compute gradients using grad_and_value on NPU to match the fused implementation. This method is used in tests
        on NPU devices to ensure consistency.
        """
        # Use grad_and_value to compute gradients and loss
        if self.student_lin.bias is not None:

            def loss_fn(student_input, student_weight, student_bias):
                return self.jsd(
                    student_input,
                    student_weight,
                    teacher_input,
                    self.teacher_lin.weight,
                    target,
                    student_bias,
                    self.teacher_lin.bias,
                    beta=self.beta,
                )

            (grad_input, grad_weight, grad_bias), loss = torch.func.grad_and_value(loss_fn, argnums=(0, 1, 2))(
                student_input, self.student_lin.weight, self.student_lin.bias
            )

            # Set gradients
            student_input.grad = grad_input
            self.student_lin.weight.grad = grad_weight
            self.student_lin.bias.grad = grad_bias
        else:

            def loss_fn(student_input, student_weight):
                return self.jsd(
                    student_input,
                    student_weight,
                    teacher_input,
                    self.teacher_lin.weight,
                    target,
                    None,  # student_bias is None when bias=False
                    self.teacher_lin.bias,
                    beta=self.beta,
                )

            (grad_input, grad_weight), loss = torch.func.grad_and_value(loss_fn, argnums=(0, 1))(
                student_input, self.student_lin.weight
            )

            # Set gradients
            student_input.grad = grad_input
            self.student_lin.weight.grad = grad_weight

        return loss


class FusedLMHeadJSD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool,
        device: torch.device,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        # smaller student model weights
        self.student_lin = torch.nn.Linear(in_features=H // 2, out_features=V, bias=bias, dtype=dtype, device=device)
        self.teacher_lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype, device=device)
        self.chunked_jsd = FusedLinearJSDLoss(
            weight_hard_loss=weight_hard_loss,
            weight_soft_loss=weight_soft_loss,
            ignore_index=ignore_index,
            temperature=temperature,
            beta=beta,
        )

    def forward(self, student_input, teacher_input, target):
        return self.chunked_jsd(
            student_input,
            self.student_lin.weight,
            teacher_input,
            self.teacher_lin.weight,
            target,
            self.student_lin.bias,
            self.teacher_lin.bias,
        )


#############################################################################
# Test the correctness of the fused linear JSD
#############################################################################


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (8, 128, 1024, 4096),
        (3, 47, 31, 123),  # random shape
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-1),
        (1.0, torch.float32, 1e-5, 5e-4),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "temperature, weight_hard_loss, weight_soft_loss, beta",
    [
        (1.0, 0.5, 0.5, 0.5),
        (2.0, 0.0, 1.0, 0.8),
        (0.5, 1.0, 0.0, 0.2),
    ],
)
@pytest.mark.parametrize("ignore_index", [-100, 42])
def test_correctness(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    temperature,
    weight_hard_loss,
    weight_soft_loss,
    beta,
    ignore_index,
):
    torch_lm_head_jsd = TorchLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        device=device,
        temperature=temperature,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        beta=beta,
        ignore_index=ignore_index,
    )
    fused_lm_head_jsd = FusedLMHeadJSD(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        device=device,
        temperature=temperature,
        weight_hard_loss=weight_hard_loss,
        weight_soft_loss=weight_soft_loss,
        beta=beta,
        ignore_index=ignore_index,
    )

    torch_lm_head_jsd.student_lin.weight.data = fused_lm_head_jsd.student_lin.weight.data = torch.rand(
        V, H // 2, device=device, dtype=dtype
    )
    torch_lm_head_jsd.teacher_lin.weight.data = fused_lm_head_jsd.teacher_lin.weight.data = torch.rand(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_jsd.student_lin.bias.data = fused_lm_head_jsd.student_lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )
        torch_lm_head_jsd.teacher_lin.bias.data = fused_lm_head_jsd.teacher_lin.bias.data = torch.rand(
            V, device=device, dtype=dtype
        )

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)

    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    target = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target[indices_to_assign] = ignore_index

    # Assign some random number of elements as ignore_index
    # On NPU, use grad_and_value for reference implementation to match the fused implementation
    if device == "npu":
        loss1 = torch_lm_head_jsd.backward_with_grad_and_value(student_input1, teacher_input, target)
        loss2 = fused_lm_head_jsd(student_input2, teacher_input, target)
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
        loss2.backward()
    else:
        loss1 = torch_lm_head_jsd(student_input1, teacher_input, target)
        loss2 = fused_lm_head_jsd(student_input2, teacher_input, target)
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)
        loss1.backward()
        loss2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(
        torch_lm_head_jsd.student_lin.weight.grad,
        fused_lm_head_jsd.student_lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )

    if bias:
        assert_verbose_allclose(
            torch_lm_head_jsd.student_lin.bias.grad,
            fused_lm_head_jsd.student_lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
        (9, 7, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "scalar, dtype, atol, rtol",
    [
        (1.0, torch.bfloat16, 5e-2, 5e-2),
        (1.0, torch.float32, 1e-4, 5e-3),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "temperature, weight_hard_loss, weight_soft_loss, beta, ignore_index",
    [(1.0, 0.5, 0.5, 0.5, -100), (2.0, 0.1, 0.9, 0.5, 42)],
)
def test_correctness_functional(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    bias,
    weight_hard_loss,
    weight_soft_loss,
    beta,
    ignore_index,
    temperature,
    atol,
    rtol,
):
    _weight = torch.rand(V, H // 2, device=device, dtype=dtype)
    student_weight1 = _weight.detach().clone().requires_grad_(True)
    student_weight2 = _weight.detach().clone().requires_grad_(True)
    teacher_weight = torch.rand(V, H, device=device, dtype=dtype)

    if bias:
        _bias = torch.rand(V, device=device, dtype=dtype)
        student_bias1 = _bias.detach().clone().requires_grad_(True)
        student_bias2 = _bias.detach().clone().requires_grad_(True)
        teacher_bias = torch.rand(V, device=device, dtype=dtype)
    else:
        student_bias1 = student_bias2 = teacher_bias = None

    _tensor = torch.rand(B * T, H // 2, device=device, dtype=dtype) * scalar
    student_input1 = _tensor.detach().clone().requires_grad_(True)
    student_input2 = _tensor.detach().clone().requires_grad_(True)
    teacher_input = torch.rand(B * T, H, device=device, dtype=dtype) * scalar

    label = torch.randint(0, V, (B * T,), device=device, dtype=torch.long)

    output1 = FusedLinearJSDFunction.apply(
        student_input1,
        student_weight1,
        teacher_input,
        teacher_weight,
        label,
        student_bias1,
        teacher_bias,
        weight_hard_loss,
        weight_soft_loss,
        beta,
        ignore_index,
        temperature,
    )
    output2 = FusedLinearJSDFunction.apply(
        student_input2,
        student_weight2,
        teacher_input,
        teacher_weight,
        label,
        student_bias2,
        teacher_bias,
        weight_hard_loss,
        weight_soft_loss,
        beta,
        ignore_index,
        temperature,
    )

    assert_verbose_allclose(output1, output2, atol=atol, rtol=rtol)

    output1.backward()
    output2.backward()

    assert_verbose_allclose(student_input1.grad, student_input2.grad, atol=atol, rtol=rtol)

    assert_verbose_allclose(student_weight1.grad, student_weight2.grad, atol=atol, rtol=rtol)

    if bias:
        assert_verbose_allclose(student_bias1.grad, student_bias2.grad, atol=atol, rtol=rtol)
