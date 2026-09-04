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
import torch.nn.functional as F
from transformers import set_seed
from transformers.testing_utils import torch_device

from trl.losses import FusedLinearKTOLoss
from trl.losses.kto_loss import FusedLinearKTOFunction

from .testing_utils import HFAlignmentLoss, assert_verbose_allclose


device = torch_device

# set random seed globally
set_seed(0)


class HFKTOLoss(HFAlignmentLoss):
    """
    Implementation of the Kahneman-Tversky Optimization (KTO) loss, adapted from Hugging Face's implementation.
    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            unpaired=True,
            compute_nll_loss=False,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        kl: torch.FloatTensor = None,
    ):
        """Compute KTO loss for a batch of policy log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps:
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            ref_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            ref_rejected_logps:
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            The losses tensor contains the KTO loss for each example in the batch.
        """
        if kl is None:
            kl = torch.zeros(1).to(policy_chosen_logps.device)

        # Chosen losses
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        if policy_chosen_logps.shape[0] != 0 or ref_chosen_logps.shape[0] != 0:
            # Eqn (7) of the KTO paper (https://huggingface.co/papers/2402.01306)
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))

        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(policy_chosen_logps.device)

        # Rejected losses
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        if policy_rejected_logps.shape[0] != 0 or ref_rejected_logps.shape[0] != 0:
            rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = torch.Tensor([]).to(policy_rejected_logps.device)

        losses = torch.cat(
            (chosen_losses, rejected_losses),
            0,
        )

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        return losses, chosen_rewards.sum(), rejected_rewards.sum()


class TorchLMHeadKTO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.KTO_loss = HFKTOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y, preference_labels, kl=None):
        return self.KTO_loss(
            weight=self.lin.weight,
            _input=x,
            target=y,
            bias=self.lin.bias,
            ref_input=ref_x,
            ref_weight=self.ref_lin.weight,
            ref_bias=self.ref_lin.bias,
            preference_labels=preference_labels,
            kl=kl,
            average_log_prob=True,
        )


class FusedLMHeadKTO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.KTO_loss = FusedLinearKTOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            average_log_prob=True,
        )

    def forward(self, x, ref_x, y, preference_labels, kl=None):
        return self.KTO_loss(
            _input=x,
            lin_weight=self.lin.weight,
            target=y,
            preference_labels=preference_labels,
            bias=self.lin.bias,
            ref_input=ref_x,
            ref_weight=self.ref_lin.weight,
            ref_bias=self.ref_lin.bias,
            kl=kl,
        )


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
@pytest.mark.parametrize("ref_bias", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
def test_correctness(B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, ignore_index, beta):
    # Preference labels shape: [B]
    # Create binary preference labels (0 or 1) for each sequence in the batch
    # Used to indicate preferred sequences (1) vs non-preferred sequences (0)
    preference_labels = torch.randint(2, (B,), dtype=torch.bool, device=device, requires_grad=False)
    num_chosen_samples = preference_labels.sum()
    num_rejected_samples = len(preference_labels) - num_chosen_samples

    # Precomputed KL divergence between policy and reference distributions
    kl = torch.randn(1, device=device, dtype=dtype)

    torch_lm_head_KTO = TorchLMHeadKTO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        ignore_index=ignore_index,
        beta=beta,
    )
    fused_lm_head_KTO = FusedLMHeadKTO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_KTO.lin.weight.data = fused_lm_head_KTO.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_KTO.ref_lin.weight.data = fused_lm_head_KTO.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_KTO.lin.bias.data = fused_lm_head_KTO.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head_KTO.ref_lin.bias.data = fused_lm_head_KTO.ref_lin.bias.data = torch.randn(
            V, device=device, dtype=dtype
        )

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

    target = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device=device,
        dtype=torch.long,
    )
    # Assign some random number of elements as ignore_index
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    loss1, aggregated_aux_outputs1 = torch_lm_head_KTO(
        x=input1, ref_x=ref_input, y=target, preference_labels=preference_labels, kl=kl
    )
    loss2, aggregated_aux_outputs2 = fused_lm_head_KTO(
        x=input2, ref_x=ref_input, y=target, preference_labels=preference_labels, kl=kl
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    # Metrics tests are flaky for bf16 due to precision issues
    if dtype == torch.float32:
        # chosen_logps
        chosen_logps_mean1 = aggregated_aux_outputs1[0] / ((num_chosen_samples) + 1e-20)
        chosen_logps_mean2 = aggregated_aux_outputs2[0] / ((num_chosen_samples) + 1e-20)
        assert_verbose_allclose(chosen_logps_mean1, chosen_logps_mean2, atol=atol, rtol=rtol)

        # chosen_logits
        chosen_logits_mean1 = aggregated_aux_outputs1[2] / ((num_chosen_samples * T * V) + 1e-20)
        chosen_logits_mean2 = aggregated_aux_outputs2[2] / ((num_chosen_samples * T * V) + 1e-20)
        assert_verbose_allclose(chosen_logits_mean1, chosen_logits_mean2, atol=atol, rtol=rtol)

        # chosen_rewards
        chosen_rewards_mean1 = aggregated_aux_outputs1[4] / ((num_chosen_samples) + 1e-20)
        chosen_rewards_mean2 = aggregated_aux_outputs2[4] / ((num_chosen_samples) + 1e-20)
        assert_verbose_allclose(chosen_rewards_mean1, chosen_rewards_mean2, atol=atol, rtol=rtol)

        # rejected_logps
        rejected_logps_mean1 = aggregated_aux_outputs1[1] / ((num_rejected_samples) + 1e-20)
        rejected_logps_mean2 = aggregated_aux_outputs2[1] / ((num_rejected_samples) + 1e-20)
        assert_verbose_allclose(rejected_logps_mean1, rejected_logps_mean2, atol=atol, rtol=rtol)

        # rejected_logits
        rejected_logits_mean1 = aggregated_aux_outputs1[3] / ((num_rejected_samples * T * V) + 1e-20)
        rejected_logits_mean2 = aggregated_aux_outputs2[3] / ((num_rejected_samples * T * V) + 1e-20)
        assert_verbose_allclose(rejected_logits_mean1, rejected_logits_mean2, atol=atol, rtol=rtol)

        # rejected_rewards
        rejected_rewards_mean1 = aggregated_aux_outputs1[5] / ((num_rejected_samples) + 1e-20)
        rejected_rewards_mean2 = aggregated_aux_outputs2[5] / ((num_rejected_samples) + 1e-20)
        assert_verbose_allclose(rejected_rewards_mean1, rejected_rewards_mean2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1, input2, atol=atol, rtol=rtol)
    assert_verbose_allclose(torch_lm_head_KTO.lin.weight, fused_lm_head_KTO.lin.weight, atol=atol, rtol=rtol)

    if bias:
        assert_verbose_allclose(torch_lm_head_KTO.lin.bias, fused_lm_head_KTO.lin.bias, atol=atol, rtol=rtol)

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_KTO.lin.weight.grad,
        fused_lm_head_KTO.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_KTO.lin.bias.grad,
            fused_lm_head_KTO.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize(
    "B, T, H, V",
    [
        (2, 2, 8, 8),
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
@pytest.mark.parametrize("ref_bias", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias):
    # Preference labels shape: [B]
    # Create binary preference labels (0 or 1) for each sequence in the batch
    # Used to indicate preferred sequences (1) vs non-preferred sequences (0)
    preference_labels = torch.randint(2, (B,), dtype=torch.bool, device=device)
    num_chosen_samples = preference_labels.sum()
    num_rejected_samples = len(preference_labels) - num_chosen_samples

    # Precomputed KL divergence between policy and reference distributions
    kl = torch.randn(1, device=device, dtype=dtype)

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

    target = torch.randint(
        0,
        V,
        (
            B,
            T,
        ),
        device=device,
        dtype=torch.long,
    )

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    weight1 = _weight.detach().clone().requires_grad_(True)
    weight2 = _weight.detach().clone().requires_grad_(True)

    _ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    ref_weight1 = _ref_weight.detach().clone().requires_grad_(True)
    ref_weight2 = _ref_weight.detach().clone().requires_grad_(True)

    _bias = torch.randn(V, device=device, dtype=dtype) if bias else None
    bias1 = _bias.detach().clone().requires_grad_(True) if bias else None
    bias2 = _bias.detach().clone().requires_grad_(True) if bias else None

    _ref_bias = torch.randn(V, device=device, dtype=dtype) if ref_bias else None
    ref_bias1 = _ref_bias.detach().clone().requires_grad_(True) if ref_bias else None
    ref_bias2 = _ref_bias.detach().clone().requires_grad_(True) if ref_bias else None

    loss1, aggregated_aux_outputs1 = FusedLinearKTOFunction.apply(
        input1,
        weight1,
        target,
        preference_labels,
        bias1,
        ref_input,
        ref_weight1,
        ref_bias1,
        kl,
    )
    loss2, aggregated_aux_outputs2 = FusedLinearKTOFunction.apply(
        input2,
        weight2,
        target,
        preference_labels,
        bias2,
        ref_input,
        ref_weight2,
        ref_bias2,
        kl,
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    # Metrics tests are flaky for bf16 due to precision issues
    if dtype == torch.float32:
        # chosen_logps
        chosen_logps_mean1 = aggregated_aux_outputs1[0] / ((num_chosen_samples) + 1e-20)
        chosen_logps_mean2 = aggregated_aux_outputs2[0] / ((num_chosen_samples) + 1e-20)
        assert_verbose_allclose(chosen_logps_mean1, chosen_logps_mean2, atol=atol, rtol=rtol)

        # chosen_logits
        chosen_logits_mean1 = aggregated_aux_outputs1[2] / ((num_chosen_samples * T * V) + 1e-20)
        chosen_logits_mean2 = aggregated_aux_outputs2[2] / ((num_chosen_samples * T * V) + 1e-20)
        assert_verbose_allclose(chosen_logits_mean1, chosen_logits_mean2, atol=atol, rtol=rtol)

        # chosen_rewards
        chosen_rewards_mean1 = aggregated_aux_outputs1[4] / ((num_chosen_samples) + 1e-20)
        chosen_rewards_mean2 = aggregated_aux_outputs2[4] / ((num_chosen_samples) + 1e-20)
        assert_verbose_allclose(chosen_rewards_mean1, chosen_rewards_mean2, atol=atol, rtol=rtol)

        # rejected_logps
        rejected_logps_mean1 = aggregated_aux_outputs1[1] / ((num_rejected_samples) + 1e-20)
        rejected_logps_mean2 = aggregated_aux_outputs2[1] / ((num_rejected_samples) + 1e-20)
        assert_verbose_allclose(rejected_logps_mean1, rejected_logps_mean2, atol=atol, rtol=rtol)

        # rejected_logits
        rejected_logits_mean1 = aggregated_aux_outputs1[3] / ((num_rejected_samples * T * V) + 1e-20)
        rejected_logits_mean2 = aggregated_aux_outputs2[3] / ((num_rejected_samples * T * V) + 1e-20)
        assert_verbose_allclose(rejected_logits_mean1, rejected_logits_mean2, atol=atol, rtol=rtol)

        # rejected_rewards
        rejected_rewards_mean1 = aggregated_aux_outputs1[5] / ((num_rejected_samples) + 1e-20)
        rejected_rewards_mean2 = aggregated_aux_outputs2[5] / ((num_rejected_samples) + 1e-20)
        assert_verbose_allclose(rejected_rewards_mean1, rejected_rewards_mean2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)
