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

from trl.losses import FusedLinearDPOLoss
from trl.losses.dpo_loss import FusedLinearDPOFunction

from .testing_utils import HFAlignmentLoss, assert_verbose_allclose


device = torch_device

# set random seed globally
set_seed(42)


class HFDPOLoss(HFAlignmentLoss):
    """
    Implementation of the Direct Preference Optimization (DPO) loss,
    adapted from Hugging Face's implementation.
    Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute DPO loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the DPO loss for each example in the batch.
        """
        # Derived from https://huggingface.co/papers/2305.18290
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        losses = -F.logsigmoid(logits_diff)
        return losses, chosen_rewards, rejected_rewards


class HFAPOZeroLoss(HFAlignmentLoss):
    """
    Implementation of the APO-zero loss.
    Reference: https://huggingface.co/papers/2408.06266
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute APO-zero loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the APO-zero loss for each example in the batch.
        """
        # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # Use this loss when you believe the chosen outputs are better than your model's default output
        losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
        losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
        losses = losses_chosen + losses_rejected

        return losses, chosen_rewards, rejected_rewards


class HFAPODownLoss(HFAlignmentLoss):
    """
    Implementation of the APO-down loss.
    Reference: https://huggingface.co/papers/2408.06266
    """

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        """Compute APO-down loss for a batch of policy log probabilities.
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            The losses tensor contains the APO-down loss for each example in the batch.
        """
        # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        # Use this loss when you believe the chosen outputs are worse than your model's default output.
        # Decrease chosen likelihood and decrease rejected likelihood more
        losses_chosen = F.sigmoid(self.beta * chosen_logratios)
        losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
        losses = losses_chosen + losses_rejected

        return losses, chosen_rewards, rejected_rewards


class HFSPPPOHARDLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        a = policy_chosen_logps - ref_chosen_logps
        b = policy_rejected_logps - ref_rejected_logps
        losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        return losses, chosen_rewards, rejected_rewards


class HFNCAPAIRLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        losses = (
            -F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards) - 0.5 * F.logsigmoid(-rejected_rewards)
        )

        return losses, chosen_rewards, rejected_rewards


class HFHingeLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        losses = torch.relu(1 - logits_diff)
        return losses, chosen_rewards, rejected_rewards


class HFBCOPAIRLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        losses = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)
        return losses, chosen_rewards, rejected_rewards


class HFRobustLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
        label_smoothing: float = 0.1,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )
        self.label_smoothing = label_smoothing

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(logits_diff)
        flipped_loss_term = -self.label_smoothing * F.logsigmoid(-logits_diff)
        losses = (clean_loss_term - flipped_loss_term) / (1 - 2 * self.label_smoothing)
        return losses, chosen_rewards, rejected_rewards


class HFEXOPAIRLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
        label_smoothing: float = 1e-3,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )
        self.label_smoothing = label_smoothing

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        epsilon = torch.tensor(self.label_smoothing, device=policy_chosen_logps.device)
        qw = torch.sigmoid(logits_diff)
        log_qw = F.logsigmoid(logits_diff)
        log_pw = torch.log1p(-epsilon)
        ql = torch.sigmoid(-logits_diff)
        log_ql = F.logsigmoid(-logits_diff)
        log_pl = torch.log(epsilon)
        losses = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)
        return losses, chosen_rewards, rejected_rewards


class HFDiscoPOPLoss(HFAlignmentLoss):
    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        use_ref_model: bool = True,
        compute_nll_loss: bool = False,
        discopop_tau: float = 0.05,
    ):
        super().__init__(
            beta=beta,
            ignore_index=ignore_index,
            use_ref_model=use_ref_model,
            compute_nll_loss=compute_nll_loss,
        )
        self.discopop_tau = discopop_tau

    def alignment_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ):
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios

        logits_diff = self.beta * (chosen_logratios - rejected_logratios)
        log_ratio_modulation = torch.sigmoid(logits_diff / self.discopop_tau)
        logistic_component = -F.logsigmoid(logits_diff)
        exp_component = torch.exp(-logits_diff)
        losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
        return losses, chosen_rewards, rejected_rewards


class TorchLMHeadDPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.dpo_loss = HFDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.dpo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadAPOZero(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.apo_loss = HFAPOZeroLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.apo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadAPODown(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.apo_loss = HFAPODownLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.apo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadSPPOHARD(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.sppo_hard = HFSPPPOHARDLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.sppo_hard(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadNCAPAIR(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.nca_pair = HFNCAPAIRLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.nca_pair(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class TorchLMHeadGenericDPO(torch.nn.Module):
    """Wrapper around an arbitrary HFAlignmentLoss subclass. Used for the new loss types."""

    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        hf_loss_cls,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        **hf_loss_kwargs,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.loss = hf_loss_cls(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
            **hf_loss_kwargs,
        ).get_batch_loss_metrics

    def forward(self, x, ref_x, y):
        return self.loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
            average_log_prob=True,
        )


class FusedLMHeadDPO(torch.nn.Module):
    def __init__(
        self,
        H: int,
        V: int,
        dtype: torch.dtype,
        bias: bool = False,
        ref_bias: bool = False,
        compute_nll_loss: bool = False,
        ignore_index: int = -100,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        discopop_tau: float = 0.05,
    ):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=bias, dtype=dtype)
        self.ref_lin = torch.nn.Linear(in_features=H, out_features=V, bias=ref_bias, dtype=dtype)
        self.dpo_loss = FusedLinearDPOLoss(
            ignore_index=ignore_index,
            beta=beta,
            use_ref_model=True,
            compute_nll_loss=compute_nll_loss,
            average_log_prob=True,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            discopop_tau=discopop_tau,
        )

    def forward(self, x, ref_x, y):
        return self.dpo_loss(
            self.lin.weight,
            x,
            y,
            self.lin.bias,
            ref_x,
            self.ref_lin.weight,
            self.ref_lin.bias,
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
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
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
    ref_bias,
    compute_nll_loss,
    ignore_index,
    beta,
):
    B = 2 * B  # dpo loss requires B to be even

    torch_lm_head_dpo = TorchLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
    )
    fused_lm_head_dpo = FusedLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
    )

    torch_lm_head_dpo.lin.weight.data = fused_lm_head_dpo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_dpo.ref_lin.weight.data = fused_lm_head_dpo.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_dpo.lin.bias.data = fused_lm_head_dpo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head_dpo.ref_lin.bias.data = fused_lm_head_dpo.ref_lin.bias.data = torch.randn(
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

    loss1, aggregated_aux_outputs1 = torch_lm_head_dpo(input1, ref_input, target)
    loss2, aggregated_aux_outputs2 = fused_lm_head_dpo(input2, ref_input, target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        if i > 4 and dtype == torch.bfloat16:
            # numerical instability in bf16 for chosen_rewards and rejected_rewards
            # temporary fix. TODO: investigate how to reduce numercial instabiltiy issue
            assert_verbose_allclose(
                aggregated_aux_outputs1[i],
                aggregated_aux_outputs2[i],
                atol=5e-1,
                rtol=rtol,
            )
            continue
        assert_verbose_allclose(
            aggregated_aux_outputs1[i],
            aggregated_aux_outputs2[i],
            atol=atol,
            rtol=rtol,
        )

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_dpo.lin.weight.grad,
        fused_lm_head_dpo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_dpo.lin.bias.grad,
            fused_lm_head_dpo.lin.bias.grad,
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
@pytest.mark.parametrize("compute_nll_loss", [True, False])
def test_correctness_functional(B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, compute_nll_loss):
    B = 2 * B

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

    loss1, aggregated_aux_outputs1 = FusedLinearDPOFunction.apply(
        input1,
        weight1,
        target,
        bias1,
        ref_input,
        ref_weight1,
        ref_bias1,
        -100,  # ignore_index
        0.1,  # beta
        compute_nll_loss,
    )
    loss2, aggregated_aux_outputs2 = FusedLinearDPOFunction.apply(
        input2,
        weight2,
        target,
        bias2,
        ref_input,
        ref_weight2,
        ref_bias2,
        -100,  # ignore_index
        0.1,  # beta
        compute_nll_loss,
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


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
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
@pytest.mark.parametrize("loss_type", ["apo_zero", "apo_down", "sppo_hard", "nca_pair"])
def test_correctness_apo_loss_types(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    ref_bias,
    compute_nll_loss,
    ignore_index,
    beta,
    loss_type,
):
    B = 2 * B  # dpo loss requires B to be even

    # Select the appropriate HF reference implementation
    if loss_type == "apo_zero":
        torch_lm_head = TorchLMHeadAPOZero
    elif loss_type == "apo_down":
        torch_lm_head = TorchLMHeadAPODown
    elif loss_type == "sppo_hard":
        torch_lm_head = TorchLMHeadSPPOHARD
    elif loss_type == "nca_pair":
        torch_lm_head = TorchLMHeadNCAPAIR
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    torch_lm_head_apo = torch_lm_head(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
    )
    fused_lm_head_apo = FusedLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
        loss_type=loss_type,
    )

    torch_lm_head_apo.lin.weight.data = fused_lm_head_apo.lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )
    torch_lm_head_apo.ref_lin.weight.data = fused_lm_head_apo.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head_apo.lin.bias.data = fused_lm_head_apo.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head_apo.ref_lin.bias.data = fused_lm_head_apo.ref_lin.bias.data = torch.randn(
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

    loss1, aggregated_aux_outputs1 = torch_lm_head_apo(input1, ref_input, target)
    loss2, aggregated_aux_outputs2 = fused_lm_head_apo(input2, ref_input, target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        if i > 4 and dtype == torch.bfloat16:
            # numerical instability in bf16 for chosen_rewards and rejected_rewards
            # temporary fix. TODO: investigate how to reduce numerical instability issue
            assert_verbose_allclose(
                aggregated_aux_outputs1[i],
                aggregated_aux_outputs2[i],
                atol=5e-1,
                rtol=rtol,
            )
            continue
        assert_verbose_allclose(
            aggregated_aux_outputs1[i],
            aggregated_aux_outputs2[i],
            atol=atol,
            rtol=rtol,
        )

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head_apo.lin.weight.grad,
        fused_lm_head_apo.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head_apo.lin.bias.grad,
            fused_lm_head_apo.lin.bias.grad,
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
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("loss_type", ["apo_zero", "apo_down", "sppo_hard", "nca_pair"])
def test_correctness_functional_apo_loss_types(
    B, T, H, V, scalar, dtype, atol, rtol, bias, ref_bias, compute_nll_loss, loss_type
):
    B = 2 * B

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

    # Call with loss_type parameter for FusedLinearDPOFunction
    loss1, aggregated_aux_outputs1 = FusedLinearDPOFunction.apply(
        input1,
        weight1,
        target,
        bias1,
        ref_input,
        ref_weight1,
        ref_bias1,
        -100,  # ignore_index
        0.1,  # beta
        compute_nll_loss,
        True,  # compiled
        True,  # use_ref_model
        False,  # average_log_prob
        1,  # chunk_size
        loss_type,
    )

    # For comparison, create a FusedLinearDPOLoss with the loss_type
    dpo_loss_fn = FusedLinearDPOLoss(
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=compute_nll_loss,
        loss_type=loss_type,
    )

    loss2, aggregated_aux_outputs2 = dpo_loss_fn(
        weight2,
        input2,
        target,
        bias2,
        ref_input,
        ref_weight2,
        ref_bias2,
    )

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


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
@pytest.mark.parametrize("compute_nll_loss", [True, False])
@pytest.mark.parametrize("ignore_index, beta", [(-100, 0.1), (42, 0.2)])
@pytest.mark.parametrize(
    "loss_type, hf_loss_cls, hf_kwargs, fused_kwargs",
    [
        ("hinge", HFHingeLoss, {}, {}),
        ("bco_pair", HFBCOPAIRLoss, {}, {}),
        ("robust", HFRobustLoss, {"label_smoothing": 0.1}, {"label_smoothing": 0.1}),
        ("exo_pair", HFEXOPAIRLoss, {"label_smoothing": 1e-3}, {"label_smoothing": 1e-3}),
        ("discopop", HFDiscoPOPLoss, {"discopop_tau": 0.05}, {"discopop_tau": 0.05}),
    ],
)
def test_correctness_extra_loss_types(
    B,
    T,
    H,
    V,
    scalar,
    dtype,
    atol,
    rtol,
    bias,
    ref_bias,
    compute_nll_loss,
    ignore_index,
    beta,
    loss_type,
    hf_loss_cls,
    hf_kwargs,
    fused_kwargs,
):
    B = 2 * B  # dpo loss requires B to be even

    torch_lm_head = TorchLMHeadGenericDPO(
        H=H,
        V=V,
        dtype=dtype,
        hf_loss_cls=hf_loss_cls,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
        **hf_kwargs,
    )
    fused_lm_head = FusedLMHeadDPO(
        H=H,
        V=V,
        dtype=dtype,
        bias=bias,
        ref_bias=ref_bias,
        compute_nll_loss=compute_nll_loss,
        ignore_index=ignore_index,
        beta=beta,
        loss_type=loss_type,
        **fused_kwargs,
    )

    torch_lm_head.lin.weight.data = fused_lm_head.lin.weight.data = torch.randn(V, H, device=device, dtype=dtype)
    torch_lm_head.ref_lin.weight.data = fused_lm_head.ref_lin.weight.data = torch.randn(
        V, H, device=device, dtype=dtype
    )

    if bias:
        torch_lm_head.lin.bias.data = fused_lm_head.lin.bias.data = torch.randn(V, device=device, dtype=dtype)
    if ref_bias:
        torch_lm_head.ref_lin.bias.data = fused_lm_head.ref_lin.bias.data = torch.randn(V, device=device, dtype=dtype)

    _input = torch.randn(B, T, H, device=device, dtype=dtype) * scalar
    input1 = _input.detach().clone().requires_grad_(True)
    input2 = _input.detach().clone().requires_grad_(True)

    ref_input = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=False) * scalar

    target = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    num_elements_to_assign = torch.randint(1, B * T // 2, (1,)).item()
    indices_to_assign = torch.randperm(B * T)[:num_elements_to_assign]
    target.view(-1)[indices_to_assign] = ignore_index

    loss1, aggregated_aux_outputs1 = torch_lm_head(input1, ref_input, target)
    loss2, aggregated_aux_outputs2 = fused_lm_head(input2, ref_input, target)

    assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    assert len(aggregated_aux_outputs1) == len(aggregated_aux_outputs2)

    for i in range(len(aggregated_aux_outputs1)):
        if i > 4 and dtype == torch.bfloat16:
            assert_verbose_allclose(
                aggregated_aux_outputs1[i],
                aggregated_aux_outputs2[i],
                atol=5e-1,
                rtol=rtol,
            )
            continue
        assert_verbose_allclose(
            aggregated_aux_outputs1[i],
            aggregated_aux_outputs2[i],
            atol=atol,
            rtol=rtol,
        )

    loss1.backward()
    loss2.backward()

    assert_verbose_allclose(input1.grad, input2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(
        torch_lm_head.lin.weight.grad,
        fused_lm_head.lin.weight.grad,
        atol=atol,
        rtol=rtol,
    )
    if bias:
        assert_verbose_allclose(
            torch_lm_head.lin.bias.grad,
            fused_lm_head.lin.bias.grad,
            atol=atol,
            rtol=rtol,
        )


def test_invalid_loss_type():
    """Test that invalid loss types raise ValueError"""
    with pytest.raises(ValueError, match="Unsupported loss_type"):
        FusedLinearDPOLoss(loss_type="invalid_loss_type")

    # Test that valid loss types don't raise errors
    valid_loss_types = [
        "sigmoid",
        "hinge",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "apo_zero",
        "apo_down",
        "discopop",
    ]
    extra_kwargs = {
        "exo_pair": {"label_smoothing": 1e-3},
        "robust": {"label_smoothing": 0.1},
    }
    for loss_type in valid_loss_types:
        loss_fn = FusedLinearDPOLoss(loss_type=loss_type, **extra_kwargs.get(loss_type, {}))
        assert loss_fn.loss_type == loss_type


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_alpha_scales_nll_loss(dtype):
    """
    Verify that alpha is actually forwarded and scales the NLL component.
    With compute_nll_loss=True, loss(alpha=2) should differ from loss(alpha=1).
    """
    B, T, H, V = 4, 16, 32, 64
    atol = 1e-4 if dtype == torch.float32 else 5e-2

    _weight = torch.randn(V, H, device=device, dtype=dtype)
    _ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

    def run(alpha):
        inp = _input.detach().clone().requires_grad_(True)
        w = _weight.detach().clone().requires_grad_(True)
        rw = _ref_weight.detach().clone().requires_grad_(True)
        loss_fn = FusedLinearDPOLoss(
            beta=0.1,
            alpha=alpha,
            compute_nll_loss=True,
            use_ref_model=True,
            average_log_prob=False,
        )
        loss, _ = loss_fn(w, inp, target, None, _input.detach(), rw, None)
        return loss

    loss_alpha1 = run(alpha=1.0)
    loss_alpha2 = run(alpha=2.0)

    assert not torch.allclose(loss_alpha1, loss_alpha2, atol=atol), (
        f"Expected losses to differ when alpha changes, but got {loss_alpha1} vs {loss_alpha2}"
    )


def test_functional_positional_arg_contract():
    """
    Pin the positional-argument contract of the public functional alias.

    `alpha` is appended at the *end* of `FusedLinearDPOFunction.forward` (not
    inserted mid-list) precisely so that existing positional `.apply()` /
    `FusedLinearDPOFunction.apply` callers don't silently shift every later argument by
    one slot. This test exercises the pre-PR positional list (no `alpha`) and
    asserts it produces the same result as the keyword-driven `nn.Module` wrapper.
    If a future param insertion shifts the positional slots, this diverges.
    """
    B, T, H, V = 4, 8, 16, 32
    dtype = torch.float32

    _input = torch.randn(B, T, H, device=device, dtype=dtype)
    target = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    _weight = torch.randn(V, H, device=device, dtype=dtype)
    _ref_weight = torch.randn(V, H, device=device, dtype=dtype)
    ref_input = torch.randn(B, T, H, device=device, dtype=dtype)

    # Pre-PR positional list: the args after `beta` are
    # compute_nll_loss, compiled, use_ref_model, average_log_prob, chunk_size, loss_type.
    loss_positional, _ = FusedLinearDPOFunction.apply(
        _input.detach().clone().requires_grad_(True),
        _weight.detach().clone().requires_grad_(True),
        target,
        None,  # bias
        ref_input,
        _ref_weight.detach().clone().requires_grad_(True),
        None,  # ref_bias
        -100,  # ignore_index
        0.1,  # beta
        True,  # compute_nll_loss
        True,  # compiled
        True,  # use_ref_model
        False,  # average_log_prob
        1,  # chunk_size
        "sigmoid",  # loss_type
    )

    loss_module, _ = FusedLinearDPOLoss(
        ignore_index=-100,
        beta=0.1,
        alpha=1.0,
        compute_nll_loss=True,
        compiled=True,
        use_ref_model=True,
        average_log_prob=False,
        chunk_size=1,
        loss_type="sigmoid",
    )(
        _weight.detach().clone().requires_grad_(True),
        _input.detach().clone().requires_grad_(True),
        target,
        None,  # bias
        ref_input,
        _ref_weight.detach().clone().requires_grad_(True),
        None,  # ref_bias
    )

    assert_verbose_allclose(loss_positional, loss_module, atol=1e-5, rtol=1e-4)


def test_label_smoothing_validation():
    """Test that invalid label_smoothing values raise ValueError for the relevant loss types."""
    with pytest.raises(ValueError, match="label_smoothing must be > 0 for loss_type='exo_pair'"):
        FusedLinearDPOLoss(loss_type="exo_pair", label_smoothing=0.0)

    with pytest.raises(ValueError, match=r"label_smoothing must lie in \[0\.0, 0\.5\) for loss_type='robust'"):
        FusedLinearDPOLoss(loss_type="robust", label_smoothing=0.5)

    with pytest.raises(ValueError, match=r"label_smoothing must lie in \[0\.0, 0\.5\) for loss_type='robust'"):
        FusedLinearDPOLoss(loss_type="robust", label_smoothing=-0.1)
