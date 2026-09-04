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

# Copyright 2024 LinkedIn Corporation
# All Rights Reserved.
#
# This source code is licensed under the BSD 2-Clause license found in the
# LICENSE file in the root directory of https://github.com/linkedin/Liger-Kernel.
#
# Adapted from Liger-Kernel v0.8.2 (000be609), src/liger_kernel/chunked_loss/dpo_loss.py.

import torch
import torch.nn.functional as F

from .fused_linear_preference import FusedLinearPreferenceBase


class FusedLinearDPOFunction(FusedLinearPreferenceBase):
    @staticmethod
    def preference_loss_fn(
        chosen_logps,
        rejected_logps,
        full_target,
        ref_chosen_logps=None,
        ref_rejected_logps=None,
        beta=0.1,
        loss_type="sigmoid",
        label_smoothing=0.0,
        discopop_tau=0.05,
    ):
        """
        Paper: https://arxiv.org/pdf/2305.18290

        Formula:
        L_DPO = -E[ log_sigmoid( β * (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x))) ) ]

        Where:
        - π(y|x): Policy (model) probability
        - π_ref(y|x): Reference model probability
        - y_w: Chosen sequence
        - y_l: Rejected sequence
        - β: Weight for the direct preference loss
        - E: Expected value over the dataset

        Args:
            chosen_logps: Log probabilities of chosen tokens (batch_size,)
            rejected_logps: Log probabilities of rejected tokens (batch_size,)
            full_target: Non chunked full target tensor
            ref_chosen_logps: Reference log probs of chosen tokens (batch_size,)
            ref_rejected_logps: Reference log probs of rejected tokens (batch_size,)
            beta: Weight for the direct preference loss
            loss_type: Variant of DPO loss to compute
            label_smoothing: Label smoothing for "robust" / "exo_pair" / cDPO
            discopop_tau: Temperature for the DiscoPOP modulation term
        """

        if ref_chosen_logps is None:
            ref_chosen_logps = torch.tensor(0.0, device=chosen_logps.device)
        if ref_rejected_logps is None:
            ref_rejected_logps = torch.tensor(0.0, device=rejected_logps.device)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios

        logits_diff = beta * (chosen_logratios - rejected_logratios)
        n_pairs = full_target.shape[0] // 2

        if loss_type == "sigmoid":
            loss = -F.logsigmoid(logits_diff).sum() / n_pairs

        elif loss_type == "hinge":
            # Hinge loss on the normalized likelihood: max(0, 1 - β * (chosen_logratios - rejected_logratios))
            losses = torch.relu(1 - logits_diff)
            loss = losses.sum() / n_pairs

        elif loss_type == "ipo":
            raise NotImplementedError(
                "loss_type='ipo' is not yet supported by the fused linear DPO loss because it requires per-sample completion-token "
                "counts to length-normalize the squared margin."
            )

        elif loss_type == "exo_pair":
            # Implements EXO-pref from the paper https://huggingface.co/papers/2402.00856 (Eq. 16)
            # Minimize KL(p_fθ || p_rh) for K=2; p_rh = [(1−ε), ε].
            if label_smoothing <= 0.0:
                raise ValueError(
                    "label_smoothing must be > 0 for loss_type='exo_pair'. The EXO paper recommends 1e-3."
                )
            epsilon = torch.tensor(label_smoothing, device=chosen_logps.device)
            qw = torch.sigmoid(logits_diff)
            log_qw = F.logsigmoid(logits_diff)
            log_pw = torch.log1p(-epsilon)
            ql = torch.sigmoid(-logits_diff)
            log_ql = F.logsigmoid(-logits_diff)
            log_pl = torch.log(epsilon)
            losses = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)
            loss = losses.sum() / n_pairs

        elif loss_type == "nca_pair":
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
            loss = losses.sum() / n_pairs

        elif loss_type == "robust":
            # cDPO / robust loss: assumes a fraction `label_smoothing` of preference labels are flipped.
            if not (0.0 <= label_smoothing < 0.5):
                raise ValueError(
                    f"label_smoothing must lie in [0.0, 0.5) for loss_type='robust'. Got {label_smoothing}."
                )
            clean_loss_term = -(1 - label_smoothing) * F.logsigmoid(logits_diff)
            flipped_loss_term = -label_smoothing * F.logsigmoid(-logits_diff)
            losses = (clean_loss_term - flipped_loss_term) / (1 - 2 * label_smoothing)
            loss = losses.sum() / n_pairs

        elif loss_type == "bco_pair":
            losses = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)
            loss = losses.sum() / n_pairs

        elif loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2
            loss = losses.sum() / n_pairs

        elif loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(beta * rejected_logratios)
            losses = losses_chosen + losses_rejected
            loss = losses.sum() / n_pairs

        elif loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected
            loss = losses.sum() / n_pairs

        elif loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            log_ratio_modulation = torch.sigmoid(logits_diff / discopop_tau)
            logistic_component = -F.logsigmoid(logits_diff)
            exp_component = torch.exp(-logits_diff)
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
            loss = losses.sum() / n_pairs

        else:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. Supported types are: sigmoid, hinge, exo_pair, nca_pair, "
                "robust, bco_pair, sppo_hard, apo_zero, apo_down, discopop"
            )

        return loss, chosen_rewards, rejected_rewards

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        ignore_index=-100,
        beta=0.1,
        compute_nll_loss=False,
        compiled=True,
        use_ref_model=True,
        average_log_prob=False,
        chunk_size=1,
        loss_type="sigmoid",
        label_smoothing=0.0,
        discopop_tau=0.05,
        alpha=1.0,
    ):
        """
        Fused linear layer with DPO loss.
        Args:
            _input (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, hidden_size)
            weight (torch.Tensor): Weight tensor. Shape: (vocab_size, hidden_size)
            target (torch.LongTensor): Target tensor. Shape: (batch_size * seq_len,)
            bias (torch.Tensor, optional): Bias tensor. Shape: (vocab_size,)
            ref_input (torch.Tensor, optional): Reference model input tensor. Shape: (batch_size * seq_len, hidden_size)
            ref_weight (torch.Tensor, optional): Reference model weight tensor. Shape: (vocab_size, hidden_size)
            ref_bias (torch.Tensor, optional): Reference model bias tensor. Shape: (vocab_size,)
            ignore_index (int): Index to ignore in loss computation
            beta (float): Weight for the direct preference loss
            compute_nll_loss (bool): Whether to compute the NLL loss
            compiled (bool): Whether to use torch compile
            use_ref_model (bool): Whether to use a reference model
            average_log_prob (bool): Whether to average the log probability per non-masked token
            chunk_size (int): Size of chunks for processing.
            loss_type (str): Variant of DPO loss to compute.
            label_smoothing (float): Label smoothing for "robust" / "exo_pair" / cDPO.
            discopop_tau (float): Temperature for the DiscoPOP modulation term.
            alpha (float): Weight for the NLL loss component
        Returns:
            torch.Tensor: Computed loss
        """
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ignore_index=ignore_index,
            beta=beta,
            alpha=alpha,
            compute_nll_loss=compute_nll_loss,
            compiled=compiled,
            use_ref_model=use_ref_model,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            average_log_prob=average_log_prob,
            chunk_size=chunk_size,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            discopop_tau=discopop_tau,
        )

    @staticmethod
    def backward(ctx, *grad_output):
        grads = FusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
        return *grads, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class FusedLinearDPOLoss(torch.nn.Module):
    """
    Fused linear layer with DPO loss.
    """

    _SUPPORTED_LOSS_TYPES = {
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
    }

    def __init__(
        self,
        ignore_index: int = -100,
        beta: float = 0.1,
        alpha: float = 1.0,
        compute_nll_loss: bool = False,
        compiled: bool = True,
        use_ref_model: bool = True,
        average_log_prob: bool = False,
        chunk_size: int = 1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        discopop_tau: float = 0.05,
    ):
        """
        Args:
            ignore_index (int): Index to ignore in the loss.
            beta (float): Weight for the direct preference loss.
            alpha (float): Weight for the NLL loss component.
            compute_nll_loss (bool): Whether to compute the NLL loss.
            compiled (bool): Whether to use the torch compiled kernel.
            use_ref_model (bool): Whether to use a reference model for the DPO loss.
            average_log_prob (bool): Whether to average the log probability per non-masked token.
            chunk_size (int): Size of chunks for processing.
            loss_type (str): Variant of DPO loss to compute. One of:
                "sigmoid", "hinge", "exo_pair", "nca_pair", "robust", "bco_pair",
                "sppo_hard", "apo_zero", "apo_down", "discopop".
            label_smoothing (float): Label smoothing for "robust" / "exo_pair" / cDPO.
            discopop_tau (float): Temperature for the DiscoPOP modulation term.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta
        self.alpha = alpha
        self.compute_nll_loss = compute_nll_loss
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.average_log_prob = average_log_prob
        self.chunk_size = chunk_size
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.discopop_tau = discopop_tau
        if self.loss_type not in self._SUPPORTED_LOSS_TYPES:
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type}. Supported types are: {self._SUPPORTED_LOSS_TYPES}"
            )
        if self.loss_type == "exo_pair" and self.label_smoothing <= 0.0:
            raise ValueError("label_smoothing must be > 0 for loss_type='exo_pair'. The EXO paper recommends 1e-3.")
        if self.loss_type == "robust" and not (0.0 <= self.label_smoothing < 0.5):
            raise ValueError(
                f"label_smoothing must lie in [0.0, 0.5) for loss_type='robust'. Got {self.label_smoothing}."
            )

    def forward(
        self,
        lin_weight,
        _input,
        target,
        bias=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return FusedLinearDPOFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            ref_input,
            ref_weight,
            ref_bias,
            self.ignore_index,
            self.beta,
            self.compute_nll_loss,
            self.compiled,
            self.use_ref_model,
            self.average_log_prob,
            self.chunk_size,
            self.loss_type,
            self.label_smoothing,
            self.discopop_tau,
            self.alpha,
        )
