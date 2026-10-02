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

"""Training-compute backend for [`~experimental.async_grpo.AsyncGRPOTrainer`].

The trainer keeps the RL algorithm (advantages, masks, the loss itself, the metrics) and delegates only the model
compute. It hands its loss to the backend as a [`GRPOLoss`], which is both data and a function of the per-token log
probs, so a backend can use it either way:

* Call it. A backend that owns the model in the trainer's own process runs one forward, calls the loss on the log
  probs, and returns a loss still connected to the graph. A backend that runs the model elsewhere evaluates the loss
  locally on the log probs the remote side sent back, takes `d(loss)/d(log_probs)`, and ships that gradient over the
  wire; the remote side back-propagates `sum(grad_log_probs * log_probs)`, a first-order surrogate whose gradient with
  respect to every parameter equals the gradient of the real loss. Neither needs to know what GRPO is.
* Read it. A backend that computes the loss next to the model reads the advantages, old log probs, mask and clipping
  bounds from its fields and runs the same objective there, avoiding the extra round trip of the surrogate.

A mixture-of-experts router loss is produced by the model rather than from its log probs, so no gradient of it survives
in `d(loss)/d(log_probs)`. It stays a backend concern, added to whatever the backend back-propagates, and reaches the
trainer only as a number to log.
"""

from dataclasses import dataclass

import torch


@dataclass
class GRPOLoss:
    """The clipped GRPO objective of one micro-batch, as data and as a function of the per-token log probs.

    Every tensor is in the target-token frame: shape `(batch_size, sequence_length - 1)`, one entry per predicted token,
    matching the log probs the backend produces.

    Args:
        old_log_probs (`torch.Tensor`):
            Log probabilities of the sampled tokens under the policy that generated them.
        advantages (`torch.Tensor`):
            Per-token advantages.
        completion_mask (`torch.Tensor`):
            1 for tokens the loss applies to, 0 for prompt and tool-result tokens.
        epsilon_low (`float`):
            Lower clipping bound of the importance ratio, as `1 - epsilon_low`.
        epsilon_high (`float`):
            Upper clipping bound of the importance ratio, as `1 + epsilon_high`.
        num_tokens_per_rank (`torch.Tensor`):
            Loss tokens per rank across the whole batch. The sum is divided by it, so that after DDP/FSDP averages
            gradients over ranks, every token in the batch carries the same weight.
        gradient_accumulation_steps (`int`):
            Micro-batches per optimizer step. The loss is divided by it, since HF auto-scaling is off for this trainer.
    """

    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    completion_mask: torch.Tensor
    epsilon_low: float
    epsilon_high: float
    num_tokens_per_rank: torch.Tensor
    gradient_accumulation_steps: int

    def __call__(self, log_probs: torch.Tensor) -> torch.Tensor:
        coef_1 = torch.exp(log_probs - self.old_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * self.advantages
        per_token_loss2 = coef_2 * self.advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        loss = (per_token_loss * self.completion_mask).sum()
        loss = loss / self.num_tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        return loss / self.gradient_accumulation_steps


@dataclass
class ForwardBackwardOutput:
    """What [`~experimental.async_grpo.async_grpo_trainer.TrainingClientProtocol.forward_backward`] returns.

    Args:
        loss (`torch.Tensor`):
            The scalar the trainer passes to `accelerator.backward`. For an in-process backend this is still attached to
            the model's graph. For an off-process backend it is a leaf carrying a hook, so back-propagating it triggers
            the remote backward instead.
        log_probs (`torch.Tensor`):
            Log probability of each target token, shape `(batch_size, sequence_length - 1)`. Detached, and provided for
            metrics.
        entropy (`torch.Tensor`):
            Per-token entropy of the model's next-token distribution, same shape as `log_probs`. Detached, since it is
            reported as a metric and never differentiated.
        aux_loss (`torch.Tensor`, *optional*):
            Mixture-of-experts router load-balancing loss, if the model produces one. Detached, and reported for
            logging only: `aux_loss_coef * aux_loss` is already part of what the backend back-propagates.
    """

    loss: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    aux_loss: torch.Tensor | None = None


class LocalTrainingClient:
    """Runs the model in the trainer's own process.

    The default backend, and the reference implementation of
    [`~experimental.async_grpo.async_grpo_trainer.TrainingClientProtocol`]. One forward pass, and the returned loss is
    still attached to it, so the trainer's backward reaches the model directly. Gradients are bit-identical to computing
    the loss inline.
    """

    def forward_backward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss: GRPOLoss,
        aux_loss_coef: float = 0.0,
    ) -> ForwardBackwardOutput:
        # MoE models: request router logits so the forward returns the load-balancing loss
        router_kwargs = {"output_router_logits": True} if aux_loss_coef else {}
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids.masked_fill(completion_mask == 0, -100),
            fused_lm_head=True,
            **router_kwargs,
        )
        log_probs = outputs.log_probs
        total = loss(log_probs)

        aux_loss = outputs.aux_loss if aux_loss_coef else None
        if aux_loss is not None:
            total = total + aux_loss_coef * aux_loss

        return ForwardBackwardOutput(
            loss=total,
            log_probs=log_probs.detach(),
            entropy=outputs.entropy.detach(),
            aux_loss=aux_loss.detach() if aux_loss is not None else None,
        )
