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

"""Training backend for [`~experimental.async_grpo.AsyncGRPOTrainer`] that owns the model.

The trainer keeps the RL algorithm (advantages, masks, the loss, the learning-rate schedule, the metrics) and hands its
loss to the backend as a [`GRPOLoss`], which is both data and a function of the per-token log probs, so a backend can
use it either way:

* Call it. A backend that runs the model elsewhere evaluates the loss locally on the log probs the remote side sent
  back, takes `d(loss)/d(log_probs)`, and ships that gradient over the wire; the remote side back-propagates
  `sum(grad_log_probs * log_probs)`, a first-order surrogate whose gradient with respect to every parameter equals the
  gradient of the real loss. The backend never needs to know what GRPO is.
* Read it. A backend that computes the loss next to the model reads the advantages, old log probs, mask and clipping
  bounds from its fields and runs the same objective there, avoiding the extra round trip of the surrogate.

A mixture-of-experts router loss is produced by the model rather than from its log probs, so no gradient of it survives
in `d(loss)/d(log_probs)`. It stays a backend concern, added to whatever the backend back-propagates, and reaches the
trainer only as a number to log.
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn
from transformers import GenerationConfig, PretrainedConfig


@dataclass
class GRPOLoss:
    """The clipped GRPO objective of one micro-batch, as data and as a function of the per-token log probs.

    Every tensor is in the target-token frame: shape `(batch_size, sequence_length - 1)`, one entry per predicted
    token, matching the log probs the backend produces.

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
        aux_loss_coef (`float`, *optional*, defaults to `0.0`):
            Coefficient of the mixture-of-experts router load-balancing loss, which the model produces rather than the
            log probs, so calling this loss does not include it. The backend adds `aux_loss_coef * aux_loss /
            gradient_accumulation_steps` to what it back-propagates. `0.0` disables it.
    """

    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    completion_mask: torch.Tensor
    epsilon_low: float
    epsilon_high: float
    num_tokens_per_rank: torch.Tensor
    gradient_accumulation_steps: int
    aux_loss_coef: float = 0.0

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
            The scalar the trainer passes to `accelerator.backward`: a detached leaf carrying a hook, so
            back-propagating it triggers the backend's backward.
        log_probs (`torch.Tensor`):
            Log probability of each target token, shape `(batch_size, sequence_length - 1)`. Detached, and provided for
            metrics.
        entropy (`torch.Tensor`):
            Per-token entropy of the model's next-token distribution, same shape as `log_probs`. Detached, since it is
            reported as a metric and never differentiated.
        aux_loss (`torch.Tensor`, *optional*):
            Mixture-of-experts router load-balancing loss, if the model produces one. Detached, and reported for
            logging only: the backend already back-propagates it, weighted as [`GRPOLoss`] describes.
    """

    loss: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    aux_loss: torch.Tensor | None = None


class _RemoteModel(nn.Module):
    """Stands in for a model whose weights live with the training client.

    Carries the model's config, which the trainer still reads (MoE and VLM detection, the pad token, the served model
    name), and a single placeholder parameter for `Trainer` and `accelerate` to place and wrap. It has no forward: the
    training client runs the model.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)
        self.placeholder = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        raise RuntimeError("The model's weights live with the training client, which runs its forward.")


class _RemoteOptimizer(torch.optim.Optimizer):
    """Steps the optimizer of a training client that owns the parameters.

    The trainer's learning-rate scheduler writes the scheduled rate into `param_groups`, and each `step` forwards it to
    the client, so the schedule drives the remote optimizer. The metrics the client returns (e.g. `grad_norm`) go to
    the trainer's metric sink, since the local placeholder parameter carries no gradient to measure.
    """

    def __init__(self, params, training_client, lr: float, metrics: defaultdict):
        super().__init__(params, {"lr": lr})
        self.training_client = training_client
        self.metrics = metrics

    def step(self, closure=None):
        for key, value in self.training_client.optimizer_step(learning_rate=self.param_groups[0]["lr"]).items():
            self.metrics[key].append(value)
