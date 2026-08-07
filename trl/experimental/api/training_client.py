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

"""Training-compute backend interface for TRL trainers.

The trainer keeps the RL algorithm (advantages, masks, the loss itself, the metrics) and delegates only the model
compute. A backend runs the forward pass and returns per-token log probs; the trainer builds its loss on those log
probs; the backend then applies `d(loss)/d(log_probs)` to the model. Because the loss never leaves the trainer, a
backend does not need to know what GRPO is, and TRL's loss variants keep evolving without touching any backend.

That split is what makes the interface implementable off-process: `log_probs` is a plain tensor, and its gradient is a
plain tensor of the same shape.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch


if TYPE_CHECKING:
    from accelerate import Accelerator


@dataclass
class ForwardOutput:
    """Per-token quantities returned by [`TrainingClientProtocol.forward`].

    Args:
        log_probs (`torch.Tensor`):
            Log probability of each target token, shape `(batch_size, sequence_length - 1)`. A leaf tensor with
            `requires_grad=True`: the trainer builds its loss on it, and the gradient that accumulates here is what
            gets handed back to [`TrainingClientProtocol.backward`].
        entropy (`torch.Tensor`):
            Per-token entropy of the model's next-token distribution, same shape as `log_probs`. Detached, since it is
            reported as a metric and never differentiated.
        aux_loss (`torch.Tensor`, *optional*):
            Mixture-of-experts router load-balancing loss, if the model produces one. Detached: it is not a function of
            `log_probs`, so it cannot reach the model through the backward pass above. The backend applies it directly,
            scaled by the `aux_loss_coef` passed to `forward`; this value is returned for logging only.
    """

    log_probs: torch.Tensor
    entropy: torch.Tensor
    aux_loss: torch.Tensor | None = None


class TrainingClientProtocol(Protocol):
    """Interface a training backend must implement to be passed as `training_client` to a TRL trainer.

    The default [`LocalTrainingClient`] runs the model in-process, which is the behavior trainers had before this
    interface existed. Implement this protocol to run the forward and backward passes somewhere else (another process,
    another set of GPUs, or a remote service) while the trainer keeps owning the loss.

    Calls always alternate: one `forward`, then at most one `backward` for the tensor it returned. A backend may
    therefore hold the autograd graph (or the remote request handle) from `forward` until `backward` consumes it.
    `backward` is not called when the trainer skips a step, so a backend must tolerate a `forward` whose gradient never
    arrives.
    """

    def forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        aux_loss_coef: float = 0.0,
    ) -> ForwardOutput:
        """Run the forward pass and return per-token log probs.

        Args:
            model (`torch.nn.Module`):
                The trainer's prepared model. In-process backends run it; off-process backends own their own weights
                and ignore this argument.
            input_ids (`torch.Tensor`):
                Token ids, shape `(batch_size, sequence_length)`.
            position_ids (`torch.Tensor`):
                Position ids, same shape as `input_ids`. In padding-free mode, sequences are concatenated into a single
                row and boundaries are the positions where this resets to zero.
            completion_mask (`torch.Tensor`):
                1 for tokens the policy generated, 0 for prompt and tool-result tokens, same shape as `input_ids`. Only
                masked-in positions need log probs.
            aux_loss_coef (`float`, *optional*, defaults to `0.0`):
                Coefficient for the mixture-of-experts auxiliary loss, already scaled for gradient accumulation. The
                backend adds `aux_loss_coef * aux_loss` to its own backward pass. `0.0` disables it.
        """
        ...

    def backward(self, grad_log_probs: torch.Tensor) -> None:
        """Apply the gradient of the trainer's loss with respect to the log probs returned by `forward`.

        The backend backpropagates `sum(grad_log_probs * log_probs)` into the model. That sum is a first-order
        surrogate whose gradient with respect to every parameter equals the gradient of the trainer's real loss, so the
        loss itself never has to cross the boundary.

        Args:
            grad_log_probs (`torch.Tensor`):
                `d(loss)/d(log_probs)`, same shape as the `log_probs` returned by `forward`.
        """
        ...

    def clip_grad_norm(self, model: torch.nn.Module, accelerator: "Accelerator", max_grad_norm: float) -> torch.Tensor:
        """Clip the gradients accumulated by `backward` and return the global norm before clipping.

        Gradients live wherever the backend put them, so the backend is what can clip them. The optimizer step itself
        is not part of this interface: [`~transformers.Trainer`] already takes one through its `optimizers` argument,
        so a backend supplies an optimizer whose `step` reaches its own parameters.

        Args:
            model (`torch.nn.Module`):
                The trainer's prepared model. In-process backends clip its parameters; off-process backends clip their
                own copy and ignore this argument.
            accelerator (`accelerate.Accelerator`):
                The trainer's accelerator. In-process backends clip through it, since sharded gradients need its
                collectives. Off-process backends ignore it.
            max_grad_norm (`float`):
                Maximum global gradient norm.
        """
        ...


class LocalTrainingClient:
    """Runs the model in the trainer's own process.

    The default backend, and the reference implementation of [`TrainingClientProtocol`]. It keeps the autograd graph
    from `forward` alive and resumes it in `backward`, so there is no second forward pass and no numerical difference
    from computing the loss inline: gradients are bit-identical to the pre-interface behavior.

    Examples:

    ```python
    >>> client = LocalTrainingClient()
    >>> output = client.forward(model, input_ids, position_ids, completion_mask)
    >>> loss = my_loss(output.log_probs)
    >>> loss.backward()  # populates output.log_probs.grad
    >>> client.backward(output.log_probs.grad)
    ```
    """

    def __init__(self):
        self._log_probs = None
        self._aux_loss = None
        self._aux_loss_coef = 0.0

    def forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        aux_loss_coef: float = 0.0,
    ) -> ForwardOutput:
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )
        # Hold the differentiable tensor; hand out a leaf so the trainer's `loss.backward()` stops at the
        # boundary and deposits `d(loss)/d(log_probs)` in `.grad` instead of reaching the model directly.
        self._log_probs = outputs["log_probs"]
        self._aux_loss = outputs["aux_loss"] if aux_loss_coef else None
        self._aux_loss_coef = aux_loss_coef
        return ForwardOutput(
            log_probs=self._log_probs.detach().requires_grad_(True),
            entropy=outputs["entropy"].detach(),
            aux_loss=self._aux_loss.detach() if self._aux_loss is not None else None,
        )

    def backward(self, grad_log_probs: torch.Tensor) -> None:
        # Trainers call this from a backward hook on `log_probs`, where grad mode is off, so building the surrogate
        # there would produce a tensor with no graph to back-propagate.
        with torch.enable_grad():
            surrogate = (self._log_probs * grad_log_probs).sum()
            if self._aux_loss is not None:
                surrogate = surrogate + self._aux_loss_coef * self._aux_loss
        surrogate.backward()
        self._log_probs = None
        self._aux_loss = None

    def clip_grad_norm(self, model: torch.nn.Module, accelerator: "Accelerator", max_grad_norm: float) -> torch.Tensor:
        return accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
