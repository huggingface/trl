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
compute. It passes its loss down as a plain Python function of the per-token log probs, and the backend calls it on the
log probs it just produced. Because the loss is a callable rather than a name, a backend does not need to know what
GRPO is, and TRL's loss variants keep evolving without touching any backend.

A backend that owns the model in the trainer's own process runs one forward, calls `loss_fn`, and returns a loss still
connected to the graph, which the trainer back-propagates as usual. A backend that runs the model elsewhere cannot
return a connected loss, so it evaluates `loss_fn` locally on the log probs the remote side sent back, takes
`d(loss)/d(log_probs)`, and ships that gradient over the wire. The remote side then back-propagates `sum(grad_log_probs
* log_probs)`, a first-order surrogate whose gradient with respect to every parameter equals the gradient of the real
loss.

Either way `loss_fn` runs in the trainer's process, so a user-defined loss never has to exist on a backend.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class ForwardBackwardOutput:
    """What [`TrainingClientProtocol.forward_backward`] returns.

    Args:
        loss (`torch.Tensor`):
            The scalar returned by `loss_fn`, which the trainer passes to `accelerator.backward`. For an in-process
            backend this is still attached to the model's graph. For an off-process backend it is a leaf carrying a
            hook, so back-propagating it triggers the remote backward instead.
        log_probs (`torch.Tensor`):
            Log probability of each target token, shape `(batch_size, sequence_length - 1)`. Detached, and provided for
            metrics: `loss_fn` already received the differentiable version.
        entropy (`torch.Tensor`):
            Per-token entropy of the model's next-token distribution, same shape as `log_probs`. Detached, since it is
            reported as a metric and never differentiated.
        aux_loss (`torch.Tensor`, *optional*):
            Mixture-of-experts router load-balancing loss, if the model produces one. Detached, and reported for
            logging only: `aux_loss_coef * aux_loss` is already folded into `loss`.
    """

    loss: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    aux_loss: torch.Tensor | None = None


class TrainingClientProtocol(Protocol):
    """Interface a training backend must implement to be passed as `training_client` to a TRL trainer.

    The default [`LocalTrainingClient`] runs the model in-process, which is the behavior trainers had before this
    interface existed. Implement this protocol to run the model somewhere else (another process, another set of GPUs,
    or a remote service) while the trainer keeps owning the loss.

    Gradient clipping and the optimizer step are deliberately absent. [`~transformers.Trainer`] already accepts an
    optimizer through its `optimizers` argument, so a backend that owns the parameters supplies one whose `step`
    reaches them and clips them on the way. That matches how existing backends are built: Tinker carries
    `grad_clip_norm` in its optimizer params, and Arctic Platform clips inside its `step` call.
    """

    def forward_backward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        aux_loss_coef: float = 0.0,
    ) -> ForwardBackwardOutput:
        """Run the forward pass, build the trainer's loss on it, and prepare the backward.

        The backward itself is left to the trainer, which calls `accelerator.backward` on the returned loss. Keeping it
        there is what preserves gradient accumulation, the grad scaler, and the `no_sync` context that DDP relies on.

        An off-process backend returns a detached loss carrying a hook, and does its remote backward from inside that
        hook. Grad mode is off there, so anything the hook recomputes must re-enable it explicitly.

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
            loss_fn (`Callable[[torch.Tensor], torch.Tensor]`):
                The trainer's loss, as a function of per-token log probs of shape `(batch_size, sequence_length - 1)`.
                It closes over everything algorithm-shaped (advantages, old log probs, the mask, clipping bounds), so
                the backend never sees any of it. Called exactly once, in the trainer's process, with grad enabled.
            aux_loss_coef (`float`, *optional*, defaults to `0.0`):
                Coefficient for the mixture-of-experts auxiliary loss, already scaled for gradient accumulation. The
                backend folds `aux_loss_coef * aux_loss` into the returned loss. `0.0` disables it.
        """
        ...


class LocalTrainingClient:
    """Runs the model in the trainer's own process.

    The default backend, and the reference implementation of [`TrainingClientProtocol`]. One forward pass, and the
    returned loss is still attached to it, so the trainer's backward reaches the model directly. Gradients are
    bit-identical to computing the loss inline, which is what trainers did before this interface existed.

    Examples:

    ```python
    >>> client = LocalTrainingClient()
    >>> outputs = client.forward_backward(model, input_ids, position_ids, completion_mask, loss_fn=my_loss)
    >>> outputs.loss.backward()
    ```
    """

    def forward_backward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        aux_loss_coef: float = 0.0,
    ) -> ForwardBackwardOutput:
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )
        log_probs = outputs["log_probs"]
        loss = loss_fn(log_probs)

        aux_loss = outputs["aux_loss"] if aux_loss_coef else None
        if aux_loss is not None:
            loss = loss + aux_loss_coef * aux_loss

        return ForwardBackwardOutput(
            loss=loss,
            log_probs=log_probs.detach(),
            entropy=outputs["entropy"].detach(),
            aux_loss=aux_loss.detach() if aux_loss is not None else None,
        )
