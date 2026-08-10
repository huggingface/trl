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
compute. A backend scores a batch and returns per-token log probs; the trainer builds its loss on those log probs; the
backend then applies `d(loss)/d(log_probs)` to the model. Because the loss never leaves the trainer, a backend does not
need to know what GRPO is, and TRL's loss variants keep evolving without touching any backend.

That split is what makes the interface implementable off-process: `log_probs` is a plain tensor, and its gradient is a
plain tensor of the same shape.

The two methods are named for what a remote backend does on the wire: score the batch without building a graph, then
run a forward and a backward together once the weights are known. See [`TrainingClientProtocol`] for why a co-located
backend does less work than those names suggest.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class ForwardOutput:
    """Per-token quantities returned by [`TrainingClientProtocol.forward_no_grad`].

    Args:
        log_probs (`torch.Tensor`):
            Log probability of each target token, shape `(batch_size, sequence_length - 1)`. A leaf tensor with
            `requires_grad=True`: the trainer builds its loss on it, and the gradient that accumulates here is what
            gets handed back to [`TrainingClientProtocol.forward_backward`]. The tensor is differentiable even though
            the backend built it without a graph of its own.
        entropy (`torch.Tensor`):
            Per-token entropy of the model's next-token distribution, same shape as `log_probs`. Detached, since it is
            reported as a metric and never differentiated.
        aux_loss (`torch.Tensor`, *optional*):
            Mixture-of-experts router load-balancing loss, if the model produces one. Detached: it is not a function of
            `log_probs`, so it cannot reach the model through the backward pass above. The backend applies it directly,
            scaled by the `aux_loss_coef` passed to `forward_no_grad`; this value is returned for logging only.
    """

    log_probs: torch.Tensor
    entropy: torch.Tensor
    aux_loss: torch.Tensor | None = None


class TrainingClientProtocol(Protocol):
    """Interface a training backend must implement to be passed as `training_client` to a TRL trainer.

    The default [`LocalTrainingClient`] runs the model in-process, which is the behavior trainers had before this
    interface existed. Implement this protocol to run the model somewhere else (another process, another set of GPUs,
    or a remote service) while the trainer keeps owning the loss.

    Calls always alternate: one `forward_no_grad`, then at most one `forward_backward` for the tensor it returned. A
    backend may therefore hold the batch (or the remote request handle) from the first call until the second consumes
    it. `forward_backward` is not called when the trainer skips a step, so a backend must tolerate a `forward_no_grad`
    whose gradient never arrives.

    The names describe the remote case, which is the one that constrains the design. A backend that cannot carry an
    autograd graph across the boundary scores the batch with no graph, then runs a second forward together with the
    backward once the trainer's gradient is known. A co-located backend does less: it keeps the graph from the first
    call and resumes it in the second, so there is no second forward and nothing is run under `no_grad`.
    [`LocalTrainingClient`] takes that route.

    Gradient clipping and the optimizer step are deliberately absent. [`~transformers.Trainer`] already accepts an
    optimizer through its `optimizers` argument, so a backend that owns the parameters supplies one whose `step`
    reaches them and clips them on the way. That matches how existing backends are built: Tinker carries
    `grad_clip_norm` in its optimizer params, and Arctic Platform clips inside its `step` call.
    """

    def forward_no_grad(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        aux_loss_coef: float = 0.0,
    ) -> ForwardOutput:
        """Score the batch and return per-token log probs.

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
                backend adds `aux_loss_coef * aux_loss` to the backward pass it runs later. `0.0` disables it.
        """
        ...

    def forward_backward(self, grad_log_probs: torch.Tensor) -> None:
        """Run the backward pass for the gradient of the trainer's loss with respect to the log probs.

        The backend backpropagates `sum(grad_log_probs * log_probs)`. That sum is a first-order surrogate whose
        gradient with respect to every parameter equals the gradient of the trainer's real loss, so the loss itself
        never has to cross the boundary.

        A remote backend reaches those log probs by running the forward again, which is what the name refers to: the
        scoring pass above could not keep a graph, so the graph is rebuilt here and consumed immediately. A co-located
        backend skips that and resumes the graph it already holds.

        A backend usually does not need a new loss function for this. A weighted cross-entropy, `sum(-weights *
        log_probs)`, is already the same surrogate: pass `weights = -grad_log_probs`. That is how Tinker implements its
        custom-loss path on top of a fixed set of server-side losses.

        A backend whose own API already fuses the forward and the backward into one call issues that call from here,
        sending `grad_log_probs` as the per-token weights.

        Args:
            grad_log_probs (`torch.Tensor`):
                `d(loss)/d(log_probs)`, same shape as the `log_probs` returned by `forward_no_grad`.
        """
        ...


class LocalTrainingClient:
    """Runs the model in the trainer's own process.

    The default backend, and the reference implementation of [`TrainingClientProtocol`]. It keeps the autograd graph
    from `forward_no_grad` alive and resumes it in `forward_backward`, so despite the names there is no second forward
    and nothing runs under `no_grad`. Gradients are bit-identical to computing the loss inline, which is what trainers
    did before this interface existed.

    Examples:

    ```python
    >>> client = LocalTrainingClient()
    >>> output = client.forward_no_grad(model, input_ids, position_ids, completion_mask)
    >>> loss = my_loss(output.log_probs)
    >>> loss.backward()  # populates output.log_probs.grad
    >>> client.forward_backward(output.log_probs.grad)
    ```
    """

    def __init__(self):
        self._log_probs = None
        self._aux_loss = None
        self._aux_loss_coef = 0.0

    def forward_no_grad(
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

    def forward_backward(self, grad_log_probs: torch.Tensor) -> None:
        # Trainers call this from a backward hook on `log_probs`, where grad mode is off, so building the surrogate
        # there would produce a tensor with no graph to back-propagate.
        with torch.enable_grad():
            surrogate = (self._log_probs * grad_log_probs).sum()
            if self._aux_loss is not None:
                surrogate = surrogate + self._aux_loss_coef * self._aux_loss
        surrogate.backward()
        self._log_probs = None
        self._aux_loss = None
