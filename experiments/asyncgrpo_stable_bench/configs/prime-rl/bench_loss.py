"""PPO-clip loss for prime-rl's `[trainer.loss] type = "custom"` hook, to express SHARED_W1's clip 0.2 / 0.28.

This is the `ppo_clip_loss` example from prime-rl's docs/algorithms.md ("Custom Loss") with asymmetric bounds.
prime-rl takes one optimizer step per batch, so the "old" policy of the PPO ratio is the inference policy mu
(`inference_logprobs`). The trainer divides the returned sum by the global rl-token count (token-level
aggregation over the step).
"""

import torch
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs


def ppo_clip_loss(inputs: LossInputs, eps_low: float = 0.2, eps_high: float = 0.28) -> LossOutputs:
    mask = inputs.loss_mask
    log_ratio = (inputs.trainer_logprobs - inputs.inference_logprobs)[mask]
    advantages = inputs.advantages[mask]
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    if inputs.loss_weights is not None:
        loss = loss * inputs.loss_weights[mask]
    is_clipped = ((ratio * advantages) > (clipped * advantages)).detach().float()
    return LossOutputs(
        loss=loss.sum(), metrics={"clip_frac": is_clipped.mean() if is_clipped.numel() else loss.new_zeros(())}
    )
