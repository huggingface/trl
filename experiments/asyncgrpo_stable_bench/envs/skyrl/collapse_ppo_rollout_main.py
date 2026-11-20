"""skyrl_w1_main plus one extra policy loss: PPO-clip (one-sided) against the sampler's logprobs.

Same as SkyRL's `regular` loss, with the ratio taken against the rollout (vLLM) logprobs instead of a recomputed pi_old,
which is what SHARED_W1 rev. 5 asks for. Registered through SkyRL's public PolicyLossRegistry (no source patch).
"""

import skyrl_w1_main
import torch
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry, reduce_loss
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean, safe_exp_delta


def ppo_rollout_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    ratio = safe_exp_delta(log_probs - rollout_logprobs, clip=20.0, out_dtype=log_probs.dtype)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    return reduce_loss(loss, loss_mask), {"clip_ratio": clip_ratio}


PolicyLossRegistry.register("ppo_rollout", ppo_rollout_policy_loss)

if __name__ == "__main__":
    skyrl_w1_main.main()
