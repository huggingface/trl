# Logging

As reinforcement learning algorithms are historically challenging to debug, it's important to pay careful attention to logging.
By default, TRL trainers like [`PPOTrainer`] and [`GRPOTrainer`] save a lot of relevant information to supported experiment trackers like Weights & Biases (wandb) or TensorBoard.

Upon initialization, pass the `report_to` argument to the respective configuration object (e.g., [`PPOConfig`] for `PPOTrainer`, or [`GRPOConfig`] for `GRPOTrainer`):

```python
# For PPOTrainer
ppo_config = PPOConfig(
    # ...,
    report_to="wandb"  # or "tensorboard"
)

# For GRPOTrainer
grpc_config = GRPOConfig(
    # ...,
    report_to="wandb"  # or "tensorboard"
)
```

If you want to log with TensorBoard, you might also need to specify logging directories, for example, by adding `logging_dir=PATH_TO_LOGS` to the configuration object (e.g., `PPOConfig` or `GRPOConfig`).

## PPO Logging

Here's a brief explanation for the logged metrics provided in the data:

* `eps`: Tracks the number of episodes per second.
* `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current policy and reference policy.
* `objective/entropy`: The mean entropy of the policy, indicating the randomness of the actions chosen by the policy.
* `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
* `objective/rlhf_reward`: The mean RLHF reward, which is `score - non_score_reward`.
* `objective/scores`: The mean scores returned by the reward model / environment.
* `policy/approxkl_avg`: The average approximate KL divergence between consecutive PPO policies. Note that this is not the same as `objective/kl`.
* `policy/clipfrac_avg`: The average fraction of policy updates that are clipped, indicating how often the policy updates are constrained to prevent large changes.
* `loss/policy_avg`: The average policy loss, indicating how well the policy is performing.
* `loss/value_avg`: The average value loss, indicating the difference between the predicted value and the actual reward.
* `val/clipfrac_avg`: The average fraction of value function updates that are clipped, similar to `policy/clipfrac_avg` but for the value function.
* `policy/entropy_avg`: The average entropy of the policy during training, indicating how diverse the policy's actions are.
* `val/ratio`: The mean ratio of the current policy probability to the old policy probability, providing a measure of how much the policy has changed.
* `val/ratio_var`: The variance of the `val/ratio`, indicating the variability in policy changes.
* `val/num_eos_tokens`: The number of end-of-sequence (EOS) tokens generated, which can indicate the number of complete responses.
* `lr`: The current learning rate used by the optimizer.
* `episode`: The current episode count in the training process.

### Crucial values
During training, many values are logged, here are the most important ones:

1. `objective/scores`: The mean scores returned by the reward model / environment.
1. `objective/rlhf_reward`: The mean RLHF reward. This is the ultimate objective of the RLHF training. If training works as intended, this metric should keep going up.
1. `objective/non_score_reward`: The mean reward from non-score-related sources (e.g., KL penalty).

Here are some parameters that are useful to monitor for stability (when these diverge or collapse to 0, try tuning variables):

1. `loss/value_avg`: The average value loss. It will spike / NaN when not going well.
1. `val/ratio`: The mean ratio of the current policy probability to the old policy probability. This number should float around 1.0. If this `ratio` is too high (e.g., 2.0 or 1000.0) or too small (e.g., 0.1), it means the updates between consecutive policies are too drastic.
1. `policy/clipfrac_avg` and `policy/approxkl_avg`: If `val/ratio` is too high, the `ratio` is going to get clipped, resulting in high `policy/clipfrac_avg` and high `policy/approxkl_avg` as well.
1. `objective/kl`: The mean KL divergence. It should stay positive and ideally not too large, so that the policy is not too far away from the reference policy.

## GRPO Logging

Here's a brief explanation for the logged metrics provided in the data for the GRPO trainer:

* `num_tokens`: Total number of input tokens processed during training so far.

**Completions:**
* `completions/mean_length`: Mean length of all generated completions (including those not ending with an EOS token).
* `completions/min_length`: Minimum length among all generated completions.
* `completions/max_length`: Maximum length among all generated completions.
* `completions/clipped_ratio`: The ratio of completions that did not end with an EOS token before reaching the maximum generation length (i.e., they were truncated).
* `completions/mean_terminated_length`: Mean length of only those completions that successfully ended with an EOS token.
* `completions/min_terminated_length`: Minimum length among completions that ended with an EOS token.
* `completions/max_terminated_length`: Maximum length among completions that ended with an EOS token.

**Rewards:**
* `rewards/{reward_func_name}/mean`: The mean reward obtained from a specific, named reward function (e.g., `rewards/my_custom_reward/mean`). This is logged for each reward function used.
* `rewards/{reward_func_name}/std`: The standard deviation of rewards from a specific, named reward function.
* `reward`: The overall mean of the (potentially weighted and, if `args.scale_rewards` is true, normalized) rewards, after group-wise normalization (advantages).
* `reward_std`: The standard deviation of the (potentially weighted) rewards *before* group-wise normalization for advantages.

**Policy and Loss Metrics:**
* `kl`: The mean Kullback-Leibler (KL) divergence between the current policy and the reference policy. This is logged only if `beta` (the KL coefficient in `GRPOConfig`) is non-zero.
* If Liger GRPOLoss is used (`use_liger_loss: True` in `GRPOConfig`):
    *   `clip_ratio`: The fraction of policy updates where the probability ratio was clipped according to the GRPO loss's epsilon bounds.
* If standard GRPOLoss is used (`use_liger_loss: False`):
    *   `clip_ratio/low_mean`: The mean fraction of instances where the probability ratio `r_t(θ)` was clipped at the lower bound `1 - epsilon_low` (occurs when advantage is negative and ratio is below the bound).
    *   `clip_ratio/low_min`: The minimum observed fraction for `clip_ratio/low_mean` across batches/processes.
    *   `clip_ratio/high_mean`: The mean fraction of instances where the probability ratio `r_t(θ)` was clipped at the upper bound `1 + epsilon_high` (occurs when advantage is positive and ratio is above the bound).
    *   `clip_ratio/high_max`: The maximum observed fraction for `clip_ratio/high_mean` across batches/processes.
    *   `clip_ratio/region_mean`: The mean fraction of instances where the probability ratio was clipped at either the lower or upper bound.

### Crucial GRPO values
During GRPO training, monitor these values for insights into performance and stability:

1.  `reward`: This is the primary objective. It reflects the (group-wise normalized) rewards the policy is achieving. It should generally increase during successful training.
1.  `kl`: If `beta > 0`, this tracks the divergence from the reference model. Keep an eye on it to ensure the policy doesn't stray too far, which can lead to instability.
1.  `clip_ratio/*` (either `clip_ratio` for Liger loss or the more detailed `clip_ratio/...` metrics for standard loss): These indicate how often the policy updates are being constrained by the GRPO clipping mechanism. Very high values might suggest that the policy is trying to change too drastically (potentially due to large advantages or a learning rate that's too high) or that the epsilon clipping range is too restrictive.
1.  `completions/clipped_ratio`: A high ratio here indicates that the model is frequently generating completions that are cut off by `max_completion_length` rather than naturally ending with an EOS token. This might suggest issues with learning sequence termination or that `max_completion_length` is too short.
1. `rewards/{reward_func_name}/mean`: Monitoring the mean of individual reward functions can help diagnose which aspects of the desired behavior the model is learning or struggling with, especially when using multiple reward sources.
