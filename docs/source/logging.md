# Logging

As reinforcement learning algorithms are historically challenging to debug, it's important to pay careful attention to logging.
By default, the TRL [`PPOTrainer`] saves a lot of relevant information to wandb or tensorboard.

Upon initialization, pass one of these two options to the [`PPOConfig`]:

```
training_args = PPOConfig(..., report_to="wandb")  # or "tensorboard"
```

If you want to log with tensorboard, add the kwarg `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

## PPO Logging

Here's a brief explanation for the logged metrics provided in the data. For a more detailed and up-to-date explanation, please refer to the [PPO Trainer documentation](ppo_trainer).

Key metrics to monitor. We want to maximize the reward, maintain a low KL divergence, and maximize entropy:

1. `eps`: Tracks the number of episodes per second.
1. `objective/kl`: The mean Kullback-Leibler (KL) divergence between the current policy and reference policy.
1. `objective/entropy`: The mean entropy of the policy, indicating the randomness of the actions chosen by the policy.
1. `objective/non_score_reward`: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where `beta` is the KL penalty coefficient and `kl` is the per-token KL divergence.
1. `objective/rlhf_reward`: The mean RLHF reward, which is `score - non_score_reward`.
1. `objective/scores`: The mean scores returned by the reward model / environment.
1. `policy/approxkl_avg`: The average approximate KL divergence between consecutive PPO policies. Note that this is not the same as `objective/kl`.
1. `policy/clipfrac_avg`: The average fraction of policy updates that are clipped, indicating how often the policy updates are constrained to prevent large changes.
1. `loss/policy_avg`: The average policy loss, indicating how well the policy is performing.
1. `loss/value_avg`: The average value loss, indicating the difference between the predicted value and the actual reward.
1. `val/clipfrac_avg`: The average fraction of value function updates that are clipped, similar to policy/clipfrac_avg but for the value function.
1. `policy/entropy_avg`: The average entropy of the policy during training, indicating how diverse the policy's actions are.
1. `val/ratio`: The mean ratio of the current policy probability to the old policy probability, providing a measure of how much the policy has changed.
1. `val/ratio_var`: The variance of the `val/ratio`, indicating the variability in policy changes.
1. `val/num_eos_tokens`: The number of end-of-sequence (EOS) tokens generated, which can indicate the number of complete responses.
1. `lr`: The current learning rate used by the optimizer.
1. `episode`: The current episode count in the training process.

### Crucial values

During training, many values are logged, here are the most important ones:

1. `objective/rlhf_reward`: This is the ultimate objective of the RLHF training. If training works as intended, this metric should keep going up.
1. `objective/scores`: The mean scores returned by the reward model / environment.
1. `objective/kl`: The mean KL divergence between the current policy and reference policy. It should stay positive so that the policy is not too far away from the reference policy.

Here are some parameters that are useful to monitor for stability (when these diverge or collapse to 0, try tuning variables):

1. `loss/value_avg`: The average value loss. It will spike / NaN when not going well.
1. `val/ratio`: This number should float around 1.0, and it gets clipped by `--cliprange 0.2` with PPO's surrogate loss. If this `ratio` is too high like 2.0 or 1000.0 or too small like 0.1, it means the updates between consecutive policies are too drastic.
1. `policy/clipfrac_avg` and `policy/approxkl_avg`: If `ratio` is too high, the `ratio` is going to get clipped, resulting in high `clipfrac` and high `approxkl` as well.
1. `objective/kl_coef`: The coefficient for Kullback-Leibler (KL) divergence in the objective function. Often increases before numerical instabilities.
