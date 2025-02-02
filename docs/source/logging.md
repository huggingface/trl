# Logging

As reinforcement learning algorithms are historically challenging to debug, it's important to pay careful attention to logging.
By default, the TRL [`PPOTrainer`] saves a lot of relevant information to wandb or tensorboard.

Upon initialization, pass one of these two options to the [`PPOConfig`]:

```
training_args = PPOConfig(..., report_to="wandb")  # or "tensorboard"
```

If you want to log with tensorboard, add the kwarg `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

## PPO Logging

Here's a brief explanation for the logged metrics provided in the data:

Key metrics to monitor. We want to maximize the reward, maintain a low KL divergence, and maximize entropy:
1. `env/reward_mean`: The average reward obtained from the environment. Alias `ppo/mean_scores`, which is used to specifically monitor the reward model.
1. `env/reward_std`: The standard deviation of the reward obtained from the environment. Alias ``ppo/std_scores`, which is used to specifically monitor the reward model.
1. `env/reward_dist`: The histogram distribution of the reward obtained from the environment.
1. `objective/kl`: The mean Kullback-Leibler (KL) divergence between the old and new policies. It measures how much the new policy deviates from the old policy. The KL divergence is used to compute the KL penalty in the objective function.
1. `objective/kl_dist`: The histogram distribution of the `objective/kl`.
1. `objective/kl_coef`: The coefficient for Kullback-Leibler (KL) divergence in the objective function. 
1. `ppo/mean_non_score_reward`: The **KL penalty** calculated by `objective/kl * objective/kl_coef` as the total reward for optimization to prevent the new policy from deviating too far from the old policy.
1. `objective/entropy`: The entropy of the model's policy, calculated by `-logprobs.sum(-1).mean()`. High entropy means the model's actions are more random, which can be beneficial for exploration.

Training stats:
1. `ppo/learning_rate`: The learning rate for the PPO algorithm.
1. `ppo/policy/entropy`: The entropy of the model's policy, calculated by `pd = torch.nn.functional.softmax(logits, dim=-1); entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)`. It measures the randomness of the policy.
1. `ppo/policy/clipfrac`: The fraction of probability ratios (old policy / new policy) that fell outside the clipping range in the PPO objective. This can be used to monitor the optimization process.
1. `ppo/policy/approxkl`: The approximate KL divergence between the old and new policies, measured by `0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)`, corresponding to the `k2` estimator in http://joschu.net/blog/kl-approx.html
1. `ppo/policy/policykl`: Similar to `ppo/policy/approxkl`, but measured by `masked_mean(old_logprobs - logprobs, mask)`, corresponding to the `k1` estimator in http://joschu.net/blog/kl-approx.html
1. `ppo/policy/ratio`:  The histogram distribution of the ratio between the new and old policies, used to compute the PPO objective.
1. `ppo/policy/advantages_mean`: The average of the GAE (Generalized Advantage Estimation) advantage estimates. The advantage function measures how much better an action is compared to the average action at a state.
1. `ppo/policy/advantages`: The histogram distribution of `ppo/policy/advantages_mean`.
1. `ppo/returns/mean`: The mean of the TD(λ) returns, calculated by `returns = advantage + values`, another indicator of model performance. See https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ for more details.
1. `ppo/returns/var`: The variance of the TD(λ) returns, calculated by `returns = advantage + values`, another indicator of model performance.
1. `ppo/val/mean`: The mean of the values, used to monitor the value function's performance.
1. `ppo/val/var` : The variance of the values, used to monitor the value function's performance.
1. `ppo/val/var_explained`: The explained variance for the value function, used to monitor the value function's performance.
1. `ppo/val/clipfrac`: The fraction of the value function's predicted values that are clipped.
1. `ppo/val/vpred`: The predicted values from the value function.
1. `ppo/val/error`: The mean squared error between the `ppo/val/vpred` and returns, used to monitor the value function's performance.
1. `ppo/loss/policy`: The policy loss for the Proximal Policy Optimization (PPO) algorithm.
1. `ppo/loss/value`: The loss for the value function in the PPO algorithm. This value quantifies how well the function estimates the expected future rewards.
1. `ppo/loss/total`: The total loss for the PPO algorithm. It is the sum of the policy loss and the value function loss.


Stats on queries, responses, and logprobs:
1. `tokens/queries_len_mean`: The average length of the queries tokens.
1. `tokens/queries_len_std`: The standard deviation of the length of the queries tokens.
1. `tokens/queries_dist`: The histogram distribution of the length of the queries tokens.
1. `tokens/responses_len_mean`: The average length of the responses tokens.
1. `tokens/responses_len_std`: The standard deviation of the length of the responses tokens.
1. `tokens/responses_dist`: The histogram distribution of the length of the responses tokens. (Costa: inconsistent naming, should be `tokens/responses_len_dist`)
1. `objective/logprobs`: The histogram distribution of the log probabilities of the actions taken by the model.
1. `objective/ref_logprobs`: The histogram distribution of the log probabilities of the actions taken by the reference model.



### Crucial values
During training, many values are logged, here are the most important ones:

1. `env/reward_mean`,`env/reward_std`, `env/reward_dist`: the properties of the reward distribution from the "environment" /  reward model
1. `ppo/mean_non_score_reward`: The mean negated KL penalty during training (shows the delta between the reference model and the new policy over the batch in the step)

Here are some parameters that are useful to monitor for stability (when these diverge or collapse to 0, try tuning variables):

1. `ppo/loss/value`: it will spike / NaN when not going well.
1. `ppo/policy/ratio`: `ratio` being 1 is a baseline value, meaning that the probability of sampling a token is the same under the new and old policy. If the ratio is too high like 200, it means the probability of sampling a token is 200 times higher under the new policy than the old policy. This is a sign that the new policy is too different from the old policy, which will likely cause overoptimization and collapse training later on.
1. `ppo/policy/clipfrac` and `ppo/policy/approxkl`: if `ratio` is too high, the `ratio` is going to get clipped, resulting in high `clipfrac` and high `approxkl` as well.
1. `objective/kl`: it should stay positive so that the policy is not too far away from the reference policy.
1. `objective/kl_coef`: The target coefficient with [`AdaptiveKLController`]. Often increases before numerical instabilities.
