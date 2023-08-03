# Training FAQ

## What metrics should I look at?
When doing classical supervised fine-tuning of language models the loss (especially the validation loss) is a good proxy for how well the training is going. However, in RL the loss doesn't tell us much about the model's performance and it's value can go up and down while the performance continues to increase. 

For example, the value loss measures how good the model is at predicting the value of each token. If the model suddenly gets better at the task the model will likely underestimate the values and the loss increase while the reward continues to climb until learned the new distribution.

The two best metrics to look at are **mean reward** and **objective KL divergence**. The main goal is to maximize the reward but at the same time we ideally want to keep the KL divergence between [0, 10] (see below for more info on the KL divergence).

## Why a reference model? And what's the KL divergence for?

RL is very efficient and optimizing a reward sometimes at the cost of exploiting the environment in unexpected ways. When doing RLHF on language mdoels the reward is usually given by a reward model. The reward models are trained predict if a human would rank the models generation high in a comparison with other generations. The langauge model we are optimizing against that reward model can learn patterns that yield high reward but are not great language. In the best case it could learn that the reward model assigns high rewards to texts with lots exclamation marks or emojis rather than good content. In the worst case it can learn patterns that don't resemble natural language at at all but get the reward model to return high rewards (similar to adversarial attacks).

To counteract this possibility we can penalize the model when it's generation are too different from what it would have generated before the RLHF loop. To quantify the difference between the reference and the model we train we use a quantity called KL divergence. 

## Negative KL-divergence

KL-divergence is a quantity that measures

Pure sampling is fine and works well, but when using generate there are a few caveats because we not always purely sample which causes KL to go negative. Essentially when the active model achieves log_p_token_active < log_p_token_ref we get negative KL-div. This can happen in a few cases:
top-k sampling: the model can smooth out the probability distribution causing the top-k tokens having a smaller probability
min_length: this ignores the EOS token until min_length is reached. thus the model can assign a very high log prob to the EOS token and very low prob to all others until min_length
batched generation: finished sequences in a batch are padded until the end. the model can learn to assign very low probabilities to the padding tokens
just a few examples. why is negative KL an issue? the effective reward is R=r-beta*KL  (where r comes from reward model) so if the model can create negative KL-div it is much easier to go in that direction (just assign very low/high prob to some tokens). i suspect this becomes more and more an issue when the actual objective (as defined by the reward model) gets harder as it might be easier to exploit any of the above rather than learning the actual reward.


## How to generate?