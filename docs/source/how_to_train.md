# Training FAQ

## What metrics should I look at?

When doing classical supervised fine-tuning of language models the loss (especially the validation loss) is a good proxy for how well the training is going. However, in RL the loss doesn't tell us much about the model's performance and it's value can go up and down while the performance continues to increase. 

For example, the value loss measures how good the model is at predicting the value of each token. If the model suddenly gets better at the task the model will likely underestimate the values and the loss increase while the reward continues to climb until learned the new distribution.

The two best metrics to look at are **mean reward** and **objective KL divergence**. The main goal is to maximize the reward but at the same time we ideally want to keep the KL divergence between [0, 10] (see below for more info on the KL divergence).

## Why a reference model? And what's the KL divergence for?

RL is very efficient and optimizing a reward sometimes at the cost of exploiting the environment in unexpected ways. When doing RLHF on language mdoels the reward is usually given by a reward model. The reward models are trained predict if a human would rank the models generation high in a comparison with other generations. The langauge model we are optimizing against that reward model can learn patterns that yield high reward but are not great language. In the best case it could learn that the reward model assigns high rewards to texts with lots exclamation marks or emojis rather than good content. In the worst case it can learn patterns that don't resemble natural language at at all but get the reward model to return high rewards (similar to adversarial attacks).

To counteract this possibility we can penalize the model when it's generation are too different from what it would have generated before the RLHF loop. To quantify the difference between the reference and the model we train we use a quantity called KL divergence. 

KL-divergence, also known as Kullback-Leibler divergence, is a measure of how one probability distribution diverges from another. It quantifies the information lost when using one distribution to approximate another, providing a non-symmetric measure of dissimilarity between them. In simple terms, it helps us understand how different two probability distributions are from each other.

To make sure we stay somewhat close to the text the reference model generates we add the KL-divergence between the active and reference model as an additional penalty to the reward: `R = r - beta * KL` (where `r` comes from reward model).

## Why is the KL-divergence negative and why is it a problem?

If you generate text by purely sampling from the model in general things work fine. But when you use the `generate` method there are a few caveats because it does not always purely sample depending on the settings which can cause KL-divergence to go negative. Essentially when the active model achieves `log_p_token_active < log_p_token_ref` we get negative KL-div. This can happen in a several cases:

- **top-k sampling**: the model can smooth out the probability distribution causing the top-k tokens having a smaller probability than those of the reference model but they still are selected
- **min_length**: this ignores the EOS token until `min_length` is reached. thus the model can assign a very high log prob to the EOS token and very low prob to all others until min_length is reached
- **batched generation**: finished sequences in a batch are padded until all generations are finished. The model can learn to assign very low probabilities to the padding tokens unless they are properly masked or removed.

These are just a few examples. Why is negative KL an issue? The total reward `R` is computed `R = r - beta * KL` so if the model can learn how to drive KL-divergence negative it effectively gets a positive reward. In many cases it can be much easier to exploit such a bug in the generation than actually learning the reward function. In addition the KL can become arbitrarily small thus the actual reward can be very small compared to it.

So how should you generate text for PPO training? Let's have a look!

## How to generate text for training?

In order to avoid the KL issues described above we recommend to use the following settings:

```python
generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token (see above)
    "top_k": 0.0, # no top-k sampling
    "top_p": 1.0, # no nucleus sampling
    "do_sample": True, # yes, we want to sample
    "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
    "max_new_tokens": 32, # specify how many tokens you want to generate at most
}
```

With these settings we usually don't encounter any issues. You can also experiments with other settings but if you encounter issues with negative KL-divergence try to go back to these and see if they persist.

## How can debug your own use-case?

Debugging the full RL pipeline can be difficult since there are many moving pieces. Here are a few tricks and suggestions to make your life easier:

- **Start from a working example**: There are several working examples in the `trl` repository. Try to start from one of those and get to your use-case step-by-step. For example, you can first just replace the model in the example and once you figure out the best hyperparameters try to switch to your dataset and reward model. If you change everything at once you won't know where a potential problem comes from.
- **Start small, scale later**: Training large models can be very slow and take several hours or days until you see any improvement. For debugging this is not a convenient timescale so try to use small model variants during the development phase and scale up once that works. That being said you sometimes have to be careful as small models might not have the capacity to solve a complicated task either.
- **Start simple**: Try to start with a minimal example and build complexity from there. Your use-case might require for example a complicated reward function consisting of many different rewards - try to use one signal first and see if you can optimize that and then add more complexity after that.
- **Inspect the generations**: It's always a good idea to inspect what the model is generating. Maybe there is a big in your post-processing or your prompt. Due to bad settings you might cut-off generations too soon. These things are very hard to see on the metrics but very obvious if you look at the generations.
- **Inspect the reward model**: If you reward is not improving over time maybe there's an issue with the reward model. You can look at extreme cases to see if it does what it should: e.g. in the sentiment case you can check if simple positive and negative examples really get different rewards. And you can look at the distribution of your dataset. Finally, maybe the reward is dominated by the query which the model can't affect so you might need to normalize this (e.g. reward of query+response minus reward of the query).

These are just a few tips that we find helpful - if you have more useful tricks feel free to open a PR to add them as well!