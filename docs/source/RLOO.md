# RLOO 

# What does REINFORCE suggests?
The REINFORCE algorithm is a classic policy gradient method in reinforcement learning. It suggests using a "baseline" to reduce the variance of the policy gradient estimates. The baseline is typically the average reward observed so far (or a running average over a window of recent rewards). By subtracting this baseline from the actual reward, the algorithm focuses updates on actions that perform better or worse than average, rather than being influenced by the absolute scale of the rewards.


In practice, for each step or episode, the policy is updated using the difference between the received reward and the baseline:

$$b_{MA} = \frac{1}{S}\sum_{s} R(x_s, y_s)$$

# What RLOO do inspired from REINFORCE?

Inspired by REINFORCE's baseline approach, RLOO (Reinforcement Learning with Leave-One-Out) uses a different but related strategy for variance reduction. Instead of using a moving average, RLOO uses additional generations from the policy/language model as a mean to reduce the varience. Therefore:

For a given prompt RLOO generates k samples, lets say k=2; (note that you examine each sample individually) so one time you take first sample and then you get the reward for current sample then you take the other as baseline. Let's break this down step by step:

1. First, generate two samples $x_1$ and $x_2$ from the policy for prompt $y$

2. For the first sample $x_1$:
   - Calculate its reward $R(x_1, y)$
   - Use $x_2$ as the baseline
   - Compute the gradient:
   $$\nabla \mathcal{L}_1 = (R(x_1, y) - R(x_2, y)) \nabla \log p_\theta(x_1)$$

3. For the second sample $x_2$:
   - Calculate its reward $R(x_2, y)$
   - Use $x_1$ as the baseline
   - Compute the gradient:
   $$\nabla \mathcal{L}_2 = (R(x_2, y) - R(x_1, y)) \nabla \log p_\theta(x_2)$$

4. The final policy update combines both gradients:
   $$\nabla \mathcal{L} = \nabla \mathcal{L}_1 + \nabla \mathcal{L}_2$$

This approach is particularly elegant because:
- Both samples are generated from the current policy state
- Each sample serves as a natural baseline for the other
- The comparison is between samples from the same policy distribution
- No historical information or previous gradient updates are needed for the baseline

This approach thechnicaly ensures that:
- Each sample is evaluated independently
- The baseline for each sample comes from the other sample
- The policy is updated based on relative performance between the samples
- The variance reduction is achieved through direct comparison between samples from the same policy


# Main sources. 
1. [RLOO Paper](https://openreview.net/pdf?id=r1lgTGL5DE)
2. [back to basics Paper](https://arxiv.org/pdf/2402.14740)
3. [REINFORCE++ Paper](https://arxiv.org/html/2501.03262v1)
4. [RLOO Blog on HF](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo)
5. [RLOO OPENRLHF](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights#147d9a33ecc9806090f3d5c749d31f05)
6. [Youtube RLOO](https://www.youtube.com/watch?v=86asXGPK6RU&ab_channel=BuzzRobot)