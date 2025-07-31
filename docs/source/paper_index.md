# Paper Index

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Group Sequence Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2507.18071

GSPO is a GRPO variant that computes importance sampling weights at the sequence level instead of per-token. To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    beta=0.0,  # GSPO set kl regularization to zero: https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306 
    epsilon=3e-4,  # GSPO paper (v2), section 5.1
    epsilon_high=4e-4,  # GSPO paper (v2), section 5.1
    gradient_accumulation_steps=1,
    steps_per_generation=4,  # partition rollout batch into 4 mini-batches. GSPO paper (v2), section 5.1. Must be 4 times gradient_accumulation_steps
)
```

While the original paper doesn’t specify the hyperparameters used, this modification only has an effect when training is slightly off-policy—for example, when `steps_per_generation > gradient_accumulation_steps` or `num_iterations > 1`. Otherwise, it is effectively equivalent to no modification.

### Policy ratio: GRPO vs. GSPO

In GSPO, the policy ratio is defined at the sequence-level. In simple terms, it is the ratio between the probability of the current policy generating a sequence over the old policy generating that same sequence. 

The sequence likelihood is defined as:

```math
\pi_\theta (y_i \mid x) = \prod_{t=1}^{|y_i|} \pi_\theta (y_{i,t} \mid x, y_{i, <t})
```

Where: 
- $\pi_\theta$ is the policy $\pi$ with parameters $\theta$; 
- $y_i$ is the i-th sequence $y$;  
- $y_{i,t}$ is the t-th token in the sequence;  
- $y_{i,<t}$ denotes all tokens before $t$;  
- $x$ is the input query; 
- and $t$ is the current token index. 

This equation expresses the sequence likelihood as the product of the conditional probabilities of each token, given the query and all previously generated tokens.

The sequence likelihood ratio $s_i (\theta)$ is defined as:


```math
s_i (\theta) = \left(\frac{\pi_\theta (y_i | x)}{\pi_{\theta_{old}} (y_i | x)} \right)^{\frac{1}{|y_i|}}
```

The exponent $\frac{1}{|y_i|}$ represents a sequence-length normalization, minimizing the influence of sequence lenght in sequence likelihood. In other terms, it computes the geometric mean of token probabilities, ensuring a fair comparison across sequences of varying lengths.

Practically, it is computed in the log space, as follows:


```math
s_i(\theta) = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \left( \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})} \right) \right)
```

While GSPO defines the policy ratio at the sequence level, GRPO operates at the token level. Specifically, GRPO computes an importance ratio for each token in the sequence:

```math
w_{i,t}(\theta) = \frac{\pi_\theta (y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}} (y_{i,t} \mid x, y_{i,<t})}
```

This token-level ratio is then combined with a shared advantage $\hat{A}_i$, and the GRPO objective clips and optimizes each token independently across the sequence.

