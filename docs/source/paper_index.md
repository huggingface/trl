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

Note that this method only has an effect when training goes slightly off-policy—for example, when `steps_per_generation > gradient_accumulation_steps` or `num_iterations > 1`. Otherwise, it is effectively equivalent to no modification.

### Policy ratio: GRPO vs. GSPO

In GSPO, the policy ratio is defined at the sequence-level. In other words, it is the ratio between the probability of the current policy generating a sequence over the old policy generating that same sequence.

The sequence likelihood is defined as:

$$
\pi_\theta (o_i \mid q) = \prod_{t=1}^{|o_i|} \pi_\theta  (o_{i,t} | q, o_{i, \lt t} ),
$$

where:

- \\( \pi_\theta \\) is the policy  \\( \pi \\) with parameters  \\(\theta\\),
- \\( o_i \\) is the  \\( i \\)-th output sequence  \\( o \\) and \\(y_{i,t}\\) is the  \\( t \\)-th token in this sequence,
- \\( q \\) is the input query,

The sequence likelihood ratio  \\( s_i (\theta) \\) is defined as:

$$
s_i (\theta) = \left(\frac{\pi_\theta (o_i | q)}{\pi_{\theta_{old}} (o_i | q)} \right)^{\frac{1}{|o_i|}}
$$

The exponent  \\( \frac{1}{|y_i|} \\) represents a sequence-length normalization, minimizing the influence of sequence lenght in sequence likelihood. In other terms, it computes the geometric mean of token probabilities, ensuring a fair comparison across sequences of varying lengths.

While GSPO defines the policy ratio at the sequence level, GRPO operates at the token level. Specifically, GRPO computes an importance ratio for each token in the sequence:

$$
w_{i,t}(\theta) = \frac{\pi_\theta (o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}} (o_{i,t} \mid q, o_{i,\lt t})}
$$

This token-level ratio is then combined with a shared advantage  \\( \hat{A}_i \\), and the GRPO objective clips and optimizes each token independently across the sequence.
