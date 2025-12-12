# Paper Index

> [!WARNING]
> Section under construction. Feel free to contribute! See https://github.com/huggingface/trl/issues/4407.

## Group Relative Policy Optimization

Papers relating to the [`GRPOTrainer`].

### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

**📜 Paper**: https://huggingface.co/papers/2402.03300

Introduces Group Relative Policy Optimization (GRPO) and shows strong math-reasoning gains from math-centric pretraining plus group-relative PPO-style optimization. Used in TRL via [`GRPOTrainer`].

```python
from trl import GRPOConfig, GRPOTrainer

# The paper doesn't specify its hyperparameters, so here we provide hyperparameters from "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning" instead.
training_args = GRPOConfig(
    loss_type="grpo",
    beta=0.001,  # "the KL coefficient to 0.001"
    epsilon=10.0, # "the GRPO clip ratio ϵ to 10"
    num_generations=16,  # "For each question, we sample 16 outputs..."
    max_completion_length=32_768,  # "...with a maximum length of 32,768"
    steps_per_generation=16,  # "To accelerate training, each rollout generates 8,192 outputs, which are randomly split into 16 minibatches"
    # "resulting in a training batch size of 512". One way to achieve this setting with 1 device is per_device_train_batch_size=4, gradient_accumulation_steps=128
    per_device_train_batch_size=4,
    gradient_accumulation_steps=128,  
)
trainer = GRPOTrainer(
    ...,
    args=training_args,
)
```

### Group Sequence Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2507.18071

GSPO is a GRPO variant that computes importance sampling weights at the sequence level instead of per-token. To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence",
    loss_type="grpo",
    beta=0.0,  # GSPO set KL regularization to zero: https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306 
    epsilon=3e-4,  # GSPO paper (v2), section 5.1
    epsilon_high=4e-4,  # GSPO paper (v2), section 5.1
    gradient_accumulation_steps=1,
    steps_per_generation=4,  # partition rollout batch into 4 mini-batches. GSPO paper (v2), section 5.1. Must be 4 times gradient_accumulation_steps
)
```

Note that this method only has an effect when training goes slightly off-policy—for example, when `steps_per_generation > gradient_accumulation_steps` or `num_iterations > 1`. Otherwise, it is effectively equivalent to no modification.

TRL also provide an experimental implementation of GSPO-token, see [Experimental - GSPO-Token](experimental#gspo-token).

#### Policy ratio: GRPO vs. GSPO

In GSPO, the policy ratio is defined at the sequence-level. In other words, it is the ratio between the probability of the current policy generating a sequence over the old policy generating that same sequence.

The sequence likelihood is defined as:

$$
\pi_\theta (o_i | q) = \prod_{t=1}^{|o_i|} \pi_\theta  (o_{i,t} | q, o_{i, < t} ),
$$

where  \\( \pi_\theta \\) is the policy  \\( \pi \\) with parameters  \\(\theta\\),  \\( o_i \\) is the  \\( i \\)-th output sequence  \\( o \\) and  \\(o_{i,t}\\) is the  \\( t \\)-th token in this sequence,  \\( q \\) is the input query. The sequence likelihood ratio  \\( s_i (\theta) \\) is defined as:

$$
s_i (\theta) = \left(\frac{\pi_\theta (o_i | q)}{\pi_{\theta_{old}} (o_i | q)} \right)^{\frac{1}{|o_i|}}
$$

The exponent  \\( \frac{1}{|o_i|} \\) represents a sequence-length normalization, minimizing the influence of sequence length in sequence likelihood. In other terms, it computes the geometric mean of token probabilities, ensuring a fair comparison across sequences of varying lengths.

While GSPO defines the policy ratio at the sequence level, GRPO operates at the token level. Specifically, GRPO computes an importance ratio for each token in the sequence:

$$
w_{i,t}(\theta) = \frac{\pi_\theta (o_{i,t} | q, o_{i,< t})}{\pi_{\theta_{\text{old}}} (o_{i,t} | q, o_{i,< t})}
$$

This token-level ratio is then combined with a shared advantage  \\( \hat{A}_i \\), and the GRPO objective clips and optimizes each token independently across the sequence.

### DAPO: An Open-Source LLM Reinforcement Learning System at Scale

**📜 Paper**: https://huggingface.co/papers/2503.14476

The DAPO algorithm includes 5 key components:

- Overlong Filtering
- Clip-Higher
- Soft Overlong Punishment
- Token-level Loss
- Dynamic Sampling (⚠️ Not supported in TRL)

To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    # Overlong Filtering
    mask_truncated_completions=True,
    # Token-level Loss
    loss_type="dapo",
    # Clip-Higher
    epsilon_high=0.28, # DAPO paper: section 4.1
    epsilon=0.2, # DAPO paper: section 4.1
    # Other parameters used
    per_device_train_batch_size=512, # mini-batch size for training in the paper, DAPO paper: section 4.1
    num_generations=16, # number of sample responses in the paper, DAPO paper: section 4.1
    max_completion_length=20480, #  maximum number of tokens for generation in the paper, DAPO paper: section 4.1
    beta=0.0, # section 2.3, DAPO paper

)
# Soft Overlong Punishment
sop_reward = get_soft_overlong_punishment(max_completion_len=20480, soft_punish_cache=4096) # DAPO paper: section 4.1
trainer = GRPOTrainer(
    ...,
    args=training_args,
    reward_funcs=[..., sop_reward],
)
```

### Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

**📜 Paper**: https://huggingface.co/papers/2506.01939

A minority of tokens with high entropy act as reasoning "forks" in the CoT path, driving exploration and performance gains for RLVR, while low-entropy majority tokens contribute little or even impede learning. RLVR mainly adjusts high-entropy tokens, largely preserving the base model’s overall entropy patterns. Thus landing on the 80/20 rule, training on only 20% of the tokens with the highest entropy is comparable or supasses full-gradient updates for Qwen3 models.

The paper's main results use vanilla DAPO (⚠️ Dynamic Sampling is not supported in TRL). To replicate the main results, use the following configuration:

```python
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_soft_overlong_punishment

training_args = GRPOConfig(
    # --- vanilla DAPO parameters (80/20 rule: section 5.2) --- #
    # Overlong Filtering
    mask_truncated_completions=True,
    # Token-level Loss
    loss_type="dapo",
    # Clip-Higher
    epsilon_high=0.28, # DAPO paper: section 4.1
    epsilon=0.2, # DAPO paper: section 4.1
    # Other parameters used
    per_device_train_batch_size=512, # mini-batch size for training in the paper, DAPO paper: section 4.1
    num_generations=16, # number of sample responses in the paper, DAPO paper: section 4.1
    max_completion_length=20480, #  maximum number of tokens for generation in the paper, DAPO paper: section 4.1
    beta=0.0, # section 2.3, DAPO paper
    # --- Gradients on the highest entropy tokens --- #
    top_entropy_quantile=0.2
)
# Soft Overlong Punishment
sop_reward = get_soft_overlong_punishment(max_completion_len=20480, soft_punish_cache=4096) # DAPO paper: section 4.1
trainer = GRPOTrainer(
    ...,
    args=training_args,
    reward_funcs=[..., sop_reward],
)
```

### Dr. GRPO: Understanding R1-Zero-Like Training: A Critical Perspective

**📜 Paper**: https://huggingface.co/papers/2503.20783

A study of R1-Zero training identifies pretraining effects on RL performance and proffers Dr. GRPO to enhance token efficiency, achieving superior accuracy on AIME 2024. To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    loss_type="dr_grpo",
    per_device_train_batch_size=1, # train_batch_size_per_device in the Training section of the repository
    num_generations=8, #  num_samples in the Training section of the repository
    max_completion_length=3000, # generate_max_length in the Training section of the repository
    beta=0.0, # beta in the Training section of the repository
)
```

### Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning (Lite PPO)

**📜 Paper**: https://huggingface.co/papers/2508.08221

The authors of this paper find that the combination of:

1. scaling rewards by the standard deviation computed over the entire batch and
2. aggregating loss over the total number of tokens

can unlock the learning capability of critic-free policies using vanilla PPO loss. Their results demonstrate that this simple combination consistently improves performance, surpassing strategies like GRPO and [DAPO](https://huggingface.co/papers/2503.14476).

TRL supports using these learnings to train a GRPO model by:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...
    scale_rewards="batch",
    loss_type="dapo",
    # Other parameters used
    beta=0.0,  # = init_kl_coef in the paper
    top_p=0.99,
    top_k=100,
    temperature=0.99,
    num_generations=8, # = num_return_sequences in the paper
    num_iterations=1,  # = ppo_epochs in the paper
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    steps_per_generation=8,  # (rollout_batch_size*num_return_sequences) / (per_device_train_batch_size*gradient_accumulation_steps)
)
```

Note that when using gradient accumulation, the loss is aggregated over the total number of tokens in the batch, but not over the accumulated batch. For more details, see the [GRPO Trainer - Loss types](grpo_trainer#loss_types).

### Truncated Importance Sampling

**📰 Blog**: https://fengyao.notion.site/off-policy-rl

Online policy learning methods commonly use an optimized inference framework for rollout generation (e.g vLLM) that is separate from the training backend. This introduces a rollout-training mismatch, exemplified in the following PPO objective:

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{inference}}}(\theta_{\mathrm{old}})}
\Bigl[
\min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}
$$

Despite  \\( \textcolor{red}{\pi_{\text{inference}}} \\) and  \\( \textcolor{blue}{\pi_{\text{training}}} \\) sharing the same model parameters  \\( \theta \\), they can produce significantly different token probabilities. This unexpected behavior implicitly breaks the on-policy assumption, and silently turns training off-policy.

Truncated Importance Sampling (TIS) addresses this issue by adapting the model update via importance-sampling correction. The gradient computation of the aforementioned PPO objective becomes

$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{inference}}}(\theta_{\mathrm{old}})}
\Bigl[
\underbrace{\min(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}{\textcolor{red}{\pi_{\text{inference}}}(a, \theta_{\mathrm{old}})}, C)}_{\text{truncated importance ratio}} \cdot
\nabla_\theta
\min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}
$$

where  \\( C \\) is a hyper-parameter. TIS is implemented in GRPO, and is enabled by selecting a `vllm_importance_sampling_mode` variant that includes the term `truncate`, such as `"sequence_truncate"` or `"token_truncate"`.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...
    use_vllm=True,
    vllm_importance_sampling_correction=True, # default True
    vllm_importance_sampling_mode="sequence_truncate", # or "token_truncate"
    vllm_importance_sampling_cap=2.0, # hyper-parameter C
)
```

### Masked Importance Sampling

**📰 Blog**: https://ringtech.notion.site/icepop

**📰 Blog**: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

Masked Importance Sampling (MIS) addresses the same issue as [Truncated Importance Sampling](#truncated-importance-sampling) but replaces clipping with masking. MIS takes a more decisive stance by discarding updates whose discrepancy exceeds a threshold  \\( C \\). We apply upper-side masking, so any ratio above  \\( C \\) is removed from the update.


$$
\small{
\mathbb{E}_{a\sim\textcolor{red}{\pi_{\text{inference}}}(\theta_{\mathrm{old}})}
\Bigl[
\underbrace{\mathbf{1}\left[
\frac{\pi_{\text{training}}(a, \theta_{\mathrm{old}})}
{\pi_{\text{inference}}(a, \theta_{\mathrm{old}})}
\le C
\right]
\cdot
\frac{\pi_{\text{training}}(a, \theta_{\mathrm{old}})}
{\pi_{\text{inference}}(a, \theta_{\mathrm{old}})}}_{\text{masked importance ratio}} \cdot
\nabla_\theta
\min\Bigl(
\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})}\,\hat A,
\;\mathrm{clip}\bigl(\frac{\textcolor{blue}{\pi_{\text{training}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{training}}}(a, \theta_{\mathrm{old}})},\,1-\epsilon,\,1+\epsilon\bigr)\,\hat A
\Bigr)
\Bigr]
}
$$

MIS is implemented for GRPO, and is enabled by selecting a `vllm_importance_sampling_mode` variant that includes the term `"mask"`, such as `"sequence_mask"` or `"token_mask"`.

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...
    use_vllm=True,
    vllm_importance_sampling_correction=True, # default True
    vllm_importance_sampling_mode="sequence_mask", # or "token_mask"
    vllm_importance_sampling_cap=2.0, # hyper-parameter C
)
```

### Sequence-level Importance Sampling

**📰 Blog**: https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda

The theoretically principled way to correct for the training-inference distribution shift is importance sampling, as introduced in the two papers above [Truncated Importance Sampling](#truncated-importance-sampling) and [Masked Importance Sampling](#masked-importance-sampling). However, the choice of formulation is crucial for keeping the gradient unbiased and ensuring stable training.

This work shows that sequence-level importance sampling is the sound approach for addressing the training–inference mismatch. Although token-level importance sampling achieves lower variance than a sequence-level ratio, it introduces bias and is therefore argued to be unsuitable for autoregressive models. The token-level gradient estimator is

$$
\mathbb{E}_{x\sim\mathcal{D},\, y\sim \pi^{\text{inference}}_\theta(\cdot|x)}
\Bigg[
  R(x,y)\,\cdot\,
  \sum_{t=0}^{|y|-1}
    \frac{\pi^{\text{training}}_\theta(y_t\,|\,x, y_{<t})}
         {\pi^{\text{inference}}_\theta(y_t\,|\,x, y_{<t})}
    \,\nabla_\theta \log \pi^{\text{training}}_\theta(y_t\,|\,x, y_{<t})
\Bigg]
$$
The correct, unbiased policy gradient estimator applies a single importance ratio over the entire generated sequence (trajectory)  \\( y \\), The Sequence-Level IS estimator looks like:

$$
\mathbb{E}_{x\sim\mathcal{D},\, y\sim \pi^{\text{inference}}_\theta(\cdot|x)}
\Bigg[
  \frac{\pi^{\text{training}}_\theta(y|x)}
       {\pi^{\text{inference}}_\theta(y|x)}
  \, R(x,y)\,
  \nabla_\theta \log \pi^{\text{training}}_\theta(y|x)
\Bigg]
$$

TRL exposes the Importance Sampling granularity level through the `vllm_importance_sampling_mode` configuration parameter where `"sequence_*"` modes implement a sequence-level importance sampling ratio and `"token_*"` a per-token ratio.

### Sample More to Think Less: Group Filtered Policy Optimization for Concise Reasoning

**📜 Paper**: https://huggingface.co/papers/2508.09726

See [Experimental - GFPO](experimental#gfpo).

### Perception-Aware Policy Optimization for Multimodal Reasoning

**📜 Paper**: https://huggingface.co/papers/2507.06448

A novel policy gradient algorithm that encourages VLMs to learn to perceive while learning to reason. This is a TRL adaptation. The TRL implementation is not the official one provided by the authors.
This is a TRL adaptation of PAPO. Note that this is not the official implementation. The official code can be found in [MikeWangWZHL/PAPO](https://github.com/MikeWangWZHL/PAPO).

```python
from trl.experimental.papo import PAPOConfig, PAPOTrainer

training_args = PAPOConfig(
    # PAPO-specific params
    perception_loss_weight=0.01,  # Weight for perception loss
    mask_ratio=0.6,  # 40% of image will be masked
    mask_type="random",  # Use patch masking (recommended)
    der_loss_weight1=0.02,
    der_loss_weight2=0.02,
    # ...other GRPO params...
)
trainer = PAPOTrainer(
    args=training_args,
    ...
)
```

### The Art of Scaling Reinforcement Learning

**📜 Paper**: https://huggingface.co/papers/2510.13786

A systematic study that defines a framework for analyzing and predicting reinforcement learning scaling in large language models, identifies key design choices that affect compute efficiency and propose a best-practice recipe called ScaleRL.

You can partially reproduce the ScaleRL recipe using the [`GRPOTrainer`] with the following configs:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    loss_type="cispo",
    epsilon_high=5.0,
    num_generations=16,
    scale_rewards="batch",
    cast_lm_head_to_fp32=True
)
```

### Soft Adaptive Policy Optimization

**📜 Paper**: https://huggingface.co/papers/2511.20347

Soft Adaptive Policy Optimization (SAPO), replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO.

To reproduce the paper's setting, use this configuration:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    loss_type="sapo",
    sapo_temperature_pos=1.0,  # default value
    sapo_temperature_neg=1.05,  # default value
    scale_rewards="group",
    ...
)
```

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

**📜 Paper**: https://huggingface.co/papers/2512.02556

DeepSeek-V3.2 technical report introduces several techniques to enhance the performance of GRPO. In TRL we implement the *Unbiased KL Estimate*, which corrects the K3 estimator (as used in the original GRPO implementation) to obtain an unbiased KL estimate using the importance-sampling
ratio between the current policy  \\( \pi_\theta \\) and the behavior policy  \\( \pi_{\text{old}} \\).

$$
\mathrm{D}_{\mathrm{KL}}\!\left(\pi_\theta(o_{i,t}) \,\|\, \pi_{\text{ref}}(o_{i,t})\right) =
\textcolor{red}{\frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}\mid q, o_{i,<t})}}
\left(
  \frac{\pi_{\text{ref}}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
  -
  \log \frac{\pi_{\text{ref}}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}
  - 1
\right).
$$

To enable this feature, set the `use_bias_correction_kl` parameter to `True` in the [`GRPOConfig`], and `beta > 0`:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    beta=0.001,  # the paper doesn't specify the value used, so we use the value from "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning"
    use_bias_correction_kl=True,
)
```

## Direct Policy Optimization

- Papers relating to the [`DPOTrainer`]

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**📜 Paper**: https://huggingface.co/papers/2305.18290

Direct Preference Optimization (DPO) fine-tunes language models more efficiently and with better performance compared to reinforcement learning from human feedback (RLHF), by directly optimizing policy training based on human preferences. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="sigmoid", # losses in Appendix B of the paper
    per_device_train_batch_size=64, #  batch size in Appendix B of the paper
    learning_rate=1e-6, # learning rate in Appendix B of the paper
    beta=0.1, # beta in Appendix B of the paper
)
```

### A General Theoretical Paradigm to Understand Learning from Human Preferences

**📜 Paper**: https://huggingface.co/papers/2310.12036

A new general objective,  \\( \Psi \\)PO, bypasses both key approximations in reinforcement learning from human preferences, allowing for theoretical analysis and empirical superiority over DPO. To reproduce the paper's setting, use this configuration: To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="ipo", # Section 5.1 of the paper
    per_device_train_batch_size=90, #  mini-batch size in Section C.1 of the paper
    learning_rate=1e-2, # learning rate in Section C.1 of the paper
)
```

These parameters only appear in the [published version](https://proceedings.mlr.press/v238/gheshlaghi-azar24a/gheshlaghi-azar24a.pdf)

### SLiC-HF: Sequence Likelihood Calibration with Human Feedback

**📜 Paper**: https://huggingface.co/papers/2305.10425

Sequence Likelihood Calibration (SLiC) is shown to be an effective and simpler alternative to Reinforcement Learning from Human Feedback (RLHF) for learning from human preferences in language models. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="hinge", # Section 2 of the paper
    per_device_train_batch_size=512, #  batch size in Section 3.2 of the paper
    learning_rate=1e-4, # learning rate in Section 3.2 of the paper
)
```

These parameters only appear in the [published version](https://openreview.net/pdf?id=0qSOodKmJaN)

### Towards Efficient and Exact Optimization of Language Model Alignment

**📜 Paper**: https://huggingface.co/papers/2402.00856

Efficient exact optimization (EXO) method is proposed to align language models with human preferences, providing a guaranteed and efficient alternative to reinforcement learning and direct preference optimization. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="exo_pair", # Section 3.2 of the paper
    per_device_train_batch_size=64, #  batch size in Section B of the paper
    learning_rate=1e-6, # learning rate in Section B of the paper
    beta=0.1, # $\beta_r$ in Section B of the paper
)
```

### Noise Contrastive Alignment of Language Models with Explicit Rewards

**📜 Paper**: https://huggingface.co/papers/2402.05369

A framework using Noise Contrastive Estimation enhances language model alignment with both scalar rewards and pairwise preferences, demonstrating advantages over Direct Preference Optimization. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="nca_pair", # Section 4.1 of the paper
    per_device_train_batch_size=32, #  batch size in Section C of the paper
    learning_rate=5e-6, # learning rate in Section C of the paper
    beta=0.01, # $\alpha$ in Section C of the paper
)
```

### Provably Robust DPO: Aligning Language Models with Noisy Feedback

**📜 Paper**: https://huggingface.co/papers/2403.00409

The paper introduces a robust direct preference optimization (rDPO) framework to address noise in preference-based feedback for language models, proving its sub-optimality gap and demonstrating its effectiveness through experiments. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="robust", # Section 3.1 of the paper
    per_device_train_batch_size=16, #  batch size in Section B of the paper
    learning_rate=1e-3, # learning rate in Section B of the paper
    beta=0.01, # $\beta$ in Section B of the paper,
    max_prompt_length=128, # max prompt length in Section B of the paper
    max_length=512, # max length in Section B of the paper
    label_smoothing=0.1 # label smoothing $\epsilon$ in section 6 of the paper

)
```

### Binary Classifier Optimization for Large Language Model Alignment

**📜 Paper**: https://huggingface.co/papers/2404.04656

Theoretical analysis and a new algorithm, Binary Classifier Optimization, explain and enhance the alignment of large language models using binary feedback signals. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="bco_pair", # Section 4 of the paper
    per_device_train_batch_size=128, #  batch size in Section C of the paper
    learning_rate=5e-7, # learning rate in Section C of the paper
    beta=0.01, # $\beta$ in Section C of the paper,
    max_prompt_length=1536, # max prompt length in Section C of the paper
    max_completion_length=512, # max completion length in Section C of the paper
)
```

For the unpaired version, the user should utilize [`experimental.bco.BCOConfig`] and [`experimental.bco.BCOTrainer`].

### Self-Play Preference Optimization for Language Model Alignment

**📜 Paper**: https://huggingface.co/papers/2405.00675

A self-play method called SPPO for language model alignment achieves state-of-the-art performance by approximating Nash equilibrium policy in a constant-sum game setting, outperforming other approaches with limited data. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="sppo_hard", # Section 3 of the paper
    per_device_train_batch_size=64, #  batch size in Section C of the paper
    learning_rate=5e-7, # learning rate in Section C of the paper
)
```

### Distributional Preference Alignment of LLMs via Optimal Transport

**📜 Paper**: https://huggingface.co/papers/2406.05882

Alignment via Optimal Transport (AOT) aligns large language models distributionally by penalizing violations of stochastic dominance between positive and negative sample distributions, achieving state-of-the-art performance on alignment benchmarks. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="aot", # Section 3 of the paper
)
```

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="aot_pair", # Section 3 of the paper
)
```

There is no additional hyperparameter in the paper.

### Discovering Preference Optimization Algorithms with and for Large Language Models

**📜 Paper**: https://huggingface.co/papers/2406.08414

An LLM-driven method automatically discovers performant preference optimization algorithms, leading to a new algorithm called DiscoPOP that blends logistic and exponential losses. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="discopop", # Section 3 of the paper
    per_device_train_batch_size=64, #  batch size in Section B.1 of the paper
    learning_rate=5e-7, # learning rate in Section B.1 of the paper
    beta=0.05, # $\beta$ in Section B.1 of the paper,
    discopop_tau=0.05 # $\tau$ in Section E of the paper
)
```

### Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment

**📜 Paper**: https://huggingface.co/papers/2408.06266

CLAIR and APO enhance LLM alignment through more contrastive preference pairs and controlled alignment objectives, improving model performance close to GPT4-turbo. To reproduce the paper's setting, use this configuration:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="apo_zero", # Section 4 of the paper
    per_device_train_batch_size=64, #  batch size in Section B.1 of the paper
    learning_rate=2e-7, # learning rate in Section 5.2 of the paper
    beta=0.1, # $\beta$ in Section 5.2 of the paper,
    max_prompt_length=512, # prompt length in Section 5.2 of the paper
    max_completion_length=512, # completion length in Section 5.2 of the paper
)
```

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="apo_down", # Section 4 of the paper
    per_device_train_batch_size=64, #  batch size in Section B.1 of the paper
    learning_rate=2e-7, # learning rate in Section 5.2 of the paper
    beta=0.1, # $\beta$ in Section 5.2 of the paper,
    max_prompt_length=512, # prompt length in Section 5.2 of the paper
    max_completion_length=512, # completion length in Section 5.2 of the paper
)
```

These parameters only appear in the [published version](https://aclanthology.org/2025.tacl-1.22.pdf)

### Statistical Rejection Sampling Improves Preference Optimization

**📜 Paper**: https://huggingface.co/papers/2309.06657

Proposes **RSO**, selecting stronger preference pairs via statistical rejection sampling to boost offline preference optimization; complements DPO/SLiC. They also introduce a new loss defined as:

$$
\mathcal{L}_{\text{hinge-norm}}(\pi_\theta)
= \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
\left[
\max\left(0,\; 1 - \left[\gamma \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \gamma \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right]\right)
\right]
$$

To train with RSO-filtered data and the hinge-norm loss, you can use the following code:

```python
from trl import DPOConfig, DPOTrainer

dataset = ...

def rso_accept(example):  # replace with your actual filter/score logic
    return example["rso_keep"]

train_dataset = train_dataset.filter(rso_accept)

training_args = DPOConfig(
    loss_type="hinge",
    beta=0.05,  # correspond to gamma in the paper
)

trainer = DPOTrainer(
    ...,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()

```

## Kahneman–Tversky Optimization

Papers relating to the [`experimental.kto.KTOTrainer`]

### KTO: Model Alignment as Prospect Theoretic Optimization

**📜 Paper**: https://huggingface.co/papers/2402.01306

KTO derives an alignment objective from prospect theory and learns directly from **binary** human feedback (liked/disliked), matching or surpassing DPO-style methods while handling imbalanced/noisy signals well.
To reproduce the paper's setting, you can use the default configuration of [`experimental.kto.KTOTrainer`]:

```python
from trl.experimental.kto import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

trainer = KTOTrainer(
    model=model,
    processing_class=tokenizer,
    args=KTOConfig(),
    train_dataset=...,
)
trainer.train()
```

## Supervised Fine-Tuning

Papers relating to the [`SFTTrainer`]

### EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes

**📜 Paper**: https://huggingface.co/papers/2508.00180

Bias-Corrected Exponential Moving Average (BEMA) improves the stability and efficiency of language model fine-tuning by reducing stochasticity and eliminating bias. To use BEMA with SFT as described in the paper, you can use the [`BEMACallback`]:

```python
from trl import BEMACallback, SFTTrainer

trainer = SFTTrainer(
    ...
    callbacks=[BEMACallback()],
)
```

### On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification

**📜 Paper**: https://huggingface.co/papers/2508.05629

Dynamic Fine-Tuning (DFT) improves the generalization of Large Language Models (LLMs) by dynamically rescaling gradients, outperforming standard Supervised Fine-Tuning (SFT) and showing competitive results in offline reinforcement learning.

$$
\mathcal{L}_{\text{DFT}}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ - \sum_{t=1}^{|y|} \textcolor{red}{\text{sg}\big(\pi_\theta(y_t \mid y_{<t}, x)\big)} \; \log \pi_\theta(y_t \mid y_{<t}, x) \right]
$$

where  \\( \text{sg}(\cdot) \\) is the stop-gradient operator. To use DFT with SFT as described in the paper, you can use the `loss_type="dft"` argument:

```python
from trl import SFTConfig

training_args = SFTConfig(
    loss_type="dft",
    ...
)
```

To closely match the paper’s setup, you can use the following configuration (see Sec. 4.1). Authors also mention that the hyperparameters are not very sensitive (Sec. 4.3):

```python
SFTConfig(
    loss_type="dft",
    learning_rate=5e-5,
    max_length=2048,
    # Target batch size 256; achieved via per-device batch 8 * grad accumulation 32
    per_device_train_batch_size=8,
    gradient_accumulation_steps=32,
)
```

## Parameter-Efficient Fine-Tuning (PEFT)

For general details on using PEFT with TRL, please refer to the [PEFT Integration](peft_integration) guide.

### LoRA: Low-Rank Adaptation of Large Language Models

**📜 Paper**: https://huggingface.co/papers/2106.09685

Low-Rank Adaptation (LoRA) reduces the number of trainable parameters and GPU memory usage in large-scale pre-trained models while maintaining or improving performance on downstream tasks. TRL integrates LoRA via the [PEFT library](https://huggingface.co/docs/peft/index) and can be easily enabled in any TRL trainer by passing a [`~peft.LoraConfig`] to the `peft_config` argument. Here is an example of using LoRA with the [`SFTTrainer`]:

```python
from trl import SFTTrainer
from peft import LoraConfig

trainer = SFTTrainer(
    ...,
    peft_config=LoraConfig(),
)
```

## Reinforce Leave-One-Out

Papers relating to the [`RLOOTrainer`]

### Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs

**📜 Paper**: https://huggingface.co/papers/2402.14740

RLOO is a variant of REINFORCE that reduces variance by using leave-one-out baselines. It computes rewards by comparing each sample against the average of all other samples in the batch, providing more stable gradients than standard REINFORCE. To reproduce the paper's setting, use this configuration:

```python
from trl import RLOOConfig

training_args = RLOOConfig(
    per_device_train_batch_size=512,  # section C Training Detail of the paper
    steps_per_generation=2  # section C Training Detail of the paper
    beta=0.03  # section C Training Detail of the paper
    num_generations=2,  # experiments of paper different num_generations={2,4}
    learning_rate=1e-6  # section C Training Detail of the paper
)
```

## Contrastive Preference Optimization

Papers relating to the [`experimental.cpo.CPOTrainer`]

### AlphaPO -- Reward shape matters for LLM alignment

**📜 Paper**: https://huggingface.co/papers/2501.03884

AlphaPO is a new Direct Alignment Algorithms (DAAs) method that leverages an alpha-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. To reproduce the paper's setting, use this configuration:

```python
from trl.experimental.cpo import CPOConfig

# Mistral-Instruct from Table 3 of the paper
training_args = CPOConfig(
    loss_type="alphapo",
    alpha=0.25,
    beta=2.5,
    simpo_gamma=0.1,
    learning_rate=7e-7,
    ...
)
```

## Reward Modeling

Papers relating to the [`RewardTrainer`]

### Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking

**📜 Paper**: https://huggingface.co/papers/2312.09244

This paper proposed an auxiliary loss function designed to directly learn a centered reward model. This auxiliary loss minimizes the squared sum of the rewards, encouraging the model to naturally produce mean-zero outputs and thereby resolving the issue of underdetermination.

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x,y^+,y^-) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y^+) - r_\theta(x, y^-)) \textcolor{red}{- \eta \cdot (r_\theta(x, y^+) + r_\theta(x, y^-))^2} \right].
$$

To use this auxiliary loss with [`RewardTrainer`], you can use the `center_rewards_coefficient` argument in [`RewardConfig`] as follows:

```python
from trl import RewardConfig

training_args = RewardConfig(
    center_rewards_coefficient=0.01,  # η in the paper
    ...
)
```

### Llama 2: Open Foundation and Fine-Tuned Chat Models

**📜 Paper**: https://huggingface.co/papers/2307.09288

In this paper, the authors propose to leverage their preference ratings being decomposed as a scale of four points (e.g., _significantly better_) to provide more informative feedback to the reward model. This is done by adding a margin to the loss function, which encourages the reward model to assign larger gaps in scores for pairs with higher preference ratings.

$$
\mathcal{L}(\theta) = - \mathbb{E}_{(x,y^+,y^-,\textcolor{red}{m}) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y^+) - r_\theta(x, y^-) \textcolor{red}{- m}) \right].
$$

You can add a margin to the loss by adding a `margin` column to the dataset. The following example shows how to set up a the "Margin Small" setting of the paper.

```python
def add_margin(example):
    preference_to_margin = {
        "significantly better": 1.0,
        "better": 2.0/3.0,
        "slightly better": 1.0/3.0,
        "negligibly better / unsure": 0.0,
    }
    return {"margin": preference_to_margin[example["preference_label"]]}

dataset = dataset.map(add_margin)
```

## Distillation

Papers relating to training a student model with the help of a teacher model.

### On-Policy Distillation

**📰 Blog**: https://thinkingmachines.ai/blog/on-policy-distillation/

On-Policy Distillation involves a student model generating rollouts for each batch of training data. We subsequently obtain the probability distributions for each token of the rollouts from both the student and teacher models. The student model is then optimized to minimize the negative Kullback-Leibler (KL) divergence between its own token distributions and those of the teacher model.

| Method                  | Sampling   | Reward signal |
|-------------------------|------------|---------------|
| Supervised finetuning   | off-policy | dense         |
| Reinforcement learning  | on-policy  | sparse        |
| On-policy distillation  | on-policy  | dense         |

On-Policy Distillation has been shown to outperform SFT, GRPO and can be used to restore generalization capabilities lost during SFT.

Additionally on-policy distillation is more compute efficient and is less prone to overfitting when trained with limited data.

To train a model with on-policy distillation using TRL, you can use the following configuration, with the [`experimental.gkd.GKDTrainer`] and [`experimental.gkd.GKDConfig`]:

```python
from trl.experimental.gkd import GKDConfig

training_args = GKDConfig(
    lmbda=1.0, # student produces rollouts for all batches
    beta=1.0, # to ensure reverse-kl as the loss function
    teacher_model_name_or_path="teacher-model", # specify the teacher model

)
```

Alternatively, you can use the [`GOLDTrainer`] and [`GOLDConfig`] to perform on-policy distillation with a similar configuration:

```python
from trl.experimental import GOLDConfig

config = GOLDConfig(
    lmbda=1.0, # student produces rollouts for all batches
    beta=1.0, # to ensure reverse-kl as the loss function
    teacher_model_name_or_path="teacher-model", # specify the teacher model

)
```

### Knowledge Distillation of Large Language Models

**📜 Paper**: https://huggingface.co/papers/2306.08543

MiniLLM is the first on-policy knowledge distillation method, which minimizes the sequence-level reverse KLD between the teacher and the student model and is optimized by reinforcement learning.

It is a generalized version of [Think Machine Lab's On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), with the option to add distribution-level single-step distillation signals (like GKD when `beta=1`) and long-context reverse KLD signals.

Alternatively, you can use the [`experimental.MiniLLMTrainer`] and [`experimental.MiniLLMConfig`] to perform MiniLLM distillation as follows:

```python
from datasets import load_dataset
from trl.experimental.minillm import MiniLLMTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

trainer = MiniLLMTrainer(
    model="Qwen/Qwen3-0.6B",
    teacher_model="Qwen/Qwen3-1.7B",
    train_dataset=dataset,
)
trainer.train()
```

For more details, see the [MiniLLM Trainer documentation](minillm) documentation.

## Distributed Training

### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**📜 Paper**: https://huggingface.co/papers/1910.02054

ZeRO (Zero Redundancy Optimizer) eliminates memory redundancies in data- and model-parallel training by partitioning optimizer states, gradients, and parameters across devices while retaining low communication volume and high computational granularity. This allows for the efficient training of large models that would otherwise not fit in GPU memory.

TRL supports ZeRO via the [DeepSpeed integration](deepspeed_integration). To use it, provide a DeepSpeed configuration file with your desired settings,

```yaml
# config.yaml
distributed_type: DEEPSPEED
num_processes: 2
deepspeed_config:
  zero_stage: 3
```

and launch the training script using `accelerate launch --config_file config_file`.

```sh
accelerate launch --config_file config.yaml train.py
```
