# A2PO

[![model badge](https://img.shields.io/badge/All_models-A2PO-blue)](https://huggingface.co/models?other=a2po,trl)

TRL supports A\*-PO (Optimal Advantage Regression) as described in the paper [Accelerating RL for LLM Reasoning with Optimal Advantage Regression](https://huggingface.co/papers/2505.20686) by Kianté Brantley, Mingyu Chen, Zhaolin Gao, Jason D. Lee, Wen Sun, Wenhao Zhan, and Xuezhou Zhang.

The abstract from the paper is the following:

> Reinforcement learning (RL) has emerged as a powerful tool for fine-tuning large language models (LLMs) to improve complex reasoning abilities. However, state-of-the-art policy optimization methods often suffer from high computational overhead and memory consumption, primarily due to the need for multiple generations per prompt and the reliance on critic networks or advantage estimates of the current policy. In this paper, we propose A\*-PO, a novel two-stage policy optimization framework that directly approximates the optimal advantage function and enables efficient training of LLMs for reasoning tasks. In the first stage, we leverage offline sampling from a reference policy to estimate the optimal value function V\*, eliminating the need for costly online value estimation. In the second stage, we perform on-policy updates using a simple least-squares regression loss with only a single generation per prompt. Theoretically, we establish performance guarantees and prove that the KL-regularized RL objective can be optimized without requiring complex exploration strategies. Empirically, A\*-PO achieves competitive performance across a wide range of mathematical reasoning benchmarks, while reducing training time by up to 2× and peak memory usage by over 30% compared to PPO, GRPO, and REBEL.

## Usage

A\*-PO assumes a **binary, verifiable reward** (`r ∈ {0, 1}`) and runs in two stages:

1. **Offline value estimation.** Before training, `num_value_samples` completions are sampled from the reference policy for every prompt and scored with `reward_funcs`. The optimal value `V*(x) = β₁·log(mean_i exp(r(x, yᵢ)/β₁))` is estimated and cached per prompt.
2. **On-policy regression.** During training, a single completion is generated per prompt from the current policy. The loss is the squared error between the implicit reward `β₂·log(π(y|x)/π_ref(y|x))` and the optimal advantage `r(x, y) − V*(x)`.

```python
from trl.experimental.a2po import A2POConfig, A2POTrainer

# A*-PO assumes a binary, verifiable reward in {0, 1}.
def reward_correct(completions, ground_truth, **kwargs):
    return [float(completion.strip() == truth) for completion, truth in zip(completions, ground_truth)]

training_args = A2POConfig(
    output_dir="Qwen2.5-0.5B-A2PO",
    num_value_samples=8,  # Stage 1: samples per prompt from the reference policy to estimate V*
    beta1=0.5,  # Stage 1: KL temperature for the V* estimate
    beta2=1e-3,  # Stage 2: KL temperature for the regression target
)
trainer = A2POTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward_correct,
    args=training_args,
    train_dataset=...,
)
trainer.train()
```

Because `V*` is estimated entirely from reference-policy samples, A\*-PO cannot exceed the reference policy's Pass@K. The official implementation can be found at [ZhaolinGao/A-PO](https://github.com/ZhaolinGao/A-PO).

## A2POTrainer

[[autodoc]] experimental.a2po.A2POTrainer
    - train
    - save_model
    - push_to_hub

## A2POConfig

[[autodoc]] experimental.a2po.A2POConfig
