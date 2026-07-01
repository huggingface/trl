# PAPO Trainer

[![model badge](https://img.shields.io/badge/All_models-PAPO-blue)](https://huggingface.co/models?other=papo,trl)

TRL supports the Perception-Aware Policy Optimization (PAPO) as described in the paper [Perception-Aware Policy Optimization for Multimodal Reasoning](https://huggingface.co/papers/2507.06448) by [Zhenhailong Wang](https://huggingface.co/mikewang), Xuehang Guo, Sofia Stoica, [Haiyang Xu](https://huggingface.co/xhyandwyy), Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, Heng Ji

The abstract from the paper is the following:

> Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be a highly effective strategy for endowing Large Language Models (LLMs) with robust multi-step reasoning abilities. However, its design and optimizations remain tailored to purely textual domains, resulting in suboptimal performance when applied to multimodal reasoning tasks. In particular, we observe that a major source of error in current multimodal reasoning lies in the perception of visual inputs. To address this bottleneck, we propose Perception-Aware Policy Optimization (PAPO), a simple yet effective extension of GRPO that encourages the model to learn to perceive while learning to reason, entirely from internal supervision signals. Notably, PAPO does not rely on additional data curation, external reward models, or proprietary models. Specifically, we introduce the Implicit Perception Loss in the form of a KL divergence term to the GRPO objective, which, despite its simplicity, yields significant overall improvements (4.4%) on diverse multimodal benchmarks. The improvements are more pronounced, approaching 8.0%, on tasks with high vision dependency. We also observe a substantial reduction (30.5%) in perception errors, indicating improved perceptual capabilities with PAPO. We conduct comprehensive analysis of PAPO and identify a unique loss hacking issue, which we rigorously analyze and mitigate through a Double Entropy Loss. Overall, our work introduces a deeper integration of perception-aware supervision into RLVR learning objectives and lays the groundwork for a new RL framework that encourages visually grounded reasoning. Project page: https://mikewangwzhl.github.io/PAPO.

## PAPOTrainer

[[autodoc]] experimental.papo.PAPOTrainer
    - train
    - save_model
    - push_to_hub

## PAPOConfig

[[autodoc]] experimental.papo.PAPOConfig
