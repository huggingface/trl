# ReST Trainer

TRL supports the ReST Trainer for training language models from preference data, as described in the paper [Reinforced Self-Training (ReST) for Language
Modelingl](https://arxiv.org/abs/2308.08998) by Gulcehre, Le Paine,  Srinivasan et al., 2023. For a full example have a look at  [`examples/rest.py`](https://github.com/huggingface/trl/blob/main/examples/rest.py).

## Introduction

ReST is an algorithm for aligning LLMs  with human preferences inspired by growing batch reinforcement learning. At a high level, given an initial LLM policy model,  ReST augments a dataset by using a reward model to filter generated samples from the policy, which are then used to improve the LLM policy using offline RL algorithms.

The algorithm can be split into two phases:

1. Grow: the model is used to generate  multiple output predictions for each context to augment the training dataset;
1. Improve: A reward model ranks and filters the augmented dataset and the policy model is fine-tuned on this filtered dataset.
