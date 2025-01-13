# GRPO Trainer

[![](https://img.shields.io/badge/All_models-GRPO-blue)](https://huggingface.co/models?other=grpo,trl)

## Overview

TRL supports the GRPO Trainer for training language models, as described in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) by [Zhihong Shao](https://huggingface.co/syhia), [Peiyi Wang](https://huggingface.co/peiyiwang89), [Qihao Zhu](https://huggingface.co/zqh11), Runxin Xu, [Junxiao Song](https://huggingface.co/haha-point), Mingchuan Zhang,Y. K. Li,Y. Wu, [Daya Guo](https://huggingface.co/guoday).

The abstract from the paper is the following:

> Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.

This post-training method was contributed by [Quentin Gallouédec](https://huggingface.co/qgallouedec).

## GRPOTrainer

[[autodoc]] GRPOTrainer

## GRPOConfig

[[autodoc]] GRPOConfig
