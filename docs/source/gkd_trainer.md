# Generalized Knowledge Distillation (GKD) Trainer

TRL support the Genralized Knowledge Distillation (GKD) Trainer to distill a student model from a larger teacher model, as described in the paper [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
](https://arxiv.org/abs/2306.13649).

The GKD Trainer is a wrapper around the `SFTTrainer` class that takes in a teacher model argument.

## How GKD works


## GKDTrainer

[[autodoc]] GKDTrainer

## GKDConfig

[[autodoc]] GKDConfig
