# Generalized Knowledge Distillation (GKD) Trainer

TRL support the Genralized Knowledge Distillation (GKD) Trainer which is a new approach to distill a student model from a larger teacher model, as described in the paper [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes
](https://arxiv.org/abs/2306.13649).  The key aspects of GKD are:
1. It addresses the train-inference distribution mismatch in auto-regressive sequence models by training the student model on its self-generated output sequences.
2. GKD allows flexibility in choosing different divergence measures between student and teacher models, which can be useful when the student lacks the capacity to fully mimic the teacher.


The GKD Trainer is a wrapper around the `SFTTrainer` class that takes in a teacher model argument.



## GKDTrainer

[[autodoc]] GKDTrainer

## GKDConfig

[[autodoc]] GKDConfig
