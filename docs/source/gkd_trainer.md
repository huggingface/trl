# Generalized Knowledge Distillation Trainer

TRL supports Generalized Knowledge Distillation (GKD), which is a new approach to distill a student model from a larger teacher model, as described in the paper [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649). The key aspects of GKD are:
1. It addresses the train-inference distribution mismatch in auto-regressive sequence models by training the student model on its self-generated output sequences.
2. GKD allows flexibility in choosing different divergence measures between student and teacher models via the generalized Jensen-Shannon Divergence (JSD), which can be useful when the student lacks the capacity to fully mimic the teacher.

The GKD Trainer is a wrapper around the [`SFTTrainer`] class that takes in a teacher model argument. It needs two parameters to be set via the [`GKDConfig`]:
* `lmbda`:  controls the student data fraction, i.e., the proportion of on-policy student-generated outputs. When `lmbda=0.0`, the loss reduces to supervised JSD where the student is trained with the token-level probabilities of the teacher. When `lmbda=1.0`, the loss reduces to on-policy JSD, where the student generates output sequences and token-specific feedback on these sequences from the teacher. For values in between [0, 1] it is random between the two based on the `lmbda` value for each batch.
* `beta`: controls the interpolation in the generalized Jensen-Shannon Divergence.  When `beta=0` the loss approximates forward KL divergence, while for `beta=1` the loss approximates reverse KL divergence. For values in between [0,1] it interpolates between the two.

The authors find that on-policy data (high `lmbda`) performs better and the optimal `beta` varied depending on the task and evaluation method.

## GKDTrainer

[[autodoc]] GKDTrainer

## GKDConfig

[[autodoc]] GKDConfig
