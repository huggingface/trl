# IW-OPD

In the paper [On the Position Bias of On-Policy Distillation](https://huggingface.co/papers/2606.22600), the authors introduce Importance-Weighted On-Policy Distillation (IW-OPD). IW-OPD addresses position bias in on-policy distillation by reweighting sampled-token updates according to accumulated teacher-student prefix discrepancy. Early tokens keep larger weights, while later tokens after high drift are downweighted.

To use IW-OPD, you can use the [`IWOPDTrainer`] class in `trl.experimental.iw_opd`.

> [!NOTE]
> IW-OPD is currently part of the `trl.experimental` namespace. APIs may change without notice while the feature is iterated on.

## Usage

```python
from trl.experimental.iw_opd import IWOPDConfig, IWOPDTrainer

training_args = IWOPDConfig(
    distillation_objective="iw_opd",
    iw_opd_gamma=0.5,
)
trainer = IWOPDTrainer(
    model="Qwen/Qwen3-0.6B",
    teacher_model="...",
    train_dataset=...,
    args=training_args,
)
trainer.train()
```

IW-OPD is an on-policy objective: `distillation_objective="iw_opd"` requires `lmbda=1.0` (the default). `iw_opd_gamma` is the importance-weight amplification from Algorithm 1 of the paper.

## IWOPDTrainer

[[autodoc]] experimental.iw_opd.IWOPDTrainer
    - train
    - save_model
    - push_to_hub

## IWOPDConfig

[[autodoc]] experimental.iw_opd.IWOPDConfig
