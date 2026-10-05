# Liger Kernel Integration

> [!WARNING]
> `use_liger_kernel=True` is deprecated in [`SFTTrainer`], [`DPOTrainer`], [`KTOTrainer`], [`GRPOTrainer`] and [`RLOOTrainer`], and will be removed in v2.0.0. Use the Hub kernels instead, with `model_init_kwargs={"use_kernels": True}`.

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. With `use_liger_kernel=True`, transformers replaces the model's `RMSNorm`, `RoPE` and `SwiGLU` layers with Liger's kernels. TRL trainers compute the log-probabilities with their own fused LM head, so Liger's fused losses don't apply.

To get the same layer kernels from the Hub instead:

```python
from trl import SFTConfig

training_args = SFTConfig(..., model_init_kwargs={"use_kernels": True})
```

To learn more about Liger-Kernel, visit their [official repository](https://github.com/linkedin/Liger-Kernel/).
