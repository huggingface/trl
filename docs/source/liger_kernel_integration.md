# Liger Kernel Integration

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduce memory usage by 60%. That way, we can **4x** our context length, as described in the benchmark below. They have implemented Hugging Face compatible `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, with more to come. The kernel works out of the box with [FlashAttention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed).

With this memory reduction, you can potentially turn off `cpu_offloading` or gradient checkpointing to further boost the performance.

| Speed Up | Memory Reduction |
| --- | --- |
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

## Supported Trainers

Liger Kernel is supported in [`SFTTrainer`], where `use_liger_kernel=True` patches the model with Liger's `RMSNorm`, `RoPE`, `SwiGLU` and fused linear cross-entropy kernels.

> [!NOTE]
> The fused DPO, KTO, GRPO and JSD losses that Liger provides now live in TRL under `use_fused_linear_loss=True`, with no extra dependency. See [Fused linear loss](reducing_memory_usage#fused-linear-loss-for-reducing-peak-memory-usage). Setting `use_liger_kernel=True` on those trainers still enables them, but this is deprecated.

## Usage

1. First, install Liger Kernel:

  ```bash
  pip install liger-kernel
  ```

2. Once installed, set `use_liger_kernel=True` in [`SFTConfig`]. No other changes are needed!

```python
from trl import SFTConfig

training_args = SFTConfig(..., use_liger_kernel=True)
```

To learn more about Liger-Kernel, visit their [official repository](https://github.com/linkedin/Liger-Kernel/).
