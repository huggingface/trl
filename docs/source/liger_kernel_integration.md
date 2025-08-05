# Liger Kernel Integration

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduce memory usage by 60%. That way, we can **4x** our context length, as described in the benchmark below. They have implemented Hugging Face compatible `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, with more to come. The kernel works out of the box with [FlashAttention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed).

With this memory reduction, you can potentially turn off `cpu_offloading` or gradient checkpointing to further boost the performance.

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

1. To use Liger-Kernel in [`SFTTrainer`], first install it by:

```bash
pip install liger-kernel
```

2. Once installed, set `use_liger_kernel` in [`SFTConfig`]. No other changes are needed!

```python
training_args = SFTConfig(
    use_liger_kernel=True,
    ...
)
```

To learn more about Liger-Kernel, visit their [official repository](https://github.com/linkedin/Liger-Kernel/).
