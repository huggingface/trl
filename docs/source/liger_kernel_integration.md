# Liger Kernel Integration

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduce memory usage by 60%. That way, we can **4x** our context length, as described in the benchmark below. They have implemented Hugging Face compatible `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, with more to come. The kernel works out of the box with [FlashAttention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed).

With this memory reduction, you can potentially turn off `cpu_offloading` or gradient checkpointing to further boost the performance.

| Speed Up | Memory Reduction |
| --- | --- |
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

## Supported Trainers

Liger Kernel is supported in the following TRL trainers:
- **SFT** (Supervised Fine-Tuning)
- **DPO** (Direct Preference Optimization)
- **GRPO** (Group Relative Policy Optimization)
- **KTO** (Kahneman-Tversky Optimization)
- **GKD** (Generalized Knowledge Distillation)

## Usage

1. First, install Liger Kernel:

  ```bash
  pip install liger-kernel
  ```

2. Once installed, set `use_liger_kernel=True` in your trainer config. No other changes are needed!

<hfoptions id="liger">
<hfoption id="SFT">

```python
from trl import SFTConfig

training_args = SFTConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="DPO">

```python
from trl import DPOConfig

training_args = DPOConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="KTO">

```python
from trl import KTOConfig

training_args = KTOConfig(..., use_liger_kernel=True)
```

</hfoption>
<hfoption id="GKD">

```python
from trl.experimental.gkd import GKDConfig

training_args = GKDConfig(..., use_liger_kernel=True)
```

</hfoption>
</hfoptions>

To learn more about Liger-Kernel, visit their [official repository](https://github.com/linkedin/Liger-Kernel/).
