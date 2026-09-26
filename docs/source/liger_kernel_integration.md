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

## Usage

1. First, install Liger Kernel:

  ```bash
  pip install liger-kernel
  ```

2. Once installed, set `use_liger_kernel=True` in your trainer config. No other changes are needed!

<Tip warning={true}>

In DPO and KTO the flag additionally replaces the full-vocabulary `log_softmax` with TRL's chunked log-probability path, which fits roughly twice the tokens. That path does not support WPO weighting (`use_weighting`), `compute_metrics`, `return_outputs`, PEFT adapters on `lm_head`, or prompt-learning PEFT; set `use_liger_kernel=False` to use any of those.

</Tip>

<Tip warning={true}>

`use_liger_kernel=True` is deprecated in [`GRPOTrainer`] and [`RLOOTrainer`], and will be removed in v2.0.0. These trainers compute the log-probabilities with a fused LM head, so only Liger's layer kernels (`RMSNorm`, `RoPE`, `SwiGLU`) apply. Use the Hub kernels instead, with `model_init_kwargs={"use_kernels": True}`.

</Tip>

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
</hfoptions>

To learn more about Liger-Kernel, visit their [official repository](https://github.com/linkedin/Liger-Kernel/).
