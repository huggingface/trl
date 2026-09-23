# Liger Kernel Integration

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduce memory usage by 60%. That way, we can **4x** our context length, as described in the benchmark below. They have implemented Hugging Face compatible `RMSNorm`, `RoPE`, `SwiGLU`, `CrossEntropy`, `FusedLinearCrossEntropy`, with more to come. The kernel works out of the box with [FlashAttention](https://github.com/Dao-AILab/flash-attention), [PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed).

With this memory reduction, you can potentially turn off `cpu_offloading` or gradient checkpointing to further boost the performance.

| Speed Up | Memory Reduction |
| --- | --- |
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

## What `use_liger_kernel=True` does

The flag controls two separate things, depending on the trainer.

**Every trainer.** Transformers patches the model's internals with Liger's Triton kernels (`RMSNorm`, `RoPE`, `SwiGLU`). This is what the speed and memory figures above measure, and it requires `liger-kernel` to be installed.

**SFT.** In addition, the loss becomes Liger's fused linear cross-entropy, which computes the loss from the hidden states without materializing the `(batch, seq_len, vocab)` logits. Because no logits are produced, metrics that need them are computed differently on this path, and `loss_type="chunked_nll"` is rejected since it is the same optimization done by TRL.

**DPO, GRPO and KTO.** In addition, the per-token log-probabilities come from TRL's own chunked implementation rather than a full-vocabulary `log_softmax`. Despite the flag name, no Liger code is involved in that computation: the fused losses TRL used to import from Liger were replaced by `_ChunkedLogProbFunction` in `trl/trainer/utils.py`. It streams the vocabulary through the LM head in chunks and recomputes in the backward pass, which lets these trainers fit roughly twice the tokens before running out of memory.

<Tip warning={true}>

On DPO, GRPO and KTO the chunked path reads `lm_head.weight` directly instead of calling the module, and calls the backbone rather than `PeftModel.forward()`. It therefore rejects, rather than silently mistraining:

| | DPO | GRPO | KTO |
| --- | --- | --- | --- |
| `use_weighting=True` (WPO) | ❌ | | |
| `compute_metrics` | ❌ | | ❌ |
| `return_outputs=True` | | | ❌ |
| PEFT adapter on `lm_head` | ❌ | ❌ | ❌ |
| Prompt-learning PEFT (PromptTuning, PrefixTuning, P-Tuning) | ❌ | ❌ | ❌ |

Set `use_liger_kernel=False` to use any of these. You then lose the model kernels as well, since one flag controls both.

</Tip>

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
