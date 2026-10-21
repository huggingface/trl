# Training on AMD GPUs

TRL trainers run on AMD Instinct GPUs without code changes. PyTorch's ROCm build exposes them as `cuda` devices, so the commands in the rest of the documentation work as they are.

## Installation

Install PyTorch's ROCm build, then TRL in the same environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install trl
```

## Attention kernels

Padding-free training and packing need a FlashAttention kernel. The `kernels-community/flash-attn2` kernel has no ROCm build, so use [`kernels-community/aiter-flash-attn`](https://huggingface.co/kernels-community/aiter-flash-attn), downloaded from the Hub by the [`kernels`](https://github.com/huggingface/kernels) library (`pip install trl[kernels]`):

```python
from trl import SFTConfig

training_args = SFTConfig(
    padding_free=True,
    model_init_kwargs={"attn_implementation": "kernels-community/aiter-flash-attn@v2"},
)
```

Models that cap their attention logits (`attn_logit_softcapping`), such as Gemma 2, are not supported by this kernel.

## Generation with vLLM

GRPO and RLOO with `use_vllm=True` need vLLM's ROCm build, see the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html). This path is not covered by CI yet.
