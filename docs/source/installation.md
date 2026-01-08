# Installation

You can install TRL either from PyPI or from source.

## PyPI

Install the library with pip or [uv](https://docs.astral.sh/uv/):

<hfoptions id="install">
<hfoption id="uv">

uv is a fast Rust-based Python package and project manager. Refer to [Installation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

```bash
uv pip install trl
```

</hfoption>
<hfoption id="pip">

```bash
pip install trl
```

</hfoption>
</hfoptions>

## Optional dependencies

TRL provides optional extras for specific use cases. Install them with `pip install trl[extra_name]`:

| Extra | Description | Example |
|-------|-------------|---------|
| `peft` | LoRA/QLoRA fine-tuning with [PEFT](https://github.com/huggingface/peft) | `pip install trl[peft]` |
| `quantization` | 4-bit/8-bit quantization with [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | `pip install trl[quantization]` |
| `vllm` | Fast generation with [vLLM](https://github.com/vllm-project/vllm) | `pip install trl[vllm]` |
| `liger` | Optimized kernels with [Liger](https://github.com/linkedin/Liger-Kernel) | `pip install trl[liger]` |
| `deepspeed` | Distributed training with [DeepSpeed](https://github.com/microsoft/DeepSpeed) | `pip install trl[deepspeed]` |
| `vlm` | Vision Language Model support | `pip install trl[vlm]` |

You can combine multiple extras: `pip install trl[peft,quantization]`

For the full list of extras, see [`pyproject.toml`](https://github.com/huggingface/trl/blob/main/pyproject.toml).

## Source

You can also install the latest version from source. First clone the repo and then run the installation with `pip`:

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
```

If you want the development install you can replace the pip install with the following:

```bash
pip install -e ".[dev]"
```

## What's next?

Now that TRL is installed, head to the [Quickstart](quickstart) to train your first model!
