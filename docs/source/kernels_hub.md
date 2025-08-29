# Kernels Hub Integration and Usage

<img src="https://github.com/user-attachments/assets/4b5175f3-1d60-455b-8664-43b2495ee1c3" width="450" height="450" alt="kernel-builder logo">

[`kernels`](https://huggingface.co/blog/hello-hf-kernels#get-started-and-next-steps) allow loading optimized compute kernels directly from the Hub.  
You can find `kernels` in [dedicated orgs](https://huggingface.co/kernels-community) or by searching for the [`kernel` tag](https://huggingface.co/models?other=kernel) within the Hub.  

Kernels are **optimized code pieces** that help in model development, training, and inference. Here, we’ll focus on their **integration with TRL**, but check out the above resources to learn more about them.

## Installation

To use kernels, install the library:

```bash
pip install kernels
```

## Using Kernels from the Hub in TRL

Kernels can directly replace attention implementations, removing the need to manually compile attention pipelines and boosting **training speed**.

You can specify a kernel when loading a model:


```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    attn_implementation="kernels-community/flash-attn"  # other options: kernels-community/flash-attn3, kernels-community/vllm-flash-attn3, kernels-community/paged-attention
)
```

Or when running a TRL training script:

```bash
python sft.py ... --attn_implementation kernels-community/flash-attn
```

Or using the TRL CLI:

```bash
trl sft ... --attn_implementation kernels-community/flash-attn
```

<Tip>

This replaces your attention pipeline with a pre-optimized kernel from the Hub, speeding up both development and training.

</Tip>


## Comparing Attention Implementations

[TODO!]

- Speed (throughput, latency)
- Memory usage

## Combining FlashAttention Kernels with Liger Kernels

You can combine **FlashAttention kernels** with **Liger kernels** for additional TRL performance improvements.

First, install the Liger kernel dependency:


```bash
pip install liger-kernel
```

Then, combine both in your code:

```python
from transformers import AutoModelForCausalLM
from trl import SFTConfig

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    attn_implementation="kernels-community/flash-attn"  # choose the desired FlashAttention variant
)

training_args = SFTConfig(
    use_liger_kernel=True,
    # ... other TRL training args
)
```

## Benchmarking FA Build-from-Source vs Hub Kernel

[TODO!]
- Scripts comparing throughput of FA compiled locally vs Hub kernel
- Speedups in training loops
- Memory savings for large models