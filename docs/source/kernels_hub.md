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

We compared different attention implementations supported in Transformers and various kernels using **TRL** and **SFT**. The tests were conducted on a machine with a single **H100 GPU**, using **Qwen3-8B**, a **batch size of 8**, **gradient accumulation of 1**, and **bfloat16** precision. These numbers are illustrative for this particular setup and may vary depending on the final training configuration.

### Latency

Latency measures the time taken for a training step, which is particularly relevant since we are conducting training using TRL. The results below show that kernel-based implementations provide noticeable improvements over more naive attention approaches. Interestingly, increasing the `max_length` of the model appears to further enhance performance.

[PLOT]

### Memory Usage

A similar trend is observed when considering memory usage. Kernel-based implementations tend to be more memory-efficient compared to naive attention, especially as model sequence lengths increase.

[PLOT]

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

Learn more about this integration [here](./liger_kernel_integration).

## Benchmarking: Flash Attention (Build-from-Source) vs. Hub Kernels

Building Flash Attention from source can be highly time-consuming, often taking several minutes to hours, depending on your hardware, CUDA/PyTorch configuration, and the availability of precompiled wheels.  

By contrast, **Hugging Face Kernels** deliver a much faster and more reliable workflow. In our benchmarks, kernels were ready to use in about **2.5 seconds**, with no compilation required. This means you can start training almost instantly, significantly accelerating development. All you need to do is specify the desired version, and `kernels` handles the rest.
