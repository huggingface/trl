# Kernels Hub Integration and Usage

<img src="https://github.com/user-attachments/assets/4b5175f3-1d60-455b-8664-43b2495ee1c3" width="450" height="450" alt="kernel-builder logo">

The [`kernels`](https://huggingface.co/blog/hello-hf-kernels#get-started-and-next-steps) library allows optimized compute kernels to be loaded directly from the Hub.  
You can find `kernels` in [dedicated orgs](https://huggingface.co/kernels-community) or by searching for the [`kernel` tag](https://huggingface.co/models?other=kernel) within the Hub.  

Kernels are **optimized code pieces** that help in model development, training, and inference. Here, we’ll focus on their **integration with TRL**, but check out the above resources to learn more about them.

## Installation

To use kernels with TRL, you'd need to install the library in your Python environment:

```bash
pip install kernels
```

## Using Kernels from the Hub in TRL

Kernels can directly replace attention implementations, removing the need to manually compile attention backends like Flash Attention and boosting training speed just by pulling the respective attention kernel from the Hub.

You can specify a kernel when loading a model:


```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    attn_implementation="kernels-community/flash-attn2"  # other options: kernels-community/vllm-flash-attn3, kernels-community/paged-attention
)
```

Or when running a TRL training script:

```bash
python sft.py ... --attn_implementation kernels-community/flash-attn2
```

Or using the TRL CLI:

```bash
trl sft ... --attn_implementation kernels-community/flash-attn2
```

> [!TIP]
> Now you can leverage faster attention backends with a pre-optimized kernel for your hardware configuration from the Hub, speeding up both development and training.

## Choosing and Pinning a Kernel Version

Kernel repositories on the Hub are versioned as branches (`v1`, `v2`, `v3`, ...), and Transformers selects a default version for each repository. That default can change from one Transformers release to the next, so the same `attn_implementation` value does not always resolve to the same build.

To control which build is loaded, append a revision to the repository id, either a version branch or a commit SHA:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    attn_implementation="kernels-community/flash-attn2@v2",  # or a commit SHA to pin an exact build
)
```

Pinning is useful to make a training run reproducible, and to stay on a known-good build when a newer one regresses. Keep in mind that each version only ships builds for a range of Torch and CUDA versions, so a pinned version may have no variant for your environment.

> [!TIP]
> `attn_implementation="flash_attention_2"` falls back to the `kernels-community/flash-attn2` Hub kernel when the `flash-attn` package is not installed, so you may be using a Hub kernel without having asked for one explicitly.

> [!WARNING]
> The `v3` builds for CUDA 12.8 currently fail in the backward pass for models using grouped-query attention. If your environment uses that CUDA version, pin `@v2` until [huggingface/kernels-community#1085](https://github.com/huggingface/kernels-community/issues/1085) is closed.

## Comparing Attention Implementations

We evaluated various attention implementations available in transformers, along with different kernel backends, using **TRL** and **SFT**.  
The experiments were run on a single **H100 GPU** with **CUDA 12.9**, leveraging **Qwen3-8B** with a **batch size of 8**, **gradient accumulation of 1**, and **bfloat16** precision.  
Keep in mind that the results shown here are specific to this setup and may vary with different training configurations.

The following figure illustrates both **latency** (time per training step) and **peak allocated memory** for the different attention implementations and kernel backends.  
Kernel-based implementations perform on par with custom-installed attention, and increasing the model’s `max_length` further enhances performance. Memory consumption is similar across all implementations, showing no significant differences. We get the same performance but with less friction, as described in [the following section](#flash-attention-vs-hub-kernels).

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/kernels_guide_latency.png" alt="Latency and Memory Usage" width="45%"/>
  <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/kernels_guide_peak_allocated_memory.png" alt="Latency and Memory Usage" width="45%"/>
</div>

## Flash Attention vs. Hub Kernels

Building Flash Attention from source can be time-consuming, often taking anywhere from several minutes to hours, depending on your hardware, CUDA/PyTorch configuration, and whether precompiled wheels are available.  

In contrast, **Hugging Face Kernels** provide a much faster and more reliable workflow. Developers don’t need to worry about complex setups—everything is handled automatically. In our benchmarks, kernels were ready to use in about **2.5 seconds**, with no compilation required. This allows you to start training almost instantly, significantly accelerating development. Simply specify the desired version, and `kernels` takes care of the rest.

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
    attn_implementation="kernels-community/flash-attn2"  # choose the desired FlashAttention variant
)

training_args = SFTConfig(
    use_liger_kernel=True,
    # ... other TRL training args
)
```

Learn more about the [Liger Kernel Integration](./liger_kernel_integration).
