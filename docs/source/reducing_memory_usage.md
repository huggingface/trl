# Reducing Memory Usage

Training workflows can often be optimized to **reduce memory consumption**, and TRL provides several built-in features to help achieve this.

Below, we outline these techniques and recommend experimenting with different combinations to figure out which configuration works best for your specific setup.

Each method includes examples for the supported trainers. If you're unsure whether a technique is compatible with your trainer, please take a look at the corresponding trainer documentation.

For additional strategies, such as **gradient checkpointing**, which is supported across all trainers, see the [`transformers` performance guide](https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-checkpointing).

## Truncation

Sequence lengths in the dataset can vary widely. When data is batched, sequences are padded to match the longest one in the batch, which can cause high memory usage, even if most sequences are relatively short.

![Truncation prompt-completion](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png)

To reduce memory usage, it's important to truncate sequences to a reasonable length. While TRL trainers truncate sequences by default, you may want to adjust the default truncation length to better align with your specific use case.

<hfoptions id="truncation">
<hfoption id="DPO">

DPO truncation is applied first to the prompt and to the completion via the `max_prompt_length` and `max_completion_length` parameters. The `max_length` parameter is then used to truncate the resulting sequence.

![DPO truncation](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_prompt_completion.png)

To set the truncation parameters, use the following code snippet:

```python
from trl import DPOConfig

training_args = DPOConfig(..., max_prompt_length=..., max_length=...)
```

You can also use the `max_completion_length` parameter to truncate the completion, though this is less common since the goal is typically to preserve the completion's full length whenever possible.

```python
from trl import DPOConfig

training_args = DPOConfig(..., max_completion_length=...)
```

</hfoption>
<hfoption id="SFT">

SFT truncation is applied to the input sequence via the `max_length` parameter.

![Truncation input ids](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_input_ids.png)

To set the truncation parameter, use the following code snippet:

```python
from trl import SFTConfig

training_args = SFTConfig(..., max_length=...)
```

</hfoption>
</hfoptions>

### How to choose the `max_length` value?

If `max_length` is too small, a significant portion of your tokens will be discarded and won't contribute to training. If it's too large, memory usage can spike, potentially leading to out-of-memory (OOM) errors. Without packing or padding-free, a large `max_length` may also result in inefficient training, as many tokens will be padding.

To help you choose an appropriate value, we provide a utility to visualize the sequence length distribution in your dataset.

<iframe src="https://trl-lib-dataset-length-profiler.hf.space" frameborder="0" width="100%" height="1000"></iframe>

## Packing

> [!TIP]
> This technique is available only for **SFT** training and setups that use **FlashAttention** (or its variants).

[Truncation](#truncation) has several drawbacks:

1. **Loss of information**: Key data at the end of a sequence may be discarded.
2. **Choosing truncation length**: Too short loses data; too long undermines efficiency.

Packing, introduced in [Raffel et al., 2020](https://huggingface.co/papers/1910.10683), addresses these issues by grouping sequences instead of truncating. It concatenates and splits dataset sequences into the desired lengths.

![Packing](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing_2.png)

Packing reduces padding by merging several sequences in one row when possible. We use an advanced method to be near-optimal in the way we pack the dataset. To enable packing, use `packing=True` in the [`SFTConfig`].

> [!TIP]
> In TRL 0.18 and earlier, packing used a more aggressive method that reduced padding to almost nothing, but had the downside of breaking sequence continuity for a large fraction of the dataset. To revert to this strategy, use `packing_strategy="wrapped"` in [`SFTConfig`].

```python
from trl import SFTConfig

training_args = SFTConfig(..., packing=True, max_length=512)
```

## PEFT for parameter-efficient fine-tuning

Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA are among the most effective techniques for reducing memory usage during training. Instead of training all model parameters, PEFT methods train only a small number of adapter parameters, significantly reducing memory requirements and enabling fine-tuning of larger models on limited hardware.

For comprehensive details on using PEFT with TRL, including various adapter methods, quantization options, and advanced configurations, see [PEFT Integration](peft_integration).

To use PEFT for reducing memory usage:

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")

peft_config = LoraConfig()

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=peft_config,
)
```

PEFT can be combined with other memory reduction techniques such as quantization (4-bit or 8-bit) for even greater memory savings. See [PEFT Integration](peft_integration) for quantization examples.


## Liger for reducing peak memory usage

> [Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduce memory usage by 60%.

For more information, see [Liger Kernel Integration](liger_kernel_integration).

To use Liger for reducing peak memory usage, use the following code snippet:

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
from trl import GKDConfig

training_args = GKDConfig(..., use_liger_kernel=True)
```

</hfoption>
</hfoptions>

## Padding-free

Padding-free batching is an alternative approach for reducing memory usage. In this method, a batch is first sampled and then flattened into a single sequence, avoiding padding. Unlike packing, which can result in incomplete sequences by combining parts of different samples, padding-free batching ensures that all sequences remain complete and intact.

![Padding-free](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/padding-free.png)

> [!WARNING]
> It's highly recommended to use padding-free batching with **FlashAttention 2** or **FlashAttention 3**. Otherwise, you may encounter batch contamination issues.

<hfoptions id="padding-free">
<hfoption id="DPO">

```python
from trl import DPOConfig

training_args = DPOConfig(..., padding_free=True, model_init_kwargs={"attn_implementation": "flash_attention_2"})
```

</hfoption>
<hfoption id="SFT">

```python
from trl import SFTConfig

training_args = SFTConfig(..., padding_free=True, model_init_kwargs={"attn_implementation": "flash_attention_2"})
```

</hfoption>
</hfoptions>

## Activation offloading

Activation offloading is a memory efficiency technique that reduces GPU VRAM usage by temporarily moving activation tensors to CPU RAM during the forward pass and bringing them back only when needed for the backward pass. This significantly reduces peak memory usage at the cost of slightly increased training time.

To enable activation offloading in your SFT training configuration:

```python
from trl import SFTConfig

training_args = SFTConfig(..., activation_offloading=True)
```

Under the hood, activation offloading implements PyTorch's [`saved_tensors_hooks`](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#hooks-for-autograd-saved-tensors) to intercept activations during the forward pass. It intelligently manages which tensors to offload based on size and context, avoiding offloading output tensors that would be inefficient. For performance optimization, it can, via a flag (which is true by default), use CUDA streams to overlap computation with CPU-GPU transfers.

## Padding Sequences to a Multiple

> [!TIP]
> This technique is supported for **SFT** and **Reward** trainers currently.

When enabled, this option ensures that all sequences are **padded to a multiple** of the specified value.  
This can improve computational efficiency on some hardware by aligning sequence lengths to memory-friendly boundaries.

<hfoptions id="pad_to_multiple_of">
<hfoption id="SFT">

```python
from trl import SFTConfig

training_args = SFTConfig(..., pad_to_multiple_of=2048)
```

</hfoption>
<hfoption id="Reward">

```python
from trl import RewardConfig

training_args = RewardConfig(..., pad_to_multiple_of=2048)
```

</hfoption>
</hfoptions>

## Disabling model gathering for generation in online methods

When using DeepSpeed ZeRO-3, model weights are sharded across multiple GPUs. Online methods involve generating completions from the model as part of the training process. During this step, the model weights are temporarily gathered on a single GPU for generation. For very large models, this gathering can lead to OOM errors, as described in this issue: [#2250](https://github.com/huggingface/trl/issues/2250#issue-2598304204).

If you encounter this issue, you can disable the gathering of model weights for generation by setting the following parameter:

<hfoptions id="ds3_gather_for_generation">
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
<hfoption id="Online DPO">

```python
from trl import OnlineDPOConfig

training_args = OnlineDPOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
<hfoption id="PPO">

```python
from trl import PPOConfig

training_args = PPOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(..., ds3_gather_for_generation=False)
```

</hfoption>
</hfoptions>

This adjustment prevents model weights from being gathered, avoiding OOM errors, but it may result in slower generation speeds.

## vLLM sleep mode

When using **vLLM** as the generation backend for online training methods, you can enable _sleep mode_ to offload vLLM parameters and cache to CPU RAM during the optimization step and reload them back to GPU VRAM when needed for weight synchronization and generation.

<hfoptions id="vllm_sleep">
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., vllm_enable_sleep_mode=True)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(..., vllm_enable_sleep_mode=True)
```

</hfoption>
</hfoptions>
