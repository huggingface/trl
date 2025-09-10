# Reducing Memory Usage

<Tip warning={true}>

Section under construction. Feel free to contribute!

</Tip>

## Truncation

Sequence lengths in the dataset can vary widely. When data is batched, sequences are padded to match the longest one in the batch, which can cause high memory usage, even if most sequences are relatively short.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png" alt="Truncation prompt-completion" width="600"/>
</div>

To reduce memory usage, it's important to truncate sequences to a reasonable length. While TRL trainers truncate sequences by default, you may want to adjust the default truncation length to better align with your specific use case.

<hfoptions id="truncation">
<hfoption id="DPO">

DPO truncation is applied first to the prompt and to the completion via the `max_prompt_length` and `max_completion_length` parameters. The `max_length` parameter is then used to truncate the resulting sequence.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_prompt_completion.png" alt="Truncation prompt-completion" width="600"/>
</div>

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

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_input_ids.png" alt="Truncation input ids" width="600"/>
</div>

To set the truncation parameter, use the following code snippet:

```python
from trl import SFTConfig

training_args = SFTConfig(..., max_length=...)
```

</hfoption>
</hfoptions>

### How to choose the `max_length` value?

If `max_length` is too small, a significant portion of your tokens will be discarded and won't contribute to training. If it's too large, memory usage can spike, potentially leading to OOM (Out-Of-Memory) errors. Without packing or padding-free, a large `max_length` may also result in inefficient training, as many tokens will be padding.

To help you choose an appropriate value, we provide a utility to visualize the sequence length distribution in your dataset.

<iframe src="https://trl-lib-dataset-length-profiler.hf.space" frameborder="0" width="100%" height="1000"></iframe>

## Packing

<Tip>

This technique applies only to SFT.

</Tip>


[Truncation](#truncation) has several drawbacks:
1. **Loss of information**: Key data at the end of a sequence may be discarded.
2. **Choosing truncation length**: Too short loses data; too long undermines efficiency.

Packing, introduced in [Raffel et al., 2020](https://huggingface.co/papers/1910.10683), addresses these issues by grouping sequences instead of truncating. It concatenates and splits dataset sequences into the desired lengths.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing_2.png" alt="Packing" width="600"/>
</div>

Packing reduces padding by merging several sequences in one row when possible. We use an advanced method to be near-optimal in the way we pack the dataset. To enable packing, use `packing=True` in the [`SFTConfig`].

<Tip>

In TRL 0.18 and earlier, packing used a more aggressive method that reduced padding to almost nothing, but had the downside of breaking sequence continuity for a large fraction of the dataset. To revert to this strategy, use `packing_strategy="wrapped"` in `SFTConfig`.

</Tip>

```python
from trl import SFTConfig

training_args = SFTConfig(..., packing=True, max_length=512)
```

<Tip warning={true}>

Packing may cause batch contamination, where adjacent sequences influence one another. This can be problematic for some applications. For more details, see [#1230](https://github.com/huggingface/trl/issues/1230).

</Tip>

## Liger for reducing peak memory usage

> [Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels designed specifically for LLM training. It can effectively increase multi-GPU training throughput by 20% and reduces memory usage by 60%.

For more information, see [Liger Kernel Integration](liger_kernel_integration)

<hfoptions id="liger">
<hfoption id="DPO">

To use Liger for reducing peak memory usage, use the following code snippet:
  
```python
from trl import DPOConfig

training_args = DPOConfig(..., use_liger_loss=True)
```

</hfoption>
<hfoption id="GRPO">

To use Liger for reducing peak memory usage, use the following code snippet:
  
```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_liger_loss=True)
```

</hfoption>
<hfoption id="KTO">

To use Liger for reducing peak memory usage, use the following code snippet:
  
```python
from trl import KTOConfig

training_args = KTOConfig(..., use_liger_loss=True)
```

</hfoption>
</hfoptions>

## Padding-free

Padding-free batching is an alternative approach for reducing memory usage. In this method, a batch is first sampled and then flattened into a single sequence, avoiding padding. Unlike packing, which can result in incomplete sequences by combining parts of different samples, padding-free batching ensures that all sequences remain complete and intact.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/padding-free.png" alt="Padding-free batching" width="600"/>
</div>

<Tip warning={true}>

It's highly recommended to use padding-free batching with **FlashAttention 2** or **FlashAttention 3**. Otherwise, you may encounter batch contamination issues.

</Tip>

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

<Tip warning={true}>

When using activation offloading with models that use Liger kernels, you must disable Liger cross entropy due to compatibility issues. The issue occurs specifically with `use_liger_kernel=True` because Liger cross entropy performs in-place operations which conflict with activation offloading. The default setting (`use_liger_kernel=False`) works:

```python
# When using activation offloading with a model that uses Liger kernels:
from trl import SFTConfig

training_args = SFTConfig(
    activation_offloading=True,
    use_liger_kernel=False,  # Disable Liger cross entropy
    # Other parameters...
)
```
</Tip>

Under the hood, activation offloading implements PyTorch's [`saved_tensors_hooks`](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#hooks-for-autograd-saved-tensors) to intercept activations during the forward pass. It intelligently manages which tensors to offload based on size and context, avoiding offloading output tensors which would be inefficient. For performance optimization, it can optionally use CUDA streams to overlap computation with CPU-GPU transfers.

## Disabling model gathering for generation in online methods

When using DeepSpeed ZeRO-3, model weights are sharded across multiple GPUs. Online methods involve generating completions from the model as part of the training process. During this step, the model weights are temporarily gathered on a single GPU for generation. For very large models, this gathering can lead to out-of-memory (OOM) errors, as described in this issue: [#2250](https://github.com/huggingface/trl/issues/2250#issue-2598304204).

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

## Context Parallelism

Context Parallelism (CP) is a parallelization technique that enables training with longer sequences by splitting the sequence dimension across multiple GPUs. Each GPU processes a portion of the sequence, allowing you to train with sequences longer than what would fit on a single GPU's memory.

For more details on CP, see the [Ultrascale Playbook - Context Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism).

CP is particularly useful when:

- You want to train with very long sequences (>32k tokens)
- Single GPU memory is insufficient for your desired sequence length
- You need to maintain sequence coherence across the full context

### Requirements and Limitations

CP has specific requirements:

1. **Accelerate 1.10 or higher** is required
2. **FSDP2 (PyTorch FSDP v2)** is required as the distributed training backend
3. **SDPA attention** - Flash Attention is currently not supported with CP
4. **Sequence length divisibility** - sequences must be divisible by `cp_size * 2`. This is now automatically handled using the `pad_to_multiple_of` parameter in the data collator, which works seamlessly with both standard and padding-free modes.

### Configuration

To enable CP, you need to configure both Accelerate and your training arguments:

#### Accelerate Configuration

Use one of the provided accelerate config files (e.g. `fsdp_context_parallel_2gpu.yaml` for 2 GPUs):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2  # Number of GPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 1
  parallelism_config_tp_size: 1
  parallelism_config_cp_size: 2  # Context parallel size
```

#### Training Configuration

```python
from trl import SFTConfig

training_args = SFTConfig(
    # required
    pad_to_multiple_of=4,           # ensures divisibility by cp_size * 2
    # to get the most out of CP
    max_length=16384,               # long sequence length
    packing=True,                   # use packing to reduce padding
    use_liger_kernel=True,          # compatible with CP
    per_device_train_batch_size=1,
    ...
)
```

Then, launch your training script with the appropriate accelerate config file:

```bash
accelerate launch --config_file fsdp_context_parallel_2gpu.yaml train.py
```

### Best Practices

1. **Use the `pad_to_multiple_of` parameter** - This is now the recommended way to ensure sequence length divisibility:
   - For `cp_size=2`: use `pad_to_multiple_of=4` (since `cp_size * 2 = 4`)
   - For `cp_size=4`: use `pad_to_multiple_of=8` (since `cp_size * 2 = 8`)
   - The data collator automatically pads sequences to the required multiple, ensuring compatibility with CP

2. **Use packing with padding** - The default BFD (Best Fit Decreasing) strategy works perfectly:
   - Preserves sequence boundaries and maintains training quality
   - Works seamlessly with both `padding_free=True` and standard padding modes

3. **Combine with other memory optimizations** like Liger kernels, bfloat16, and gradient checkpointing

4. **Start with smaller context parallel sizes** (2-4 GPUs) before scaling up

5. **Monitor memory usage** across all GPUs to ensure balanced workload

## vLLM sleep mode

When using vLLM as the generation backend, you can enable _sleep mode_ to offload vLLM parameters and cache to CPU RAM during the optimization step and reload them back to GPU VRAM when needed for weight synchronization and generation.

<hfoptions id="vllm_sleep">
<hfoption id="GRPO">

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., vllm_sleep_enabled=True)
```

</hfoption>
<hfoption id="RLOO">

```python
from trl import RLOOConfig

training_args = RLOOConfig(..., vllm_sleep_enabled=True)
```

</hfoption>
</hfoptions>
