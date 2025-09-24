# Distributing Training

<Tip warning={true}>
Section under construction. Feel free to contribute!
</Tip>

## Multi-GPU Training with TRL

The trainers in TRL use [🤗 Accelerate](https://github.com/huggingface/accelerate) to enable distributed training across multiple GPUs or nodes. To do so, first create an [🤗 Accelerate](https://github.com/huggingface/accelerate) config file by running

```bash
accelerate config
```

and answering the questions according to your multi-GPU / multi-node setup. You can then launch distributed training by running:

```bash
accelerate launch train.py
```

We also provide config files in the [examples folder](https://github.com/huggingface/trl/tree/main/examples/accelerate_configs) that can be used as templates. To use these templates, simply pass the path to the config file when launching a job, e.g.:

```shell
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml train.py <SCRIPT_ARGS>
```

This automatically distributes the workload across all available GPUs.

Under the hood, [🤗 Accelerate](https://github.com/huggingface/accelerate) creates one model per GPU. Each process:
- Processes its own batch of data
- Computes the loss and gradients for that batch
- Shares gradient updates across all GPUs

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/multi_gpu.png)

The effective batch size is calculated as:

$$
\text{Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{num\_devices} \times \text{gradient\_accumulation\_steps}
$$

To maintain a consistent batch size when scaling to multiple GPUs, make sure to update `per_device_train_batch_size` and `gradient_accumulation_steps` accordingly.

Example, these configurations are equivalent, and should yield the same results:

| Number of GPUs | Per device batch size | Gradient accumulation steps | Comments |
| --- | --- | --- | --- |
| 1 | 32 | 1 | Possibly high memory usage, but faster training |
| 1 | 4 | 8 | Lower memory usage, slower training |
| 8 | 4 | 1 | Multi-GPU to get the best of both worlds |

<Tip> 

Having one model per GPU can lead to high memory usage, which may not be feasible for large models or low-memory GPUs. In such cases, you can leverage [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), which provides optimizations like model sharding, Zero Redundancy Optimizer, mixed precision training, and offloading to CPU or NVMe. Check out our [DeepSpeed Integration](deepspeed_integration) guide for more details.

</Tip>

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

Use one of the provided accelerate config files (e.g. [`context_parallel_2gpu.yaml`](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/context_parallel_2gpu.yaml) for 2 GPUs):

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: true  # Enable activation checkpointing for memory efficiency
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
    gradient_checkpointing=False,   # The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously
    per_device_train_batch_size=1,
    ...
)
```

Then, launch your training script with the appropriate accelerate config file:

```bash
accelerate launch --config_file context_parallel_2gpu.yaml train.py
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

### Benchmarking Context Parallelism

We benchmarked CP to highlight its potential improvements in training efficiency.  
Our experiments were conducted using **1, 2, 4, and 8 H100 GPUs**, though the results can be extended to larger clusters with more nodes and GPUs.

For the setup, we fine-tuned an **8B model** ([Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)) using the provided accelerate configuration  
([`context_parallel_2gpu.yaml`](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/context_parallel_2gpu.yaml)).  
We adjusted `num_processes` and `parallelism_config_cp_size` based on the number of GPUs for each run.  
Training was performed with the [sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) example script, combined with the parameters described above.

The results below summarize the **maximum trainable sequence length** and **iterations per second** for different numbers of GPUs. A value marked as `OOM` indicates that the configuration ran out of memory and could not be trained.  

These results show that **Context Parallelism (CP) scales effectively with more GPUs**, enabling training on much longer sequences. With **8 GPUs**, context lengths of over **300k tokens** become feasible, unlocking training with extremely long contexts while maintaining reasonable throughput.  

<div class="flex justify-center">
  <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/context_parallelism_max_length_plot.png" alt="CP Max content length" width="45%"/>
  <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/context_parallelism_s_it_plot.png" alt="CP seconds/iteration" width="45%"/>
</div>

<Tip>

Accelerate also supports **N-Dimensional Parallelism (ND-parallelism)**, which enables you to combine different parallelization strategies to efficiently distribute model training across multiple GPUs.  

You can learn more and explore configuration examples in the [Accelerate ND-parallelism guide](https://github.com/huggingface/accelerate/blob/main/examples/torch_native_parallelism/README.md#nd-parallelism).

</Tip>


**Further Reading on Context Parallelism**  

- [Accelerate: Context Parallelism Guide](https://github.com/huggingface/accelerate/blob/main/docs/source/concept_guides/context_parallelism.md)  
- [Accelerate Example: 128k Sequence Length](https://github.com/huggingface/accelerate/blob/main/examples/torch_native_parallelism/README.md#context-parallelism-128k-sequence-length)  
- [Hugging Face Blog: Enabling Long-Context Training with Sequence Parallelism in Axolotl](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl)  
- [Snowflake Engineering Blog: Arctic Long Sequence Training (ALST) — Scalable and Efficient Training for Multi-Million Token Sequences (Note that they use a different strategy)](https://www.snowflake.com/en/engineering-blog/arctic-long-sequence-training-multi-million-token-ai/)

## Multi-Node Training

We're working on a guide for multi-node training. Stay tuned! 🚀