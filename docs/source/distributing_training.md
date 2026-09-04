# Distributing Training

> [!WARNING]
> Section under construction. Feel free to contribute!

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

![multi gpu](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/multi_gpu.png)

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

> [!TIP]
> Having one model per GPU can lead to high memory usage, which may not be feasible for large models or low-memory GPUs. In such cases, you can leverage [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), which provides optimizations like model sharding, Zero Redundancy Optimizer, mixed precision training, and offloading to CPU or NVMe. Check out our [DeepSpeed Integration](deepspeed_integration) guide for more details.

> [!TIP]
> Training on very long sequences has its own guide: [Training Beyond 1M Tokens](long_context_training).

## Training MoE Models at the 100B–753B Scale

Mixture-of-experts models put most of their parameters in the expert weights, and expert parallelism shards exactly those: `DistributedConfig(tp_size=E, fsdp_size=D, enable_expert_parallel=True)` at `from_pretrained` time places the experts across `E` GPUs and shards everything else — dense trunk, and optimizer state for both — across `D`. The model loads directly onto this 2-D mesh, so no rank ever materializes it whole, and [`SFTTrainer`] trains it like any other model:

```python
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-5.2",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=32, fsdp_size=2, enable_expert_parallel=True),
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)
```

This needs the 2-D mesh support from [transformers#48516](https://github.com/huggingface/transformers/pull/48516) (stacked on [transformers#48205](https://github.com/huggingface/transformers/pull/48205)) until it is released; accelerate and peft from PyPI are enough.

Verified configurations (H100 nodes, sequence length 2048, bf16, per-device batch 1, the default `loss_type="chunked_nll"`, gradient checkpointing). The two 8-node rows are runnable examples: [`examples/sft_glm_5_2_expert_parallel/`](https://github.com/huggingface/trl/tree/main/examples/sft_glm_5_2_expert_parallel) (753B, LoRA) and [`examples/sft_glm_4_5_air_full_finetune/`](https://github.com/huggingface/trl/tree/main/examples/sft_glm_4_5_air_full_finetune) (110B, full fine-tuning):

| Model | Training | GPUs | Config | Step time | Peak GPU memory |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | full FT | 8 (1 node) | `ep=8` | 0.9 s | 34 GB |
| GLM-4.5-Air (110B) | full FT | 64 (8 nodes) | `ep=32 × fsdp=2` | 3.0 s | 41 GB |
| GLM-4.6 (357B) | LoRA | 16 (2 nodes) | `ep=16` | 4.4 s | 73 GB |
| GLM-5.2 (753B) | LoRA | 64 (8 nodes) | `ep=32 × fsdp=2` | 3.1 s | 56 GB |

These are real training runs, not just "it fits": GLM-5.2 goes from loss 3.3 to 2.3 in 50 steps on tulu-3 chat data, GLM-4.5-Air full fine-tuning from 3.9 to 1.2 in 20 (re-run on the example as-is with transformers#48516 and accelerate/peft/trl from main).

Read the step times together with the batch they process: the expert-parallel group works on a *shared* batch (only the `fsdp` dimension is data-parallel), so a step in the `ep=32 × fsdp=2` rows is 2 sequences — about 4,096 tokens — not 64. Step time tells you the update cadence; multiply by the `fsdp` size, not the GPU count, to get throughput.

What to expect operationally at this scale:

1. **Loading is the slow part, and it is filesystem-bound.** A cold multi-node load reads the checkpoint at well under 1 GiB/s per node (every node reads the full checkpoint, and the loader's access pattern defeats readahead) — 13 to 31 minutes for the models above. The same load from a warm page cache runs at over 10 GiB/s. If your nodes have the RAM, warming the page cache first with large sequential reads recovers most of that gap: with [transformers#48227](https://github.com/huggingface/transformers/pull/48227), set `HF_SHARD_PREFETCH=4` and the loader does it before reading. Measured on GLM-4.6 (665 GiB, 8 nodes, cold): 17–52 minutes → 465 s prefetch + 63 s load. The floor is the filesystem's aggregate bandwidth shared across nodes (~10 GiB/s on our Lustre): `checkpoint_bytes × nodes / aggregate`.
2. **Full-model saving gathers to one rank.** `trainer.save_model()` writes a standard HF checkpoint, at roughly 0.2–0.4 GiB/s (about 15 minutes for the 206 GiB Air checkpoint). LoRA adapter saves are seconds regardless of model size.
3. **Memory scales the way the mesh says it should.** Expert parameters, gradients, and optimizer state divide by `ep × fsdp`; full fine-tuning of the 110B model peaks at 41 GB per GPU across 64 GPUs.

### Making it fast

Every lever below is measured (Qwen3-30B-A3B and GLM-4.5-Air, 8–16×H100); together they sustain ~48× the naive-defaults throughput at 30B, with healthy convergence:

- **`packing=True`** — chat samples are short and padding dominates otherwise: ~4.5× effective tokens/s.
- **`gradient_accumulation_steps=4`** — amortizes per-optimizer-step communication: +33%.
- **Keep `gradient_checkpointing=True`** — disabling it is *slower* per token here and halves the usable batch.
- **Raise `per_device_train_batch_size` to your memory budget** — sequence length doesn't matter once packed (per-token cost is flat); batch does.
- **`dataset_num_proc=16`** — tokenization otherwise silently costs many single-threaded minutes at scale.
- **Measure with ≥30 steps** — short runs understate steady-state throughput by 10–25%.
- Loading: `HF_SHARD_PREFETCH=4` (see above). Saving: sharded DCP with `dcp.async_save` blocks training seconds, not minutes; consolidate to HF format offline.

Scaling across nodes costs ~15% per-GPU throughput at 2 nodes (85% scaling efficiency, GLM-4.5-Air, ep=16). MFU rises with active-parameter count: ~5% at 3.35B-active, ~9.5% at 13.5B-active on this stack.

## Multi-Node Training

When a single machine doesn't have enough GPUs, TRL can scale training across multiple machines (nodes) using [🤗 Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#multi-node-training).

### Accelerate Configuration
Create an `accelerate` config file (e.g., `multi_node.yaml`) for multi-node training. Key fields:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 2
machine_rank: 0  # 0 for main node, 1 for second node
main_process_ip: 10.0.0.1  # IP of rank 0 node
main_process_port: 29500
num_processes: 16  # total processes across nodes
mixed_precision: bf16
use_cpu: false
same_network: true
```

Adjust `num_processes` to match the total number of GPUs across all nodes.

> [!NOTE]
> Replace `10.0.0.1` with the actual IP address of the rank 0 (main) node.

### Launching

#### Option 1: Manual Launch (Non-HPC)

Run the following on each node manually:
```bash
# Node 0 (main node)
accelerate launch --config_file multi_node.yaml --machine_rank 0 train.py

# Node 1
accelerate launch --config_file multi_node.yaml --machine_rank 1 train.py
```
#### Option 2: SLURM Launch (HPC Clusters)

For clusters using SLURM job scheduler, create a job script (e.g., `slurm_job.sh`):
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --job-name=trl_multi

srun accelerate launch --config_file multi_node.yaml train.py
```

Then submit the job:
```bash
sbatch slurm_job.sh
```

SLURM automatically distributes the training across all requested nodes and GPUs, and `srun` configures the necessary environment variables for multi-node communication.

**Key SLURM directives:**
- `--nodes=2`: Request 2 compute nodes
- `--gpus-per-node=8`: Allocate 8 GPUs per node (16 total)
- `--job-name`: Label for tracking in the job queue

You can combine multi-node with DeepSpeed by setting `distributed_type: DEEPSPEED` and adding a `deepspeed_config` block. See the [DeepSpeed integration guide](https://huggingface.co/docs/trl/en/deepspeed_integration).

### Further Reading

- [Accelerate: Launching Scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [Accelerate: Example Zoo](https://huggingface.co/docs/accelerate/usage_guides/training_zoo)
- [SLURM Workload Manager Documentation](https://slurm.schedmd.com/) - For cluster job scheduling



