# Distributing Training

> [!WARNING]
> Section under construction. Feel free to contribute!

## Multi-GPU Training with TRL

The trainers in TRL are launched with [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html), PyTorch's distributed launcher. It starts one process per GPU:

```bash
torchrun --nproc_per_node 8 train.py <SCRIPT_ARGS>
```

The `trl` CLI does the same under the hood, so `trl sft ...` runs on every GPU of the machine; pass `--nproc_per_node` to use fewer.

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

## Multi-Node Training

When a single machine doesn't have enough GPUs, torchrun can scale training across multiple machines (nodes). Every node runs the same command with the total number of nodes, its own rank, and a rendezvous endpoint on the main node.

### Option 1: Manual Launch (Non-HPC)

Run the following on each node manually:

```bash
# Node 0 (main node, IP 10.0.0.1)
torchrun --nnodes 2 --nproc_per_node 8 --node_rank 0 --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:29500 train.py

# Node 1
torchrun --nnodes 2 --nproc_per_node 8 --node_rank 1 --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:29500 train.py
```

> [!NOTE]
> Replace `10.0.0.1` with the actual IP address of the rank 0 (main) node.

### Option 2: SLURM Launch (HPC Clusters)

For clusters using SLURM job scheduler, create a job script (e.g., `slurm_job.sh`):

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --job-name=trl_multi

MAIN_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
srun torchrun --nnodes 2 --nproc_per_node 8 --node_rank $SLURM_NODEID --rdzv_backend c10d --rdzv_endpoint $MAIN_NODE:29500 train.py
```

Then submit the job:

```bash
sbatch slurm_job.sh
```

`srun` starts one `torchrun` per node, and each of them starts one process per GPU.

**Key SLURM directives:**
- `--nodes=2`: Request 2 compute nodes
- `--gpus-per-node=8`: Allocate 8 GPUs per node (16 total)
- `--job-name`: Label for tracking in the job queue

You can combine multi-node with DeepSpeed by setting `deepspeed` in the training config. See the [DeepSpeed integration guide](deepspeed_integration).

### Further Reading

- [torchrun (Elastic Launch)](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [SLURM Workload Manager Documentation](https://slurm.schedmd.com/) - For cluster job scheduling



