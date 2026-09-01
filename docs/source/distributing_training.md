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
> Training on very long sequences has its own guide: [Long Context Training](long_context_training).

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



