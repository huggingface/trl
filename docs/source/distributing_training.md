# Distributing Training

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

## Training MoE Models at the 100B–753B Scale

Mixture-of-experts models put most of their parameters in the expert weights, and expert parallelism shards exactly those. `DistributedConfig(fsdp_size=D, ep_size=E)` at `from_pretrained` time splits the experts across `E` GPUs with token dispatch (each rank keeps its own tokens and only the routed token/expert pairs travel, an all-to-all each way) and shards everything else, dense trunk and optimizer state, across all `D` GPUs. The model loads directly onto this mesh, so no rank ever materializes it whole, and [`SFTTrainer`] trains it like any other model:

```python
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "zai-org/GLM-5.2",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(fsdp_size=64, ep_size=8),
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset)
```

The model's own expert plan selects token dispatch, so the two sizes are the whole configuration. Expert parallelism landed on transformers `main` on 2026-10-06 ([transformers#48857](https://github.com/huggingface/transformers/pull/48857) to [transformers#49160](https://github.com/huggingface/transformers/pull/49160)) and is not in a release yet; accelerate and peft from PyPI are enough.

The two 8-node configurations are runnable examples: [`examples/sft_glm_5_2_expert_parallel/`](https://github.com/huggingface/trl/tree/main/examples/sft_glm_5_2_expert_parallel) (753B, LoRA) and [`examples/sft_glm_4_5_air_full_finetune/`](https://github.com/huggingface/trl/tree/main/examples/sft_glm_4_5_air_full_finetune) (110B, full fine-tuning).

Measured configurations (H100 nodes, sequence length 2048, bf16, per-device batch 1, the default `loss_type="chunked_nll"`, gradient checkpointing). These numbers were measured with the earlier *masked* implementation (`enable_expert_parallel=True`, where the expert-parallel group shares one batch and only the `fsdp` dimension is data-parallel), so a step in the `ep=32 × fsdp=2` rows processes 2 sequences, about 4,096 tokens, not 64. With token dispatch every rank trains on its own sequences, so the same step processes 64; the step times and memory below are a lower bound on what the examples above do per step, not a measurement of them:

| Model | Training | GPUs | Config | Step time | Peak GPU memory |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | full FT | 8 (1 node) | `ep=8` | 0.9 s | 34 GB |
| GLM-4.5-Air (110B) | full FT | 64 (8 nodes) | `ep=32 × fsdp=2` | 3.0 s | 41 GB |
| GLM-4.6 (357B) | LoRA | 16 (2 nodes) | `ep=16` | 4.4 s | 73 GB |
| GLM-5.2 (753B) | LoRA | 64 (8 nodes) | `ep=32 × fsdp=2` | 3.1 s | 56 GB |

These are real training runs, not just "it fits": GLM-5.2 goes from loss 3.3 to 2.3 in 50 steps on tulu-3 chat data, GLM-4.5-Air full fine-tuning from 3.9 to 1.2 in 20.

Token dispatch, measured on GLM-4.5-Air (110B) with LoRA r16 on 4 H100s (`fsdp_size=4, ep_size=4`, packed sequences, 20 steps, transformers main of 2026-10-06):

| per-device batch | grad. accum. | max_length | step time | tokens/s (4 GPUs) | peak GPU memory |
|---|---|---|---|---|---|
| 1 | 1 | 2048 | 1.9 s | 4.2k | 54 GB |
| 2 | 1 | 2048 | 3.1 s | 5.3k | 56 GB |
| 2 | 4 | 2048 | 11.5 s | 5.7k | 57 GB |
| 1 | 1 | 4096 | 3.0 s | 5.5k | 56 GB |
| 4 | 1 | 2048 | 6.9 s | 4.7k | 61 GB |

On the same 4 GPUs and per-rank batch, the earlier masked implementation moves 2.9k tokens/s: every rank ran the whole group's batch, so its step time was spent on 4x fewer distinct tokens. Splitting the experts across all the GPUs (`ep_size` equal to the GPU count) was also the fastest and the leanest choice here; `ep_size=2` and `ep_size=1` cost 20% more per step and up to 10 GB more.

What to expect operationally at this scale:

1. **Loading is the slow part, and it is filesystem-bound.** A cold multi-node load reads the checkpoint at well under 1 GiB/s per node (every node reads the full checkpoint, and the loader's access pattern defeats readahead), 13 to 31 minutes for the models above. The same load from a warm page cache runs at over 10 GiB/s. If your nodes have the RAM, warming the page cache first with large sequential reads recovers most of that gap: with [transformers#48227](https://github.com/huggingface/transformers/pull/48227), set `HF_SHARD_PREFETCH=4` and the loader does it before reading. Measured on GLM-4.6 (665 GiB, 8 nodes, cold): 17–52 minutes → 465 s prefetch + 63 s load. The floor is the filesystem's aggregate bandwidth shared across nodes (~10 GiB/s on our Lustre): `checkpoint_bytes × nodes / aggregate`.
2. **Full-model saving gathers to one rank.** `trainer.save_model()` writes a standard HF checkpoint, at roughly 0.2–0.4 GiB/s (about 15 minutes for the 206 GiB Air checkpoint). LoRA adapter saves are seconds regardless of model size.
3. **Memory scales the way the mesh says it should.** Dense parameters, gradients and optimizer state divide by `fsdp_size`; expert ones divide by `ep_size` and again by the number of ranks that hold the same experts (`fsdp_size / ep_size`), so by the whole GPU count. Full fine-tuning of the 110B model peaked at 41 GB per GPU across 64 GPUs with the masked implementation.

### Making it fast

Every lever below is measured (Qwen3-30B-A3B and GLM-4.5-Air, 8–16×H100, masked implementation); together they sustain ~48× the naive-defaults throughput at 30B, with healthy convergence:

- **`packing=True`** — chat samples are short and padding dominates otherwise: ~4.5× effective tokens/s.
- **`gradient_accumulation_steps=4`** — amortizes per-optimizer-step communication: +33%.
- **Keep `gradient_checkpointing=True`** — disabling it is *slower* per token here and halves the usable batch.
- **Raise `per_device_train_batch_size` to your memory budget** — sequence length doesn't matter once packed (per-token cost is flat); batch does.
- **`dataset_num_proc=16`** — tokenization otherwise silently costs many single-threaded minutes at scale.
- **Measure with ≥30 steps** — short runs understate steady-state throughput by 10–25%.
- Loading: `HF_SHARD_PREFETCH=4` (see above). Saving: sharded DCP with `dcp.async_save` blocks training seconds, not minutes; consolidate to HF format offline.

Scaling across nodes costs ~15% per-GPU throughput at 2 nodes (85% scaling efficiency, GLM-4.5-Air, ep=16). MFU rises with active-parameter count: ~5% at 3.35B-active, ~9.5% at 13.5B-active on this stack.

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



