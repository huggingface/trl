# ruff: noqa
"""Test per-layer compile + FSDP2: 1 node vs 2 nodes, same script.

Also tests with reshard_after_forward=False to isolate prefetching overlap.

Usage:
  # 1 node:
  srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:8 --ntasks-per-node=1 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    torchrun --nproc_per_node=8 benchmark/test_compile_fsdp_multinode.py'

  # 2 nodes:
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_fsdp_multinode.py'
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM


SEQ_LEN = 16384
BATCH_SIZE = 1
N_WARMUP = 2
N_STEPS = 10


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_model():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def apply_fsdp(model, reshard_after_forward=True):
    mp_policy = MixedPrecisionPolicy()
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward)


def apply_per_layer_compile(model):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)


def benchmark(model, label, rank, world_size):
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = input_ids.clone()

    # Warmup
    for _ in range(N_WARMUP):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_step = (elapsed / N_STEPS) * 1000
    tokens_per_sec = (BATCH_SIZE * SEQ_LEN * world_size * N_STEPS) / elapsed

    if rank == 0:
        print(
            f"  {label}: {ms_per_step:.0f} ms/step, TPS={tokens_per_sec:.0f}, TPS/GPU={tokens_per_sec / world_size:.0f}"
        )
    return ms_per_step


def run_config(label, compile_layers, reshard_after_forward, rank, world_size):
    """Load model, optionally compile, apply FSDP, benchmark, cleanup."""
    if rank == 0:
        print(f"\n--- {label} ---")

    model = load_model()

    if compile_layers:
        apply_per_layer_compile(model)

    apply_fsdp(model, reshard_after_forward=reshard_after_forward)
    ms = benchmark(model, label, rank, world_size)

    del model
    torch.cuda.empty_cache()
    dist.barrier()
    return ms


def main():
    rank, world_size, local_rank = setup()
    n_nodes = world_size // 8

    if rank == 0:
        print(f"Nodes: {n_nodes}, GPUs: {world_size}, seq_len={SEQ_LEN}")
        print("=" * 70)

    results = {}

    # Test 1: Eager + FSDP2 (baseline)
    results["eager"] = run_config(
        "Eager + FSDP2", compile_layers=False, reshard_after_forward=True, rank=rank, world_size=world_size
    )

    # Test 2: Compiled + FSDP2
    results["compiled"] = run_config(
        "Compiled + FSDP2", compile_layers=True, reshard_after_forward=True, rank=rank, world_size=world_size
    )

    # Test 3: Compiled + FSDP2, no reshard (no prefetch overhead)
    results["compiled_no_reshard"] = run_config(
        "Compiled + FSDP2 (no reshard)",
        compile_layers=True,
        reshard_after_forward=False,
        rank=rank,
        world_size=world_size,
    )

    # Test 4: Eager + FSDP2, no reshard (to compare)
    results["eager_no_reshard"] = run_config(
        "Eager + FSDP2 (no reshard)",
        compile_layers=False,
        reshard_after_forward=False,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"SUMMARY ({n_nodes} node(s), {world_size} GPUs, FSDP2, seq_len={SEQ_LEN}):")
        print(f"  Eager + FSDP2:              {results['eager']:.0f} ms/step")
        print(
            f"  Compiled + FSDP2:           {results['compiled']:.0f} ms/step  ({results['compiled'] / results['eager']:.2f}x)"
        )
        print(
            f"  Eager + no reshard:         {results['eager_no_reshard']:.0f} ms/step  ({results['eager_no_reshard'] / results['eager']:.2f}x)"
        )
        print(
            f"  Compiled + no reshard:      {results['compiled_no_reshard']:.0f} ms/step  ({results['compiled_no_reshard'] / results['eager']:.2f}x)"
        )
        print(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
