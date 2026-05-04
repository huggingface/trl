# ruff: noqa
"""Test per-layer compile + FSDP2 on a single node (8 GPUs).

Isolates whether the slowdown is from FSDP comm overlap or cross-node latency.
Run: srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:8 --ntasks-per-node=1 --exclusive --time=00:20:00 --qos=normal \
  bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && \
  torchrun --nproc_per_node=8 benchmark/test_compile_fsdp_1node.py'
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World size: {world_size}, Device: {device}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    seq_len = 16384
    batch_size = 1
    n_warmup = 3
    n_steps = 10

    def apply_fsdp(m):
        mp_policy = MixedPrecisionPolicy()
        for layer in m.model.layers:
            fully_shard(layer, mp_policy=mp_policy)
        fully_shard(m, mp_policy=mp_policy)

    def run_steps(m, label, n_warmup, n_steps):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        # Warmup
        for i in range(n_warmup):
            out = m(input_ids=input_ids, labels=labels)
            out.loss.backward()
            for p in m.parameters():
                p.grad = None
        torch.cuda.synchronize()
        dist.barrier()

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n_steps):
            out = m(input_ids=input_ids, labels=labels)
            out.loss.backward()
            for p in m.parameters():
                p.grad = None
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms_per_step = (elapsed / n_steps) * 1000

        if rank == 0:
            tokens_per_sec = (batch_size * seq_len * world_size * n_steps) / elapsed
            print(
                f"{label}: {ms_per_step:.0f} ms/step, TPS={tokens_per_sec:.0f}, TPS/GPU={tokens_per_sec / world_size:.0f}"
            )
        return ms_per_step

    # --- Test 1: EAGER + FSDP ---
    if rank == 0:
        print("\n=== EAGER + FSDP2 ===")
    apply_fsdp(model)
    eager_ms = run_steps(model, "Eager+FSDP2", n_warmup, n_steps)

    # Clean up
    del model
    torch.cuda.empty_cache()

    # --- Test 2: PER-LAYER COMPILE + FSDP ---
    if rank == 0:
        print("\n=== PER-LAYER COMPILE + FSDP2 ===")
    model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model2.train()
    model2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Per-layer compile BEFORE FSDP
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    for layer in model2.model.layers:
        layer.compile(backend="inductor", fullgraph=True)

    apply_fsdp(model2)
    compiled_ms = run_steps(model2, "Compiled+FSDP2", n_warmup, n_steps)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY (1 node, {world_size} GPUs, FSDP2, seq_len={seq_len}):")
        print(f"  Eager+FSDP2:    {eager_ms:.0f} ms/step")
        print(f"  Compiled+FSDP2: {compiled_ms:.0f} ms/step")
        print(f"  Ratio: {compiled_ms / eager_ms:.2f}x")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
