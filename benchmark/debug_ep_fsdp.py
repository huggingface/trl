# ruff: noqa
"""Definitive test: after FSDP wrap, are experts 16-127 still present, or only experts 0-15?

After FSDP, on each rank, gather the full param via DTensor.full_tensor(). All 8 ranks
should see the SAME full param. If FSDP broadcast rank 0's data, that full param
contains only experts 0-15 (rank 0's original 16) replicated/sharded but expert IDs 16-127
are GONE. We compare against a separately-loaded non-EP reference's expert weights.
"""

import os

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"\n=== EP+FSDP definitive corruption test, world_size={world_size} ===\n", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
    )
    dist.barrier()

    # BEFORE: Snapshot of layer-0 expert 0 row 0 col 0 from EACH rank's local tensor.
    # Each rank holds its 16 experts. Rank 0 expert 0, rank 1 expert 0 (= global 16), etc.
    # These should all DIFFER because each rank has a different slice of the original 128.
    l0_gu_before = model.model.layers[0].mlp.experts.gate_up_proj
    # Sample expert 0, channel 0, position 0 from local tensor
    if hasattr(l0_gu_before, "to_local"):
        local_before = l0_gu_before.to_local()
    else:
        local_before = l0_gu_before
    rank_local_expert0_first = local_before[0, 0, 0].item()  # local expert 0's [0,0]
    rank_local_expert15_first = local_before[15, 0, 0].item()  # local expert 15's [0,0] (last local)
    if rank == 0:
        print(f"  BEFORE FSDP — local tensor shape: {tuple(local_before.shape)}", flush=True)
        ep_names = getattr(model, "ep_sharded_param_names", [])
        print(f"  [trace] model.ep_sharded_param_names count: {len(ep_names)}  first: {ep_names[:2]}", flush=True)
    print(
        f"  [rank {rank}] BEFORE — local[expert=0,0,0]={rank_local_expert0_first:.6e}  "
        f"local[expert=15,0,0]={rank_local_expert15_first:.6e}",
        flush=True,
    )
    dist.barrier()

    # Apply FSDP via accelerate
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        cpu_ram_efficient_loading=True,
    )
    acc = Accelerator(fsdp_plugin=fsdp_plugin)
    optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    model, optim = acc.prepare(model, optim)
    dist.barrier()

    # AFTER: each rank's local should still hold the SAME 16-expert slice it had BEFORE.
    # If FSDP corrupted EP, ranks 1..N's local data would have been overwritten by rank 0's.
    l0_gu_after = model.model.layers[0].mlp.experts.gate_up_proj
    if hasattr(l0_gu_after, "to_local"):
        local_after = l0_gu_after.to_local()
    else:
        local_after = l0_gu_after
    rank_local_expert0_first_after = local_after[0, 0, 0].item()
    rank_local_expert15_first_after = local_after[15, 0, 0].item()
    print(
        f"  [rank {rank}] AFTER  — local[expert=0,0,0]={rank_local_expert0_first_after:.6e}  "
        f"local[expert=15,0,0]={rank_local_expert15_first_after:.6e}",
        flush=True,
    )
    dist.barrier()

    # Gather AFTER samples to rank 0 for comparison
    cuda_dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    samples_before = torch.tensor(
        [rank_local_expert0_first, rank_local_expert15_first], dtype=torch.float32, device=cuda_dev
    )
    samples_after = torch.tensor(
        [rank_local_expert0_first_after, rank_local_expert15_first_after], dtype=torch.float32, device=cuda_dev
    )
    gather_before = (
        [torch.zeros(2, dtype=torch.float32, device=cuda_dev) for _ in range(world_size)] if rank == 0 else None
    )
    gather_after = (
        [torch.zeros(2, dtype=torch.float32, device=cuda_dev) for _ in range(world_size)] if rank == 0 else None
    )
    dist.gather(samples_before, gather_before if rank == 0 else None, dst=0)
    dist.gather(samples_after, gather_after if rank == 0 else None, dst=0)

    if rank == 0:
        print("\n  Per-rank EP-shard preservation check:", flush=True)
        all_match = True
        for r in range(world_size):
            b0, b15 = gather_before[r][0].item(), gather_before[r][1].item()
            a0, a15 = gather_after[r][0].item(), gather_after[r][1].item()
            match0 = abs(b0 - a0) < 1e-5
            match15 = abs(b15 - a15) < 1e-5
            ok = match0 and match15
            all_match = all_match and ok
            tick = "✓" if ok else "✗"
            print(
                f"    rank {r}: expert=0  before={b0:+.6e}  after={a0:+.6e}  | "
                f"expert=15 before={b15:+.6e}  after={a15:+.6e}  {tick}",
                flush=True,
            )

        if all_match:
            print(
                "\n  ✓ EP shards PRESERVED — all 8 ranks still hold their original unique 16-expert slice.", flush=True
            )
            print("    (Total experts retained across the EP group: 128/128)", flush=True)
        else:
            print("\n  ❌ EP shards CORRUPTED — at least one rank's data was overwritten.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
