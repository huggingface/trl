"""
Minimal `from_pretrained` repro under EP, for stepping through the transformers
loading code with pdb. Two functions: DS+EP and FSDP+EP. No optimizer, no
accelerator.prepare, no training step.

Run (1 node, 2 GPUs, tiny MoE — fast pdb iteration):
    srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:2 --ntasks-per-node=1 \\
         --exclusive --time=00:30:00 --qos=normal \\
      bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && \\
        torchrun --standalone --nproc-per-node=2 benchmark/debug_ep_loading.py {ds|fsdp}'

For Qwen3-30B-A3B on 8 GPUs, swap --nproc-per-node=8 and pass --model Qwen/Qwen3-30B-A3B.

Breakpoints in transformers code: gate on RANK==0 to avoid hanging other ranks ::

    if int(os.environ.get("RANK", "0")) == 0:
        breakpoint()
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig


def load_ds_ep(model_name, ep_size):
    """Triggers DS branch in GroupedGemmParallel.post_shard_wrap → plain Parameter,
    tagged with allreduce=False, group_name='ep_size_N'."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=("ds", "fsdp"))
    p.add_argument("--model", default="trl-internal-testing/tiny-Qwen3MoeForCausalLM")
    p.add_argument("--ep-size", type=int, default=int(os.environ.get("WORLD_SIZE", "2")))
    # FSDP only: setting this also flips FSDP_CPU_RAM_EFFICIENT_LOADING=True so
    # `is_fsdp_enabled()` returns True. Without it, FSDP+EP loading is identical to DS+EP
    # (every param goes to cuda:LOCAL_RANK). With it, the model builds on `meta`,
    # `_materialize_copy` skips `.to(device)`, and weights stay as mmap-backed CPU views —
    # the rank-0 broadcast pattern from benchmark/fsdp2_loading_walkthrough.md.
    #
    # ⚠ EP + cpu_ram_efficient_loading is currently broken (C2 in upstream_todo.md):
    #   1. The rank-0 disk-load short-circuit doesn't reach `convert_and_load_state_dict_in_model`,
    #      so every rank still materializes its own copy → flag is a no-op at the loading stage.
    #   2. `fsdp2_prepare_model`'s rank-0 broadcast clobbers each rank's distinct EP expert slice.
    # Use this flag for *debugging* the load-time codepaths only — don't expect a usable model.
    p.add_argument("--fsdp-cpu-ram-efficient", action="store_true")
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    world_size = int(os.environ["WORLD_SIZE"])

    if args.ep_size == world_size:
        ep_mesh = dist.init_device_mesh("cuda", (args.ep_size,), mesh_dim_names=("tp",))

    ep_mesh = dist.init_device_mesh("cuda", (world_size // args.ep_size, args.ep_size), mesh_dim_names=("dp", "tp"))

    if args.backend == "ds":
        _ = HfDeepSpeedConfig({"train_micro_batch_size_per_gpu": 1, "zero_optimization": {"stage": 2}})
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            distributed_config=DistributedConfig(enable_expert_parallel=True),
            device_mesh=ep_mesh,
        )

    else:
        os.environ["ACCELERATE_USE_FSDP"] = "True"
        if args.fsdp_cpu_ram_efficient:
            os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            distributed_config=DistributedConfig(enable_expert_parallel=True),
            device_mesh=ep_mesh,
        )

    rank = int(os.environ["RANK"])
    if rank == 0:
        print(f"backend={args.backend} has_ep={model.has_ep} _tp_size={model._tp_size}")

    # Print one expert and one dense param from EVERY rank — this is the only way to see what
    # `--fsdp-cpu-ram-efficient` actually changes. Path B in `_move_missing_keys_from_meta_to_device`
    # only fires on non-rank-0, replacing post-load cuda values with cpu zeros placeholders. Without
    # printing from rank > 0 the flag looks like a no-op (rank 0 always stays on cuda).
    expert = next((p for n, p in model.named_parameters() if "experts" in n and "layers.0" in n), None)
    dense = next((p for n, p in model.named_parameters() if "self_attn.q_proj" in n and "layers.0" in n), None)
    msg = (
        f"[rank {rank}] expert: DTensor={isinstance(expert.data, DTensor) if expert is not None else None} "
        f"device={expert.device if expert is not None else None} | "
        f"dense: device={dense.device if dense is not None else None} "
        f"all_zero={(dense.data == 0).all().item() if dense is not None else None}"
    )
    # Serialize prints by rank for readability
    for r in range(int(os.environ["WORLD_SIZE"])):
        if r == rank:
            print(msg, flush=True)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
