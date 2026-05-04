# ruff: noqa
"""Reproduce PR #45473 verification conditions: small input, EP=8."""

import os

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.distributed import DistributedConfig


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"\n=== EP vs non-EP, world_size={world_size}, tiny input ===\n", flush=True)

    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    model_ep = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        config=cfg,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
    )
    dist.barrier()

    device = model_ep.device

    # Build inputs of various sizes; rank 0 generates them, broadcast to all ranks.
    g = torch.Generator(device="cpu").manual_seed(0)
    inputs = {
        "tiny (1,3)": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "med (1,32)": torch.randint(0, model_ep.config.vocab_size, (1, 32), generator=g),
        "large (2,64)": torch.randint(0, model_ep.config.vocab_size, (2, 64), generator=g),
    }
    for k, v in inputs.items():
        inputs[k] = v.to(device)
        # Broadcast from rank 0 so all ranks have identical token ids
        dist.broadcast(inputs[k], src=0)

    if rank == 0:
        cfg_ref = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
        model_ref = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-30B-A3B",
            config=cfg_ref,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        model_ref.eval()
    model_ep.eval()

    for label, ids in inputs.items():
        with torch.no_grad():
            logits_ep = model_ep(ids).logits
            if rank == 0:
                logits_ref = model_ref(ids).logits

        if rank == 0:
            diff = (logits_ref - logits_ep).abs()
            print(f"\n  [{label}] max_diff={diff.max().item():.4e} mean_diff={diff.mean().item():.4e}", flush=True)
            print(f"    ref logits[0,0,:5]: {logits_ref[0, 0, :5].tolist()}", flush=True)
            print(f"    ep  logits[0,0,:5]: {logits_ep[0, 0, :5].tolist()}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
