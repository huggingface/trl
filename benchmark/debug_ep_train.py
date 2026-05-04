# ruff: noqa
"""5-step SFT training with EP=8 on single node — verifies training correctness.

If first-step loss ~1.6 → fix works end-to-end.
If loss ~62 → corruption persists despite fix.
"""

import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"\n=== EP=8 single-node correctness test, world_size={world_size} ===\n", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
    )
    dist.barrier()

    # Wrap with FSDP2 via accelerate (cpu_ram_efficient_loading=False — needed for EP fix)
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        cpu_ram_efficient_loading=False,
    )
    # User-side opt-in: tell FSDP to skip the EP-sharded modules. Uses accelerate's
    # existing `ignored_modules` API, no accelerate patch needed.
    ep_module_names = list({n.rsplit(".", 1)[0] for n in model.ep_sharded_param_names})
    fsdp_plugin.ignored_modules = [model.get_submodule(n) for n in ep_module_names]
    if rank == 0:
        print(f"  ignored_modules: {len(fsdp_plugin.ignored_modules)} experts modules", flush=True)
    acc = Accelerator(fsdp_plugin=fsdp_plugin)
    # Verify EP params are DTensors (transformers should have wrapped them at load time)
    from torch.distributed.tensor import DTensor

    n_ep_dt = sum(
        1 for n in model.ep_sharded_param_names if isinstance(dict(model.named_parameters())[n].data, DTensor)
    )
    if rank == 0:
        print(
            f"  EP params that are DTensors after from_pretrained: {n_ep_dt}/{len(model.ep_sharded_param_names)}",
            flush=True,
        )

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    model, optim = acc.prepare(model, optim)

    if rank == 0:
        print(f"  acc.prepare done. ep_param count: {len(model.ep_sharded_param_names)}", flush=True)

    # Load a small batch from real dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    if rank == 0:
        ds = load_dataset("THUDM/LongAlign-10k", split="train").select(range(4))
        text = "\n".join(d.get("text", str(d)) for d in ds)
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
    else:
        ids = torch.zeros((1, 1024), dtype=torch.long)

    ids = ids.to(model.device if hasattr(model, "device") else f"cuda:{rank}")
    dist.broadcast(ids, src=0)
    labels = ids.clone()

    model.train()
    for step in range(5):
        optim.zero_grad()
        out = model(input_ids=ids, labels=labels)
        loss = out.loss
        loss.backward()
        # Check grad norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gn = p.grad.detach().to_local() if hasattr(p.grad, "to_local") else p.grad.detach()
                grad_norm += gn.float().pow(2).sum().item()
        grad_norm = grad_norm**0.5
        optim.step()
        if rank == 0:
            print(f"  step {step}: loss={loss.item():.4f}  grad_norm={grad_norm:.4e}", flush=True)
            if step == 0:
                if loss.item() < 5:
                    print("  ✓ FIRST-STEP LOSS HEALTHY (<5) — EP+FSDP correctness fix works!", flush=True)
                else:
                    print(
                        f"  ❌ first-step loss={loss.item():.2f} — expected ~1.6, EP corruption persists", flush=True
                    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
