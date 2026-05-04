# ruff: noqa
"""
EP repro: Liger + Qwen3-MoE with transformers Expert Parallelism.

Single-GPU passed (see `test_liger_qwen3_moe.py`). Now test with EP — does Liger
break specifically because of how EP shards experts and uses sentinel routing?

EP path uses transformers' `from_pretrained(distributed_config=...)` which
shards `gate_up_proj` to `[num_local_experts, ...]` and replaces non-local
expert IDs in `top_k_index` with a sentinel.

Run with 2 GPUs, EP=2:
    srun --partition=hopper-prod --gres=gpu:h100:2 --nodes=1 --ntasks=1 \
        --time=00:10:00 --qos=normal --pty bash -c \
        'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && \
         torchrun --nproc_per_node=2 /fsx/amine_dirhoussi/trl/benchmark/test_liger_qwen3_moe_ep.py'
"""

# ruff: noqa: T201
import os
import sys
import traceback

import torch
import torch.distributed as dist


def main():
    # Distributed init
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log("=" * 80)
    log(f"Liger + Qwen3-MoE + EP={world_size} repro")
    log("=" * 80)

    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.distributed.configuration_utils import DistributedConfig

    # Tiny Qwen3-MoE config
    cfg = Qwen3MoeConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        vocab_size=1024,
        max_position_embeddings=2048,
        torch_dtype=torch.bfloat16,
    )

    log(
        f"Config: experts={cfg.num_experts}, per-rank with EP={world_size} → "
        f"{cfg.num_experts // world_size} local experts/rank"
    )
    log("")

    # === Step 1: build a fresh state_dict (no checkpoint needed) ===
    log(
        "[1/5] Instantiating model on meta device, materializing weights, "
        "saving state_dict for from_pretrained EP load..."
    )
    save_dir = f"/tmp/qwen3_moe_tiny_rank{rank}"
    if rank == 0:
        # Build on rank 0, save state_dict, others wait
        cfg.save_pretrained("/tmp/qwen3_moe_tiny_cfg")
        m = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
        m.save_pretrained("/tmp/qwen3_moe_tiny_cfg")
    dist.barrier()
    log("   saved.")
    log("")

    # === Step 2: load with EP via from_pretrained ===
    log(
        "[2/5] Loading with EP via from_pretrained(distributed_config=DistributedConfig(enable_expert_parallel=True))..."
    )
    device_mesh = dist.init_device_mesh("cuda", (world_size,))
    try:
        model = Qwen3MoeForCausalLM.from_pretrained(
            "/tmp/qwen3_moe_tiny_cfg",
            distributed_config=DistributedConfig(enable_expert_parallel=True),
            device_mesh=device_mesh,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = model.to(f"cuda:{local_rank}")
        model.eval()
    except Exception as e:
        log(f"   FAILED to load with EP: {type(e).__name__}: {e}")
        if rank == 0:
            traceback.print_exc()
        return 1

    # Inspect experts module
    expert_module = model.model.layers[0].mlp.experts
    if rank == 0:
        log(f"   Experts module type: {type(expert_module).__name__}")
        log(f"   self.num_experts: {expert_module.num_experts}")
        log(f"   gate_up_proj type: {type(expert_module.gate_up_proj).__name__}")
        try:
            log(f"   gate_up_proj.shape: {tuple(expert_module.gate_up_proj.shape)}")
        except Exception:
            pass
        log(f"   has_ep on model: {getattr(model, 'has_ep', None)}")
        log(f"   model.config._experts_implementation: {getattr(model.config, '_experts_implementation', '<unset>')}")
        log(f"   expert_module.forward: {expert_module.forward}")
        log(f"   expert_module.forward.__func__: {getattr(expert_module.forward, '__func__', None)}")
    log("")

    # === Step 3: forward pass without Liger (sanity) — also trace ===
    log(f"[3/5] Forward pass WITHOUT Liger (EP={world_size})...")
    input_ids = torch.randint(0, cfg.vocab_size, (1, 32), device=f"cuda:{local_rank}")

    # Same trace on eager
    if rank == 0:
        eager_orig_fwd = model.model.layers[0].mlp.experts.forward

        def eager_traced_forward(self, hidden_states, top_k_index, top_k_weights):
            print(
                f"   [trace EAGER rank0 layer0] top_k_index dtype={top_k_index.dtype}, "
                f"shape={tuple(top_k_index.shape)}, "
                f"min={top_k_index.min().item()}, max={top_k_index.max().item()}, "
                f"unique={torch.unique(top_k_index).tolist()}",
                flush=True,
            )
            print(
                f"   [trace EAGER rank0 layer0] self.num_experts={self.num_experts}, "
                f"gate_up_proj.shape={tuple(self.gate_up_proj.shape)}",
                flush=True,
            )
            return eager_orig_fwd(hidden_states, top_k_index, top_k_weights)

        import types as _types

        model.model.layers[0].mlp.experts.forward = _types.MethodType(
            eager_traced_forward, model.model.layers[0].mlp.experts
        )

    try:
        with torch.no_grad():
            out_ref = model(input_ids).logits
        finite = torch.isfinite(out_ref).all().item()
        log(f"   SUCCESS — logits.shape={tuple(out_ref.shape)}, finite={finite}")
    except Exception as e:
        log(f"   FAILED: {type(e).__name__}: {e}")
        if rank == 0:
            traceback.print_exc()
        return 1
    log("")

    # === Step 4: apply Liger WITHOUT swiglu — the proposed workaround ===
    log("[4/5] Applying apply_liger_kernel_to_qwen3_moe(model=model, swiglu=False) — keeps grouped_mm experts...")
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

        apply_liger_kernel_to_qwen3_moe(
            rope=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=True,
            swiglu=False,  # ← key: keep transformers' EP-aware experts forward
            model=model,
        )
        if rank == 0:
            expert_module = model.model.layers[0].mlp.experts
            log(f"   Patched. _get_name() = {expert_module._get_name()}")
            log(f"   self.num_experts after patch: {expert_module.num_experts}")
    except Exception as e:
        log(f"   FAILED to apply Liger: {type(e).__name__}: {e}")
        if rank == 0:
            traceback.print_exc()
        return 1
    log("")

    log(f"[5/5] Forward pass WITH Liger (EP={world_size}, swiglu=False)...")
    try:
        torch.cuda.synchronize()
        with torch.no_grad():
            out_liger = model(input_ids).logits
        torch.cuda.synchronize()
        finite = torch.isfinite(out_liger).all().item()
        log(f"   SUCCESS (unexpected) — logits.shape={tuple(out_liger.shape)}, finite={finite}")
        max_diff = (out_ref - out_liger).abs().max().item()
        log(f"   max |out_ref - out_liger| = {max_diff:.4e}")
    except Exception as e:
        log(f"   FAILED at: {type(e).__name__}: {e}")
        if rank == 0:
            traceback.print_exc()
        return 2
    log("")

    log("=" * 80)
    log("EP repro complete.")
    log("=" * 80)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
