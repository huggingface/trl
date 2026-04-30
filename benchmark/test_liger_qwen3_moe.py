"""
Minimal repro: Liger kernel on Qwen3-MoE — single GPU, no EP.

Goal: see whether Liger's `apply_liger_kernel_to_qwen3_moe(model=model)` works
at all on a tiny Qwen3-MoE config. If it works here we know the failure on the
30B model is EP-specific. If it fails here too, the failure is more fundamental.

Usage:
    srun --partition=hopper-prod --gres=gpu:h100:1 --nodes=1 --ntasks=1 \
        --time=00:05:00 --qos=normal --pty bash -c \
        'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && \
         python /fsx/amine_dirhoussi/trl/benchmark/test_liger_qwen3_moe.py'
"""

# ruff: noqa: T201
import sys
import traceback

import torch


def main():
    print("=" * 80)
    print("Liger + Qwen3-MoE single-GPU repro")
    print("=" * 80)

    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    # Tiny Qwen3-MoE config — same architecture but small enough to fit on 1 GPU and run fast.
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

    print(
        f"Config: hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
        f"experts={cfg.num_experts} (top_k={cfg.num_experts_per_tok}), "
        f"moe_int={cfg.moe_intermediate_size}, vocab={cfg.vocab_size}"
    )
    print()

    # === Step 1: instantiate model on GPU ===
    print("[1/5] Instantiating Qwen3MoeForCausalLM (no Liger yet)...")
    model = Qwen3MoeForCausalLM(cfg).to("cuda", dtype=torch.bfloat16)
    model.eval()

    # Inspect the experts module
    expert_module = model.model.layers[0].mlp.experts
    print(f"   Experts module type: {type(expert_module).__name__}")
    print(f"   gate_up_proj.shape: {tuple(expert_module.gate_up_proj.shape)}")
    print(f"   down_proj.shape:    {tuple(expert_module.down_proj.shape)}")
    print(f"   self.num_experts (set on instance): {expert_module.num_experts}")
    print()

    # === Step 2: forward pass without Liger (sanity) ===
    print("[2/5] Forward pass WITHOUT Liger (sanity check)...")
    input_ids = torch.randint(0, cfg.vocab_size, (1, 32), device="cuda")
    try:
        with torch.no_grad():
            out_ref = model(input_ids).logits
        print(f"   SUCCESS — logits.shape={tuple(out_ref.shape)}, finite={torch.isfinite(out_ref).all().item()}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print()

    # === Step 3: apply Liger ===
    print("[3/5] Applying apply_liger_kernel_to_qwen3_moe(model=model)...")
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe

        apply_liger_kernel_to_qwen3_moe(model=model)
        print("   Liger applied. Inspecting patched experts module...")
        expert_module = model.model.layers[0].mlp.experts
        print(f"   Experts module class name (after patch): {expert_module._get_name()}")
        print(f"   forward fn: {expert_module.forward}")
    except Exception as e:
        print(f"   FAILED to apply Liger: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print()

    # === Step 4: forward pass WITH Liger (the test) ===
    print("[4/5] Forward pass WITH Liger...")
    try:
        with torch.no_grad():
            out_liger = model(input_ids).logits
        print(f"   SUCCESS — logits.shape={tuple(out_liger.shape)}, finite={torch.isfinite(out_liger).all().item()}")
        # Compare to reference
        max_diff = (out_ref - out_liger).abs().max().item()
        print(f"   max |out_ref - out_liger| = {max_diff:.4e}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 2
    print()

    # === Step 5: backward pass ===
    print("[5/5] Forward + backward WITH Liger (training-style)...")
    model.train()
    try:
        labels = torch.randint(0, cfg.vocab_size, (1, 32), device="cuda")
        out = model(input_ids, labels=labels)
        loss = out.loss
        print(f"   forward loss = {loss.item():.4f}")
        loss.backward()
        # Check gate_up_proj has grad
        g = model.model.layers[0].mlp.experts.gate_up_proj.grad
        finite_grad = torch.isfinite(g).all().item() if g is not None else False
        print(
            f"   gate_up_proj.grad exists={g is not None}, finite={finite_grad}, "
            f"max={g.abs().max().item() if g is not None else 'N/A'}"
        )
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 3
    print()

    print("=" * 80)
    print("ALL STEPS PASSED. Liger works on tiny Qwen3-MoE without EP.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
