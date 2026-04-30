Hey 👋 running into a nasty `torch.compile` regression with accelerate's FSDP2 path and wanted to flag it — would love a second pair of eyes.

**TL;DR**: per-layer `torch.compile` + raw `fully_shard()` gives **1.28× speedup** (25% → 32% MFU). Same compiled model through `accelerate.fsdp2_prepare_model()` is **2.4× slower than eager** (23% → 10% MFU). The compile itself is clean — zero graph breaks, zero recompiles — so something in the accelerate wrapping path is invalidating the compiled kernels.

**Setup**: Qwen3-30B-A3B (MoE, 128 experts, 48 layers) · 2×8 H100 SXM 80GB · FSDP2 DP=16 · seq_len=16384 · SFTTrainer w/ grad ckpt, bf16, packing.

| Setup | MFU | ms/step | vs eager |
|-------|-----|---------|----------|
| raw `fully_shard` + eager | 25.0% | 3,888 | — |
| raw `fully_shard` + per-layer compile | **32.1%** | **3,031** | **1.28× faster** |
| accelerate `fsdp2_prepare_model` + eager | 23.4% | 4,160 | — |
| accelerate `fsdp2_prepare_model` + per-layer compile | 9.8% | 9,900 | **2.4× slower** |

**What's different in `fsdp2_prepare_model` vs raw `fully_shard`** (`accelerate/utils/fsdp_utils.py:621-746`):
1. `set_auto_wrap_policy(model)` — scans for transformer layer class
2. `model.to("meta")` when `cpu_ram_efficient_loading=True`
3. `fully_shard()` per layer + root
4. `fsdp2_load_full_state_dict()` — reloads weights from saved state dict
5. `tie_weights()`
6. fp32 upcast of trainable params when `mixed_precision != "no"`

**Ruled out individually** (each still hits ~9.8% MFU):
- fp32 upcast → tested with `bf16=False, mixed_precision=no` → still slow
- `cpu_ram_efficient_loading=false` → still slow
- SFT-specific compute_loss (entropy/accuracy/logits) → still slow
- SFT data collator → still slow
- Recompilations → `TORCH_LOGS=recompiles` → zero
- Input shapes / attention_mask handling → identical `(1, 16384)` in both paths
- `torch._dynamo.reset()` before compile → still slow
- HF Trainer (not SFTTrainer) → fast, but uses different collator path

Repros (both scripts are self-contained, same model/data/loop, only FSDP2 application differs):
- Fast path (raw `fully_shard`): https://gist.github.com/AmineDiro/be1fd63af7f715c9431ee378d5d79dc6
- Slow path (accelerate `fsdp2_prepare_model`): https://gist.github.com/AmineDiro/2457fbee70662d584a116cc3ca80dd07

Questions:
1. Any idea which of steps 2–6 is the actual culprit? Meta round-trip + state dict reload is my top suspect but individual tests didn't pin it down.
2. Could `fsdp2_prepare_model` detect pre-compiled layers (`hasattr(layer, '_compiled_call_impl')`) and use a lighter wrapping path matching raw `fully_shard`?
3. Option to skip the meta round-trip + fp32 upcast when the model's already on the right device/dtype?

Happy to dig further / test patches. Full writeup w/ environment + torchtitan reference: <attach `accelerate_fsdp_compile_issue.md` if needed>
