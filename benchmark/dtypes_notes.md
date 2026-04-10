# Dtype Configuration in TRL + FSDP2 Training

## What dtypes are used during training

A training step involves three things, each can have a different dtype:

1. **Parameters** — the model weights. Stored in some dtype, used in forward/backward.
2. **Activations/gradients** — intermediate computation results. Same dtype as whatever the forward pass runs in.
3. **Optimizer state** — Adam maintains fp32 copies of params + momentum + variance. Always fp32.

"bf16 mixed precision training" means: params and activations are bf16 during forward/backward (fast, low memory), but the optimizer updates happen in fp32 (accurate).

## Who controls what

There are **three independent systems** that can set the training dtype, and they interact badly.

### 1. Model dtype — `--dtype bfloat16` (TRL's `ModelConfig`)

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
```

Controls what dtype the model **parameters are loaded in**. If you load in bf16, the forward pass naturally runs in bf16 because `matmul(bf16_weight, bf16_input) → bf16_output`. No casting needed.

**Controls:** parameter storage dtype at load time. That's it.

### 2. Trainer flag — `--bf16 true` (HF `TrainingArguments`)

This is a high-level flag. It does two things:

- Tells the Trainer to use `torch.autocast(dtype=bf16)` for the forward pass
- Sets `self.args.mixed_precision = "bf16"`, which gets passed to Accelerator:

```python
# transformers/trainer.py line 700
self.accelerator = Accelerator(mixed_precision="bf16", ...)
```

**Controls:** autocast context + tells Accelerator to configure mixed precision on the distributed backend (FSDP/DeepSpeed).

**Gotcha:** TRL's `_BaseConfig.__post_init__` (line 105 of `base_config.py`) auto-sets `bf16=True` when it's not explicitly provided:

```python
self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16
```

So `bf16=True` is **always on** in TRL unless you explicitly pass `--bf16 false`.

### 3. Accelerate config — `mixed_precision: bf16` (YAML)

This is what the Accelerator reads from your YAML. But the Trainer's `--bf16` flag **overrides** it:

```python
# The Trainer passes mixed_precision to Accelerator, which takes precedence
Accelerator(mixed_precision="bf16")  # from --bf16 true
Accelerator(mixed_precision="no")    # from --bf16 false
```

When the Accelerator gets `mixed_precision="bf16"` and the backend is FSDP2, it creates:

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,      # cast params to bf16 in forward/backward
    reduce_dtype=torch.bfloat16,     # allreduce gradients in bf16
    output_dtype=torch.bfloat16,     # cast output back to bf16
    cast_forward_inputs=True,        # cast ALL inputs to bf16 (DEFAULT)
)
```

Note: accelerate does NOT set `cast_forward_inputs=False`. There is a TODO comment in `accelerate/utils/dataclasses.py` line 2106:
```python
# TODO(s1ro1): `cast_forward_inputs` for FSDP2?
```

**Controls:** how FSDP2 wraps the model — param casting, gradient reduction dtype, and input tensor casting.

## The override chain

```
accelerate YAML: mixed_precision: 'no'
        ↓
   overridden by
        ↓
Trainer: --bf16 true  →  Accelerator(mixed_precision="bf16")
        ↓
   overridden by (implicitly)
        ↓
TRL _BaseConfig.__post_init__: bf16=True  (auto-set when bf16=None and fp16=False)
```

So even if you set `mixed_precision: 'no'` in the YAML and don't pass `--bf16` on the CLI, TRL's base config auto-sets `bf16=True`, which feeds `mixed_precision="bf16"` to the Accelerator, which creates the `MixedPrecisionPolicy` with `cast_forward_inputs=True`.

## The FSDP2 crash

```
FSDP2 forward hook sees input_ids (dtype=Long)
  → cast_forward_inputs=True
    → input_ids.to(torch.bfloat16)
      → nn.Embedding(input_ids)
        → CRASH: "Expected Long, got BFloat16"
```

`cast_forward_inputs=True` casts **every** tensor to `param_dtype`, including index tensors (`input_ids`, `labels`, `position_ids`) that must stay Long/Int. It's too coarse-grained.

## The fix for benchmarks

```bash
--bf16 false       # Trainer passes mixed_precision="no" to Accelerator
                   # → FSDP2 gets empty MixedPrecisionPolicy (no casting)
                   # → input_ids stays Long

--dtype bfloat16   # model.from_pretrained(dtype=bf16)
                   # → params are bf16, forward runs in bf16 naturally
                   # → no autocast or FSDP casting needed
```

Combined with `mixed_precision: 'no'` in the accelerate YAML.

The model still trains in bf16 (because weights are bf16), but FSDP2 doesn't try to cast inputs.

## Summary table

| Parameter | Layer | What it controls | Side effect with FSDP2 |
|---|---|---|---|
| `--dtype bfloat16` | Model loading | Param storage dtype | None — just loads weights in bf16 |
| `--bf16 true` | HF Trainer | Autocast + `mixed_precision` flag to Accelerator | Triggers FSDP2 `MixedPrecisionPolicy` with `cast_forward_inputs=True` |
| `mixed_precision: bf16` (YAML) | Accelerate config | FSDP2 `MixedPrecisionPolicy` | Same as above, but overridden by Trainer's `--bf16` |
| TRL `_BaseConfig` | TRL defaults | Auto-sets `bf16=True` when not specified | Implicitly triggers the above chain |
