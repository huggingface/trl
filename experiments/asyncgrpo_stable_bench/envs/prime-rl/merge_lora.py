"""Fold a prime-rl PEFT adapter (filesystem-broadcast dir) into the base HF checkpoint -> complete HF dir for eval.

prime-rl ships no LoRA export (tools/convert_dcp_to_bf16.py refuses LoRA checkpoints), so this is bench tooling, not a
library change. W += (alpha / r) * B @ A in fp32, cast back to the base dtype; every other tensor (routed experts, router,
vision tower, MTP head) is copied byte for byte, and every non-weight file (config, tokenizer, chat template, preprocessor
configs) is copied, so the output loads like the base checkpoint.
Usage: python merge_lora.py <adapter_dir> <out_dir> [base_model]   (base defaults to the adapter's base_model_name_or_path)
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


adapter_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
base = Path(snapshot_download(sys.argv[3] if len(sys.argv) > 3 else cfg["base_model_name_or_path"], local_files_only=True))
scaling = cfg["lora_alpha"] / cfg["r"]

lora = {}
for f in sorted(adapter_dir.glob("adapter_model*.safetensors")):
    with safe_open(f, "pt") as fh:
        for k in fh.keys():
            name = k.removeprefix("base_model.model.")
            module, kind = name.rsplit(".lora_", 1)
            lora.setdefault(module, {})[kind.split(".")[0]] = fh.get_tensor(k)
assert lora and all(set(v) == {"A", "B"} for v in lora.values()), "adapter must hold lora_A and lora_B per module"

out_dir.mkdir(parents=True, exist_ok=True)
for f in base.iterdir():
    if not f.name.endswith(".safetensors"):
        shutil.copy(f, out_dir / f.name)

merged, max_delta = set(), 0.0
for shard in sorted(base.glob("*.safetensors")):
    tensors = {}
    with safe_open(shard, "pt") as fh:
        for k in fh.keys():
            t = fh.get_tensor(k)
            module = k.removesuffix(".weight")
            if k.endswith(".weight") and module in lora:
                delta = scaling * (lora[module]["B"].float() @ lora[module]["A"].float())
                assert delta.shape == t.shape, (k, delta.shape, t.shape)
                t = (t.float() + delta).to(t.dtype)
                merged.add(module)
                max_delta = max(max_delta, delta.abs().max().item())
            tensors[k] = t.contiguous()
    save_file(tensors, out_dir / shard.name, metadata={"format": "pt"})

missing = sorted(set(lora) - merged)
assert not missing, f"{len(missing)} adapter modules have no base weight, e.g. {missing[:3]}"
print(json.dumps({"adapter": str(adapter_dir), "out": str(out_dir), "base": str(base), "modules_merged": len(merged),
                  "scaling": scaling, "max_abs_delta": max_delta}))
