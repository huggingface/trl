"""Turn a verl LoRA-only FSDP2 checkpoint (actor.checkpoint.save_lora_only=true: model_world_size_W_rank_R.pt holding
only the lora_A/lora_B DTensor shards) into a PEFT adapter dir that launch/trl/merge_lora.py merges into
the base for the shared eval. Eval glue: verl's own model_merger writes a full-size HF model next to the adapter, which a
LoRA-only checkpoint can't fill.

Keeps the language-model adapters only. target_modules=all-linear also wraps the vision tower's Linears, which get no
gradient on text-only data; their lora_B must still be all zero (asserted, recorded in export_check.json).

    python t2sql_export.py <ckpt>/global_step_N/actor <out_adapter_dir>
"""

import glob
import json
import os
import re
import sys

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard


actor_dir, out = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(os.path.join(actor_dir, "model_world_size_*_rank_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
shards = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
meta = json.load(open(os.path.join(actor_dir, "lora_train_meta.json")))
full = {}
for key, value in shards[0].items():
    if isinstance(value, DTensor):
        placement = value.placements[0]
        locals_ = [s[key]._local_tensor for s in shards]
        full[key] = torch.cat(locals_, dim=placement.dim) if isinstance(placement, Shard) else locals_[0]
        assert isinstance(placement, (Shard, Replicate)), placement
    else:
        full[key] = value
adapter, dropped_b_max, modules = {}, 0.0, set()
for key, value in full.items():
    name = key.replace(".default.weight", ".weight")
    if ".language_model." not in name:
        if ".lora_B." in name:
            dropped_b_max = max(dropped_b_max, value.abs().max().item())
        continue
    adapter[name] = value.to(torch.bfloat16).contiguous()
    modules.add(re.sub(r"\.lora_[AB]\.weight$", "", name).rsplit(".", 1)[1])
assert dropped_b_max == 0.0, f"non-language-model lora_B is nonzero ({dropped_b_max})"
os.makedirs(out, exist_ok=True)
save_file(adapter, os.path.join(out, "adapter_model.safetensors"))
config = {
    "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": meta["r"], "lora_alpha": meta["lora_alpha"],
    "target_modules": ".*language_model.*\\.(" + "|".join(sorted(modules)) + ")", "lora_dropout": 0.0, "bias": "none",
    "base_model_name_or_path": "Qwen/Qwen3.5-35B-A3B",
}
json.dump(config, open(os.path.join(out, "adapter_config.json"), "w"), indent=1)
check = {
    "actor_dir": actor_dir, "ranks": len(paths), "lora_tensors": len(full), "exported": len(adapter),
    "wrapped_modules": len(adapter) // 2, "module_names": sorted(modules), "dropped_non_lm_lora_B_absmax": dropped_b_max,
    "lora_B_nonzero": sum(int(v.abs().sum() > 0) for k, v in adapter.items() if ".lora_B." in k),
}
json.dump(check, open(os.path.join(out, "export_check.json"), "w"), indent=1)
print(json.dumps(check))
