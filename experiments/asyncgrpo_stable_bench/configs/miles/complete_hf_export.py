"""Make a miles HF export of a Qwen3.5 VLM checkpoint loadable as the original model.

miles trains and exports only the language model (`--save-hf`, megatron_to_hf_mode=raw); the vision tower is frozen and
never synced, so the export lacks every tensor the base checkpoint has outside the text model (the `visual.*` weights,
and the MTP head if the base has one). This copies those missing tensors, unchanged, from the base checkpoint into
one extra shard and adds them to the index. Tensors present in the export are never touched. Idempotent.

    python complete_hf_export.py <base hf dir> <export dir>
"""

import glob
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file


def tensor_files(d):
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def main(base, export):
    index_path = os.path.join(export, "model.safetensors.index.json")
    index = json.load(open(index_path))
    have = set(index["weight_map"])
    extra_name = "model-base-frozen.safetensors"
    if extra_name in set(index["weight_map"].values()):
        print(f"{export}: already completed")
        return
    missing = {}
    for f in tensor_files(base):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k not in have:
                    missing[k] = s.get_tensor(k)
    if not missing:
        print(f"{export}: nothing missing")
        return
    save_file(missing, os.path.join(export, extra_name), metadata={"format": "pt"})
    for k, t in missing.items():
        index["weight_map"][k] = extra_name
        index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + t.numel() * t.element_size()
    json.dump(index, open(index_path, "w"), indent=2)
    prefixes = sorted({".".join(k.split(".")[:2]) for k in missing})
    print(f"{export}: added {len(missing)} frozen tensors from {base} (prefixes {prefixes})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
