"""Turn a NeMo-RL Automodel consolidated HF export (<ckpt>/policy/weights/model/consolidated) into a complete
Qwen3.5-35B-A3B checkpoint dir for the shared vLLM eval, without copying the weights:
- the export's safetensors shards are symlinked;
- base tensors the export lacks (e.g. vision tower or MTP, never trained) go into one extra shard;
- base non-weight files (config, tokenizer, chat template, preprocessor configs) are copied when missing;
- a merged model.safetensors.index.json is written. Shape/dtype mismatches and extra keys fail loudly.
Writes complete.json (counts) last, which the eval helper waits for.

    python complete_ckpt.py <consolidated_dir> <out_dir> [base_model]
"""

import glob
import json
import os
import shutil
import sys

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


src, out = sys.argv[1], sys.argv[2]
base = snapshot_download(sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3.5-35B-A3B", local_files_only=True)


def tensor_files(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def meta(path, key):
    with safe_open(path, "pt") as h:
        s = h.get_slice(key)
        return tuple(s.get_shape()), s.get_dtype()


b, e = tensor_files(base), tensor_files(src)
extra = sorted(set(e) - set(b))
assert not extra, f"{len(extra)} exported keys not in base, e.g. {extra[:5]}"
# ALLOW_DTYPE=1: accept dtype-only differences (NeMo-RL's Megatron export writes Qwen3.5's fp32 GDN A_log/norm in bf16)
same = (lambda x, y: x[0] == y[0]) if os.environ.get("ALLOW_DTYPE") == "1" else (lambda x, y: x == y)
bad = [k for k in sorted(e) if not same(meta(e[k], k), meta(b[k], k))]
dtype_diff = sorted(k for k in e if meta(e[k], k)[1] != meta(b[k], k)[1])
assert not bad, f"{len(bad)} shape/dtype mismatches, e.g. {[(k, meta(e[k], k), meta(b[k], k)) for k in bad[:3]]}"
missing = sorted(set(b) - set(e))

os.makedirs(out, exist_ok=True)
weight_map = {}
for f in sorted(set(e.values())):
    dst = os.path.join(out, os.path.basename(f))
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(f), dst)
for k, f in e.items():
    weight_map[k] = os.path.basename(f)
if missing:
    tensors = {}
    for k in missing:
        with safe_open(b[k], "pt") as h:
            tensors[k] = h.get_tensor(k)
    save_file(tensors, os.path.join(out, "base-untrained.safetensors"), metadata={"format": "pt"})
    weight_map.update(dict.fromkeys(missing, "base-untrained.safetensors"))
for f in os.listdir(base):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.join(base, f), os.path.join(out, f))
json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
groups = {}
for k in missing:
    groups[k.split(".")[1] if k.startswith("model.") else k.split(".")[0]] = groups.get(k.split(".")[1] if k.startswith("model.") else k.split(".")[0], 0) + 1
summary = {"src": src, "base": base, "exported": len(e), "missing_from_base": len(missing), "missing_groups": groups, "dtype_differs": len(dtype_diff)}
json.dump(summary, open(os.path.join(out, "complete.json"), "w"))
print(json.dumps(summary))
