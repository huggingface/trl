"""Make a prime-rl HF export a complete Qwen3.5 checkpoint (SHARED_W1 rev. 6): every tensor of the base checkpoint present.

Tensors the export lacks (e.g. MTP, or vision if not exported) are copied from the base checkpoint into an extra shard
and added to the index; base assets (preprocessor configs, chat template) are copied if missing. Also reports keys the
export has that the base lacks, shape/dtype mismatches, and whether exported vision weights equal the base (frozen).
The base checkpoint is the run's own model (read from the run's resolved trainer config), not a fixed one.
Usage: python complete_hf.py <run>/hf/step_N [<run>/hf/step_M ...]
"""

import glob
import json
import os
import shutil
import sys

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def base_dir(d):
    """The run's base model (`<run>/hf/step_N` -> `<run>/configs/latest/resolved/trainer.json`), so S and M both work."""
    run = os.path.dirname(os.path.dirname(os.path.abspath(d)))
    name = json.load(open(os.path.join(run, "configs/latest/resolved/trainer.json")))["model"]["name"]
    return snapshot_download(name, local_files_only=True)


def tensor_files(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(d, f) for k, f in wm.items()}
    out = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as h:
            out.update(dict.fromkeys(h.keys(), f))
    return out


def get(path, key):
    with safe_open(path, "pt") as h:
        return h.get_tensor(key)


for d in sys.argv[1:]:
    BASE = base_dir(d)
    base = tensor_files(BASE)
    print(f"{d}: base checkpoint {BASE}")
    exp = tensor_files(d)
    missing = sorted(set(base) - set(exp))
    extra = sorted(set(exp) - set(base))
    bad = []
    for k in sorted(set(base) & set(exp)):
        a, b = get(exp[k], k), get(base[k], k)
        if a.shape != b.shape or a.dtype != b.dtype:
            bad.append((k, tuple(a.shape), str(a.dtype), tuple(b.shape), str(b.dtype)))
    vis = [k for k in sorted(set(base) & set(exp)) if k.startswith("model.visual.")]
    vis_diff = max(
        (float((get(exp[k], k).float() - get(base[k], k).float()).abs().max()) for k in vis[:20]), default=None
    )
    lm = [k for k in sorted(set(base) & set(exp)) if k.startswith("model.language_model.layers.")][:5]
    lm_diff = max((float((get(exp[k], k).float() - get(base[k], k).float()).abs().max()) for k in lm), default=None)
    print(
        f"{d}: export {len(exp)} tensors, base {len(base)}; missing {len(missing)} "
        f"(prefixes {sorted({'.'.join(k.split('.')[:2]) for k in missing})}), extra {extra[:5]}, shape/dtype mismatches {bad[:5]}"
    )
    print(
        f"  vision tensors exported {len(vis)}, max |export - base| over first 20: {vis_diff}; "
        f"language layers max |export - base| over 5 tensors: {lm_diff} (should be > 0 after training)"
    )
    if missing:
        save_file(
            {k: get(base[k], k).contiguous() for k in missing},
            os.path.join(d, "model-base-extra.safetensors"),
            metadata={"format": "pt"},
        )
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(idx):
            j = json.load(open(idx))
        else:
            j = {"metadata": {}, "weight_map": {k: os.path.basename(f) for k, f in exp.items()}}
        j["weight_map"].update(dict.fromkeys(missing, "model-base-extra.safetensors"))
        json.dump(j, open(idx, "w"), indent=2)
        print(f"  copied {len(missing)} tensors from the base checkpoint into model-base-extra.safetensors")
    for f in ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"]:
        if not os.path.exists(os.path.join(d, f)) and os.path.exists(os.path.join(BASE, f)):
            shutil.copy(os.path.join(BASE, f), os.path.join(d, f))
            print(f"  copied {f} from base")
    cfg = json.load(open(os.path.join(d, "config.json")))
    print(
        f"  config architectures {cfg.get('architectures')}, model_type {cfg.get('model_type')}, "
        f"tie_word_embeddings {cfg.get('tie_word_embeddings', cfg.get('text_config', {}).get('tie_word_embeddings'))}"
    )
    if cfg.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        shutil.copy(os.path.join(BASE, "config.json"), os.path.join(d, "config.json"))
        print(
            "  config.json replaced by the base checkpoint's (architectures was not Qwen3_5ForConditionalGeneration)"
        )
