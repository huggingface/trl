"""
Merge a LoRA checkpoint of `launch/trl/moe_w2.py` (PEFT adapter dir, e.g. <run>/checkpoint-20) into the base model and save a
full HF model that vLLM serves (W2 eval: launch/trl/w2.sbatch STAGE=probe EVAL_MODEL=<out>). Runs on the CPU of a GPU
node (35B in bf16 = ~70 GB of host memory).

    python merge_lora.py <adapter_dir> <out_dir> [base_model]

Checks that the merge changed exactly the LoRA-wrapped tensors and nothing else (the routed experts, router and vision
tower stay bit-identical to the base), and writes the counts to <out_dir>/merge_check.json.
"""

import json
import os
import shutil
import sys
import time

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText


adapter_dir, out_dir = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else json.load(open(os.path.join(adapter_dir, "adapter_config.json")))[
    "base_model_name_or_path"
]
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
before = {name: p.detach().clone() for name, p in model.named_parameters()}
model = PeftModel.from_pretrained(model, adapter_dir)
n_lora_b_nonzero = sum(int(p.abs().sum() > 0) for n, p in model.named_parameters() if "lora_B" in n)
model = model.merge_and_unload()
after = dict(model.named_parameters())
changed = [n for n in before if not torch.equal(before[n], after[n].detach())]
check = {
    "adapter": adapter_dir,
    "base": base,
    "tensors": len(before),
    "changed": len(changed),
    "lora_B_nonzero": n_lora_b_nonzero,
    "changed_outside_language_model": [n for n in changed if "language_model" not in n],
    "changed_experts_or_router": [n for n in changed if ".mlp.experts." in n or n.endswith(".mlp.gate.weight")],
    "changed_sample": changed[:5],
}
assert set(after) == set(before), "merge changed the parameter names"
assert not check["changed_outside_language_model"] and not check["changed_experts_or_router"], check
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir, max_shard_size="5GB")
# Tokenizer, chat template, processor and generation configs from the base snapshot (vLLM loads the processor for this
# multimodal architecture even with images off); config.json and the weights come from save_pretrained above.
snapshot = snapshot_download(base) if not os.path.isdir(base) else base
for name in os.listdir(snapshot):
    if not name.endswith((".safetensors", ".bin", ".pth")) and name not in ("config.json", "model.safetensors.index.json"):
        src = os.path.join(snapshot, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(out_dir, name))
check["wall_s"] = round(time.time() - t0, 1)
json.dump(check, open(os.path.join(out_dir, "merge_check.json"), "w"), indent=1)
print(json.dumps(check, indent=1))
