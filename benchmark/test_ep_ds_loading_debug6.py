# ruff: noqa
"""
Diagnostic v6: Complete EP+DS loading with WeightConverter + standard path.

Root cause summary:
1. get_init_context 'pass' branch → real tensors on CPU → 30B model OOM/hangs
2. _load_pretrained_model uses zero3 path → GatheredParameters on non-zero3 params → hangs
3. Missing weight_mapping → per-expert ckpt keys can't map to fused model keys

Fix: meta device + standard loading + weight_mapping + EP tp_plan
"""

import copy
import os
import time

import torch
import torch.distributed as dist


rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
_start = time.time()


def log(msg):
    if rank == 0:
        print(f"[{time.time() - _start:6.1f}s][rank0] {msg}", flush=True)


def log_all(msg):
    print(f"[{time.time() - _start:6.1f}s][rank{rank}] {msg}", flush=True)


# Init
log("1. init dist")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)

# Set DS env (like accelerate)
ds_config = {
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 3, "overlap_comm": True, "contiguous_gradients": True},
}
from transformers.integrations.deepspeed import HfDeepSpeedConfig


dschf = HfDeepSpeedConfig(ds_config)
from transformers.integrations import is_deepspeed_zero3_enabled


log(f"   is_ds_zero3={is_deepspeed_zero3_enabled()}")

# Imports
from transformers import initialization as init
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.integrations.tensor_parallel import distribute_model, initialize_tensor_parallelism
from transformers.modeling_utils import (
    LoadStateDictConfig,
    _get_dtype,
    _get_resolved_checkpoint_files,
    apply_patches,
    local_torch_dtype,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from transformers.utils.generic import ContextManagers


MODEL_NAME = "Qwen/Qwen3-30B-A3B"

# Load config
log("2. Loading config + checkpoints")
config, _ = Qwen3MoeForCausalLM.config_class.from_pretrained(MODEL_NAME, return_unused_kwargs=True)
config._attn_implementation = "flash_attention_2"
distributed_config = DistributedConfig(enable_expert_parallel=True)

ep_mesh = dist.init_device_mesh("cuda", (world_size,))
device_map, device_mesh, tp_size = initialize_tensor_parallelism(
    tp_plan=config.base_model_tp_plan, device_mesh=ep_mesh
)
device_map = None  # EP+DS: clear

checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
    pretrained_model_name_or_path=MODEL_NAME,
    variant=None,
    gguf_file=None,
    use_safetensors=None,
    download_kwargs={},
    user_agent={"file_type": "model", "framework": "pytorch"},
    is_remote_code=False,
    transformers_explicit_filename=None,
    tqdm_class=None,
)
config, dtype = _get_dtype(torch.bfloat16, checkpoint_files, config, sharded_metadata, None, True, None)
log(f"   {len(checkpoint_files)} files, dtype={dtype}")

# Create model on META DEVICE
log("3. Creating model on meta")
config2 = copy.deepcopy(config)
init_contexts = [
    local_torch_dtype(dtype, Qwen3MoeForCausalLM.__name__),
    init.no_tie_weights(),
    apply_patches(),
    torch.device("meta"),
    init.meta_device_safe_creation_ops(),
]
with ContextManagers(init_contexts):
    model = Qwen3MoeForCausalLM(config2)
log(f"   OK, first param: {next(model.parameters()).device}")

# distribute_model (EP hooks)
log("4. distribute_model")
model = distribute_model(model, config2.base_model_tp_plan, distributed_config, device_mesh, tp_size)
log("   OK")

# Get weight conversions (WeightConverter for per-expert → fused)
log("5. Getting weight_conversions")
key_mapping = getattr(config2, "transformers_key_mapping", None)
weight_conversions = get_model_conversion_mapping(model, key_mapping, None)
log(f"   Got {len(weight_conversions)} converters")
for wc in weight_conversions[:5]:
    log(f"   {type(wc).__name__}: {getattr(wc, 'source_patterns', None)} -> {getattr(wc, 'target_patterns', None)}")

# Standard weight loading (NOT zero3 path)
log("6. Standard weight loading with weight_mapping")
from safetensors import safe_open
from transformers.core_model_loading import convert_and_load_state_dict_in_model


all_pointer = set()
merged_state_dict = {}
for f in checkpoint_files:
    file_pointer = safe_open(f, framework="pt", device="cpu")
    all_pointer.add(file_pointer)
    for k in file_pointer.keys():
        merged_state_dict[k] = file_pointer.get_slice(k)
log(f"   Loaded {len(merged_state_dict)} keys")

load_config = LoadStateDictConfig(
    pretrained_model_name_or_path=MODEL_NAME,
    sharded_metadata=sharded_metadata,
    device_map=device_map,
    dtype=dtype,
    device_mesh=device_mesh,
    weight_mapping=weight_conversions,  # KEY: include weight converters!
)

try:
    loading_info, _ = convert_and_load_state_dict_in_model(
        model=model,
        state_dict=merged_state_dict,
        load_config=load_config,
        tp_plan=model.tp_plan,  # EP plan for sharding
        disk_offload_index=None,
    )
    log("   Loading OK!")
    log(f"   Missing keys: {len(loading_info.missing_keys)}")
    if loading_info.missing_keys:
        log(f"   Missing: {sorted(loading_info.missing_keys)[:10]}")
    log(f"   Unexpected keys: {len(loading_info.unexpected_keys)}")
    if loading_info.unexpected_keys:
        log(f"   Unexpected: {sorted(loading_info.unexpected_keys)[:5]}")
    log(f"   Mismatched: {len(loading_info.mismatched_keys)}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

for p in all_pointer:
    p.__exit__(None, None, None)

# Check expert weights
log("7. Checking expert weights")
for name, param in model.named_parameters():
    if "experts" in name:
        log(f"   '{name}': shape={param.shape}, device={param.device}, dtype={param.dtype}")
        break

meta_count = sum(1 for p in model.parameters() if p.device.type == "meta")
log(f"   Meta params: {meta_count}")
if meta_count > 0:
    meta_names = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    log(f"   Meta param names: {meta_names[:10]}")

# Finalize (move missing from meta, init missing weights, tie weights)
log("8. Finalizing model")
try:
    loading_info = Qwen3MoeForCausalLM._finalize_model_loading(model, load_config, loading_info)
    model.eval()
    log("   Finalization OK")
    meta_count = sum(1 for p in model.parameters() if p.device.type == "meta")
    log(f"   Meta params after finalize: {meta_count}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

# Forward pass
log("9. Forward pass")
try:
    if meta_count > 0:
        model = model.to_empty(device=f"cuda:{local_rank}")
    else:
        model = model.to(f"cuda:{local_rank}")
    x = torch.randint(0, 1000, (1, 8), device=f"cuda:{local_rank}")
    with torch.no_grad():
        out = model(input_ids=x, use_cache=False)
    log(f"   Forward OK! logits={out.logits.shape}, nan={out.logits.isnan().any()}")
    log(f"   logits max={out.logits.max().item():.3f}, min={out.logits.min().item():.3f}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

# DeepSpeed wrap
log("10. DeepSpeed initialize")
try:
    import deepspeed
    from deepspeed import comm as ds_comm

    if not ds_comm.is_initialized():
        ds_comm.init_distributed("nccl")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    log("   DeepSpeed init OK!")

    # Forward + backward
    x = torch.randint(0, 1000, (1, 8), device=f"cuda:{local_rank}")
    labels = x.clone()
    out = model_engine(input_ids=x, labels=labels, use_cache=False)
    log(f"   Forward loss={out.loss.item():.4f}")
    model_engine.backward(out.loss)
    model_engine.step()
    log("   Backward + step OK!")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
