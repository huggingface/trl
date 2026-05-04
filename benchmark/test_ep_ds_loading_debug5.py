# ruff: noqa
"""
Diagnostic v5: Same as v4 but investigate missing keys and fix expert loading.
v4 showed: 96 missing keys, expert weights still on meta device.
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


# Step 1: Init
log("1. init dist")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)

# Step 2: Set DS env
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

# Step 3: Setup
from transformers import initialization as init
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.integrations.tensor_parallel import distribute_model, initialize_tensor_parallelism
from transformers.modeling_utils import (
    LoadStateDictConfig,
    _get_dtype,
    _get_resolved_checkpoint_files,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from transformers.utils.generic import ContextManagers


config, _ = Qwen3MoeForCausalLM.config_class.from_pretrained("Qwen/Qwen3-30B-A3B", return_unused_kwargs=True)
config._attn_implementation = "flash_attention_2"
distributed_config = DistributedConfig(enable_expert_parallel=True)

ep_mesh = dist.init_device_mesh("cuda", (world_size,))
device_map, device_mesh, tp_size = initialize_tensor_parallelism(
    tp_plan=config.base_model_tp_plan, device_mesh=ep_mesh
)
device_map = None  # EP+DS: clear device_map

checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
    pretrained_model_name_or_path="Qwen/Qwen3-30B-A3B",
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

# Create model on meta
log("2. Creating model on meta")
from transformers.modeling_utils import apply_patches, local_torch_dtype


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
log("   Model on meta OK")

# distribute_model
log("3. distribute_model")
model = distribute_model(model, config2.base_model_tp_plan, distributed_config, device_mesh, tp_size)
log(f"   OK. tp_plan type: {'ep_plan' if hasattr(config2, 'distributed_config') else 'tp_plan'}")

# Check what ep_plan contains
ep_plan = model.tp_plan  # returns ep_plan when EP enabled
log(f"4. EP plan keys ({len(ep_plan)}):")
for k, v in ep_plan.items():
    log(f"   {k}: {v}")

# Check model state dict
log("5. Model state dict (meta)")
meta_sd = model.state_dict()
expert_keys_in_model = [k for k in meta_sd if "experts" in k]
log(f"   Total model keys: {len(meta_sd)}")
log(f"   Expert keys in model: {len(expert_keys_in_model)}")
if expert_keys_in_model:
    log(f"   Sample expert keys: {expert_keys_in_model[:5]}")

# Check checkpoint keys
log("6. Checkpoint keys")
from safetensors import safe_open


all_ckpt_keys = set()
for f in checkpoint_files:
    fp = safe_open(f, framework="pt", device="cpu")
    for k in fp.keys():
        all_ckpt_keys.add(k)
    fp.__exit__(None, None, None)
expert_keys_in_ckpt = [k for k in sorted(all_ckpt_keys) if "experts" in k]
log(f"   Total ckpt keys: {len(all_ckpt_keys)}")
log(f"   Expert keys in ckpt: {len(expert_keys_in_ckpt)}")
if expert_keys_in_ckpt:
    log(f"   Sample ckpt expert keys: {expert_keys_in_ckpt[:5]}")

# Check key mismatch
model_expert_set = set(expert_keys_in_model)
ckpt_expert_set = set(expert_keys_in_ckpt)
only_in_model = model_expert_set - ckpt_expert_set
only_in_ckpt = ckpt_expert_set - model_expert_set
log("7. Key comparison (experts only):")
log(f"   Only in model: {len(only_in_model)}")
if only_in_model:
    log(f"   Sample: {list(only_in_model)[:5]}")
log(f"   Only in ckpt: {len(only_in_ckpt)}")
if only_in_ckpt:
    log(f"   Sample: {list(only_in_ckpt)[:5]}")

# Now try loading with tp_plan={} (skip EP sharding during load, like the _ep_skip_tp_load flag)
log("8. Loading with tp_plan={} (skip TP/EP sharding)")
all_pointer = set()
merged_state_dict = {}
for f in checkpoint_files:
    file_pointer = safe_open(f, framework="pt", device="cpu")
    all_pointer.add(file_pointer)
    for k in file_pointer.keys():
        merged_state_dict[k] = file_pointer.get_slice(k)

from transformers.core_model_loading import convert_and_load_state_dict_in_model


load_config = LoadStateDictConfig(
    pretrained_model_name_or_path="Qwen/Qwen3-30B-A3B",
    sharded_metadata=sharded_metadata,
    device_map=device_map,
    dtype=dtype,
    device_mesh=device_mesh,
)
try:
    loading_info, _ = convert_and_load_state_dict_in_model(
        model=model,
        state_dict=merged_state_dict,
        load_config=load_config,
        tp_plan={},  # Skip EP sharding — just load weights straight
        disk_offload_index=None,
    )
    log(f"   Loading OK! Missing: {len(loading_info.missing_keys)}")
    if loading_info.missing_keys:
        missing_list = sorted(loading_info.missing_keys)
        log(f"   Missing keys sample: {missing_list[:10]}")

    # Check expert param
    for name, param in model.named_parameters():
        if "experts" in name:
            log(f"   Expert '{name}': shape={param.shape}, device={param.device}")
            break

except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

for p in all_pointer:
    p.__exit__(None, None, None)

# Step 9: Move to GPU and try forward
log("9. Moving to GPU and forward")
try:
    model = model.to_empty(device=f"cuda:{local_rank}")
    # Check for meta tensors
    meta_count = sum(1 for p in model.parameters() if p.device.type == "meta")
    log(f"   Meta params remaining: {meta_count}")
    if meta_count > 0:
        for name, param in model.named_parameters():
            if param.device.type == "meta":
                log(f"   Still meta: {name}")
                break

    x = torch.randint(0, 1000, (1, 8), device=f"cuda:{local_rank}")
    with torch.no_grad():
        out = model(input_ids=x, use_cache=False)
    log(f"   Forward OK! logits={out.logits.shape}, nan={out.logits.isnan().any()}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
