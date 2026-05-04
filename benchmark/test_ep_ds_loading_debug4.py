# ruff: noqa
"""
Diagnostic v4: Test the correct approach for EP+DS loading.

Root cause analysis from v3:
- The 'pass' branch in get_init_context creates 30B model with REAL tensors on CPU
  (4 procs * 60GB = 240GB CPU RAM, extremely slow). This is wrong.
- The torchrun test (that works) uses META DEVICE (fast) via the else branch.
- The difference: torchrun has is_deepspeed_zero3_enabled()=False, so it uses else branch.

Fix hypothesis:
1. Use META DEVICE for model creation (same as torchrun/normal path)
2. Force STANDARD weight loading path (not zero3) - this calls shard_and_distribute_module
   which correctly handles EP sharding via the hooks registered by distribute_model.

This test validates both:
a) Meta device + distribute_model (hooks only) works
b) Standard weight loading with EP sharding works, even when DS env is set
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


# Step 1: Init torch distributed
log("1. init torch distributed")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
log(f"   OK ws={world_size}")

# Step 2: Set DS env vars LIKE accelerate does
ds_config = {
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
}

log("2. Setting DS env vars")
from transformers.integrations.deepspeed import HfDeepSpeedConfig


dschf = HfDeepSpeedConfig(ds_config)
from transformers.integrations import is_deepspeed_zero3_enabled


log(f"   is_deepspeed_zero3_enabled() = {is_deepspeed_zero3_enabled()}")

# Step 3: Import
log("3. Importing")
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


log("   OK")

# Step 4: Load config
log("4. Loading config")
config, _ = Qwen3MoeForCausalLM.config_class.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    return_unused_kwargs=True,
)
config._attn_implementation = "flash_attention_2"
distributed_config = DistributedConfig(enable_expert_parallel=True)
log(f"   config: {config.model_type}, layers={config.num_hidden_layers}")

# Step 5: Device mesh + TP init
log("5. Device mesh + TP init")
ep_mesh = dist.init_device_mesh("cuda", (world_size,))
device_map, device_mesh, tp_size = initialize_tensor_parallelism(
    tp_plan=config.base_model_tp_plan, device_mesh=ep_mesh
)
# EP+DS: clear device_map
device_map = None
log(f"   device_mesh={device_mesh}, tp_size={tp_size}")

# Step 6: Resolve checkpoints
log("6. Resolve checkpoints")
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
log(f"   Found {len(checkpoint_files)} files")

# Step 7: Get dtype
log("7. Dtype")
from transformers.modeling_utils import apply_patches, local_torch_dtype


config, dtype = _get_dtype(torch.bfloat16, checkpoint_files, config, sharded_metadata, None, True, None)
log(f"   dtype={dtype}")

# Step 8: Create model on META DEVICE (fast, like torchrun path)
log("8. Creating model on META DEVICE")
config2 = copy.deepcopy(config)
init_contexts = [
    local_torch_dtype(dtype, Qwen3MoeForCausalLM.__name__),
    init.no_tie_weights(),
    apply_patches(),
    # KEY: use meta device, same as the 'else' branch in get_init_context
    torch.device("meta"),
    init.meta_device_safe_creation_ops(),
]
with ContextManagers(init_contexts):
    model = Qwen3MoeForCausalLM(config2)
log(f"   Model created on meta! First param device: {next(model.parameters()).device}")
log(f"   First param shape: {next(model.parameters()).shape}")

# Step 9: distribute_model (adds EP hooks, does NOT touch weights)
log("9. distribute_model (EP hooks)")
model = distribute_model(model, config2.base_model_tp_plan, distributed_config, device_mesh, tp_size)
log("   distribute_model OK")
log(f"   tp_plan: {list(model.tp_plan.keys())[:5]}...")

# Step 10: STANDARD weight loading (NOT zero3 path)
log("10. Standard weight loading (NOT zero3)")
load_config = LoadStateDictConfig(
    pretrained_model_name_or_path="Qwen/Qwen3-30B-A3B",
    sharded_metadata=sharded_metadata,
    device_map=device_map,
    dtype=dtype,
    device_mesh=device_mesh,
)

# Manually do what _load_pretrained_model does, but force the standard path
from safetensors import safe_open
from transformers.core_model_loading import convert_and_load_state_dict_in_model


log("   Loading safetensors...")
all_pointer = set()
merged_state_dict = {}
for f in checkpoint_files:
    file_pointer = safe_open(f, framework="pt", device="cpu")
    all_pointer.add(file_pointer)
    for k in file_pointer.keys():
        merged_state_dict[k] = file_pointer.get_slice(k)
log(f"   Loaded {len(merged_state_dict)} keys from {len(checkpoint_files)} files")

# Key: use model.tp_plan (which returns ep_plan when EP enabled) for sharding
# but this is the standard loading path, NOT zero3
log("   convert_and_load_state_dict_in_model with tp_plan (EP plan)...")
try:
    loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
        model=model,
        state_dict=merged_state_dict,
        load_config=load_config,
        tp_plan=model.tp_plan,
        disk_offload_index=None,
    )
    log("   Loading OK!")
    log(f"   Missing keys: {len(loading_info.missing_keys)}")
    # Check first param is now real (not meta)
    first_param = next(model.parameters())
    log(f"   First param: device={first_param.device}, shape={first_param.shape}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

# Close file pointers
for p in all_pointer:
    p.__exit__(None, None, None)

# Step 11: Verify model
log("11. Verifying model")
try:
    expert_mod = model.model.layers[0].mlp.experts
    log(f"   experts type: {type(expert_mod).__name__}")
    log(f"   num_experts attr: {expert_mod.num_experts if hasattr(expert_mod, 'num_experts') else 'N/A'}")

    # Check a random expert weight
    for name, param in model.named_parameters():
        if "experts" in name:
            log(f"   Expert param '{name}': shape={param.shape}, device={param.device}")
            break
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

# Step 12: Forward pass
log("12. Forward pass")
try:
    model = model.to(f"cuda:{local_rank}")
    x = torch.randint(0, 1000, (1, 8), device=f"cuda:{local_rank}")
    with torch.no_grad():
        out = model(input_ids=x, use_cache=False)
    log(f"   Forward OK! logits shape={out.logits.shape}, nan={out.logits.isnan().any()}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
