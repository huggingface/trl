# ruff: noqa
"""
Diagnostic v3: Step through from_pretrained manually.
Instead of monkey-patching, replicate the exact from_pretrained steps
one by one to find the exact hang point.
"""

import copy
import os
import sys
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

# Step 3: Import model class
log("3. Importing model class and config")
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.integrations.tensor_parallel import distribute_model, initialize_tensor_parallelism
from transformers.modeling_utils import _get_dtype, _get_resolved_checkpoint_files, _is_ds_init_called
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from transformers.utils.generic import ContextManagers


log("   OK")

# Step 4: Load config
log("4. Loading config")
config, model_kwargs = Qwen3MoeForCausalLM.config_class.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    return_unused_kwargs=True,
)
config._attn_implementation = "flash_attention_2"
distributed_config = DistributedConfig(enable_expert_parallel=True)
log(f"   config loaded: {config.model_type}, layers={config.num_hidden_layers}")

# Step 5: Create device mesh and initialize TP
log("5. initialize_tensor_parallelism")
ep_mesh = dist.init_device_mesh("cuda", (world_size,))
device_map, device_mesh, tp_size = initialize_tensor_parallelism(
    tp_plan=config.base_model_tp_plan, device_mesh=ep_mesh
)
log(f"   device_map={device_map}, device_mesh={device_mesh}, tp_size={tp_size}")

# EP + DS: clear device_map
log("5b. Clearing device_map for EP+DS")
device_map = None
log(f"   device_map={device_map}")

# Step 6: Resolve checkpoint files
log("6. Resolving checkpoint files")
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
log(f"   Found {len(checkpoint_files)} checkpoint files")

# Step 7: Get dtype
log("7. Getting dtype")
config, dtype = _get_dtype(torch.bfloat16, checkpoint_files, config, sharded_metadata, None, True, None)
log(f"   dtype={dtype}")

# Step 8: Get init context
log("8. get_init_context")
_has_ep = distributed_config is not None and getattr(distributed_config, "enable_expert_parallel", False)
log(f"   _has_ep={_has_ep}, is_ds_zero3={is_deepspeed_zero3_enabled()}, _is_ds_init_called={_is_ds_init_called}")
model_init_context = Qwen3MoeForCausalLM.get_init_context(dtype, False, _is_ds_init_called, None, distributed_config)
log(f"   context managers: {[type(c).__name__ for c in model_init_context]}")

# Step 9: Create model (THIS IS WHERE WE THINK IT HANGS)
log("9. Creating model (cls(config))")
log(f"   This creates Qwen3MoeForCausalLM with {config.num_hidden_layers} layers on CPU...")
config2 = copy.deepcopy(config)
try:
    with ContextManagers(model_init_context):
        log("   Inside context managers, calling cls(config)...")
        model = Qwen3MoeForCausalLM(config2)
        log(f"   Model created! Type: {type(model).__name__}")
        log(f"   First param: {next(model.parameters()).shape}, device={next(model.parameters()).device}")
except Exception as e:
    log(f"   FAILED during model creation: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()
    dist.destroy_process_group()
    sys.exit(1)

# Step 10: distribute_model (EP hooks)
log("10. distribute_model (EP hooks)")
try:
    model = distribute_model(model, config2.base_model_tp_plan, distributed_config, device_mesh, tp_size)
    log("   distribute_model OK")
except Exception as e:
    log(f"   FAILED during distribute_model: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()
    dist.destroy_process_group()
    sys.exit(1)

# Step 11: _load_pretrained_model
log("11. _load_pretrained_model")
try:
    from transformers.modeling_utils import LoadStateDictConfig

    load_config = LoadStateDictConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-30B-A3B",
        sharded_metadata=sharded_metadata,
        device_map=device_map,
        dtype=dtype,
        device_mesh=device_mesh,
    )
    log(f"   Calling _load_pretrained_model with {len(checkpoint_files)} files...")
    loading_info, disk_offload_index = Qwen3MoeForCausalLM._load_pretrained_model(
        model, None, checkpoint_files, load_config
    )
    log("   _load_pretrained_model OK")
except Exception as e:
    log(f"   FAILED during _load_pretrained_model: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
