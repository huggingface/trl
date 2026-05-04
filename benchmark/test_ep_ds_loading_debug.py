# ruff: noqa
"""
Diagnostic: reproduce the EP+DS hang with accelerate-style DS env.
Hypothesis: _load_pretrained_model goes to zero3 loading path
(GatheredParameters) even though model was created without zero.Init.

Test plan:
1. Set DS env vars like accelerate does (making is_deepspeed_zero3_enabled() return True)
2. Load model with EP + distributed_config
3. Instrument with prints to find the exact hang point
"""

import json
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

# Step 2: Set DS env vars LIKE accelerate does (this is what makes it hang)
# accelerate sets these when configured with DeepSpeed
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
ds_config_path = f"/tmp/ds_config_debug_rank{rank}.json"
with open(ds_config_path, "w") as f:
    json.dump(ds_config, f)

log("2. Setting DS env vars (simulating accelerate)")
os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = ds_config_path

# Now set the HF deepspeed config so is_deepspeed_zero3_enabled() returns True
from transformers.integrations.deepspeed import HfDeepSpeedConfig


dschf = HfDeepSpeedConfig(ds_config)
log("   DS env set. Checking is_deepspeed_zero3_enabled()...")

from transformers.integrations import is_deepspeed_zero3_enabled


log(f"   is_deepspeed_zero3_enabled() = {is_deepspeed_zero3_enabled()}")

# Step 3: Monkey-patch critical functions to trace the hang
import transformers.integrations.deepspeed as ds_integration
import transformers.modeling_utils as mu


# Patch _load_state_dict_into_zero3_model to add tracing
_orig_zero3_load = ds_integration._load_state_dict_into_zero3_model


def _traced_zero3_load(model_to_load, state_dict, load_config=None):
    log(f"   >>> ENTERED _load_state_dict_into_zero3_model (state_dict keys: {len(state_dict)})")
    log(f"   >>> First param ds_status: {next(model_to_load.parameters()).data.device}")
    # Check if params are zero3-partitioned
    first_param = next(model_to_load.parameters())
    log(f"   >>> First param shape: {first_param.shape}, dtype: {first_param.dtype}")
    has_ds_id = hasattr(first_param, "ds_id")
    has_ds_shape = hasattr(first_param, "ds_shape")
    log(f"   >>> Has ds_id: {has_ds_id}, has ds_shape: {has_ds_shape}")
    if has_ds_id:
        log(f"   >>> ds_id: {first_param.ds_id}, ds_shape: {first_param.ds_shape}, ds_status: {first_param.ds_status}")
    else:
        log("   >>> WARNING: params NOT zero3-partitioned but using zero3 loading path!")
        log("   >>> This is likely the bug — GatheredParameters on non-partitioned params will hang")

    # Try calling the original to see where exactly it hangs
    log("   >>> Calling original _load_state_dict_into_zero3_model...")
    result = _orig_zero3_load(model_to_load, state_dict, load_config)
    log("   >>> _load_state_dict_into_zero3_model completed OK")
    return result


ds_integration._load_state_dict_into_zero3_model = _traced_zero3_load
# Also patch at the import site in modeling_utils
mu._load_state_dict_into_zero3_model = _traced_zero3_load

# Patch distribute_model to trace
from transformers.integrations import tensor_parallel as tp_mod


_orig_distribute = tp_mod.distribute_model


def _traced_distribute(model, tp_plan, distributed_config, device_mesh, tp_size):
    log(f"   >>> ENTERED distribute_model (tp_size={tp_size})")
    log(f"   >>> tp_plan keys: {list(tp_plan.keys()) if tp_plan else None}")
    result = _orig_distribute(model, tp_plan, distributed_config, device_mesh, tp_size)
    log("   >>> distribute_model completed OK")
    return result


tp_mod.distribute_model = _traced_distribute
# Also need to patch the reference in modeling_utils since it imported the function directly
mu.distribute_model = _traced_distribute

# Step 4: Load model with EP
log("3. Loading model with EP (this should reproduce the hang)")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


ep_mesh = dist.init_device_mesh("cuda", (world_size,))
log(f"   device_mesh created: {ep_mesh}")

try:
    log("   Calling from_pretrained...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        device_mesh=ep_mesh,
        attn_implementation="flash_attention_2",
    )
    log("   from_pretrained completed OK!")
    log(f"   Model experts: {model.model.layers[0].mlp.experts.num_experts}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
