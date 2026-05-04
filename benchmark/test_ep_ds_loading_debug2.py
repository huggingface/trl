# ruff: noqa
"""
Diagnostic v2: Fine-grained tracing inside from_pretrained to find exact hang.
Previous test showed hang happens during from_pretrained, BEFORE distribute_model
or _load_pretrained_model. So it hangs during model instantiation (cls(config)).

Hypothesis: Even with the 'pass' branch in get_init_context, something else in the
from_pretrained flow triggers zero.Init or a NCCL deadlock during model __init__.
"""

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

log("2. Setting DS env vars (simulating accelerate)")
os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"

from transformers.integrations.deepspeed import HfDeepSpeedConfig


dschf = HfDeepSpeedConfig(ds_config)

from transformers.integrations import is_deepspeed_zero3_enabled


log(f"   is_deepspeed_zero3_enabled() = {is_deepspeed_zero3_enabled()}")

# Step 3: Monkey-patch EVERYTHING inside from_pretrained to trace execution

import transformers.modeling_utils as mu


# Patch get_init_context to trace
_orig_get_init_context = mu.PreTrainedModel.get_init_context


@classmethod
def _traced_get_init_context(cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels, distributed_config=None):
    log(f"   >>> get_init_context(dtype={dtype}, is_quantized={is_quantized}, ds_init={_is_ds_init_called})")
    _has_ep = distributed_config is not None and getattr(distributed_config, "enable_expert_parallel", False)
    log(f"   >>> _has_ep={_has_ep}, is_ds_zero3={is_deepspeed_zero3_enabled()}")
    result = _orig_get_init_context.__func__(
        cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels, distributed_config
    )
    log(f"   >>> get_init_context returned {len(result)} context managers: {[type(c).__name__ for c in result]}")
    return result


mu.PreTrainedModel.get_init_context = _traced_get_init_context

# Patch the model __init__ (Qwen3MoeForCausalLM)
# We'll do this after config is loaded

# Patch distribute_model
from transformers.integrations import tensor_parallel as tp_mod


_orig_distribute = tp_mod.distribute_model


def _traced_distribute(model, tp_plan, distributed_config, device_mesh, tp_size):
    log("   >>> ENTERED distribute_model")
    result = _orig_distribute(model, tp_plan, distributed_config, device_mesh, tp_size)
    log("   >>> distribute_model completed")
    return result


tp_mod.distribute_model = _traced_distribute
mu.distribute_model = _traced_distribute

# Patch _load_pretrained_model
_orig_load = mu.PreTrainedModel._load_pretrained_model


@staticmethod
def _traced_load(model, state_dict, checkpoint_files, load_config, expected_keys=None):
    log(f"   >>> ENTERED _load_pretrained_model (files={len(checkpoint_files) if checkpoint_files else 0})")
    result = _orig_load(model, state_dict, checkpoint_files, load_config, expected_keys)
    log("   >>> _load_pretrained_model completed")
    return result


mu.PreTrainedModel._load_pretrained_model = _traced_load

# Patch zero.Init to trace if it gets called
try:
    import deepspeed

    _orig_zero_init = deepspeed.zero.Init

    class _TracedZeroInit(_orig_zero_init):
        def __init__(self, *args, **kwargs):
            log("   >>> WARNING: deepspeed.zero.Init() called! This should NOT happen with EP+DS")
            log("   >>> Stack trace:")
            import traceback

            if rank == 0:
                traceback.print_stack()
            super().__init__(*args, **kwargs)

    deepspeed.zero.Init = _TracedZeroInit
except Exception as e:
    log(f"   Failed to patch zero.Init: {e}")

# Patch the PreTrainedModel.post_init to trace
_orig_post_init = mu.PreTrainedModel.post_init


def _traced_post_init(self):
    log(f"   >>> post_init called (model type: {type(self).__name__})")
    # Check if zero3 init is active
    log(f"   >>> _is_ds_init_called={mu._is_ds_init_called}")
    _orig_post_init(self)
    log("   >>> post_init completed")


mu.PreTrainedModel.post_init = _traced_post_init

# Step 4: Load model with EP
log("3. Loading model with EP")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


ep_mesh = dist.init_device_mesh("cuda", (world_size,))
log(f"   device_mesh: {ep_mesh}")

try:
    log("   Calling from_pretrained...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B",
        dtype=torch.bfloat16,
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        device_mesh=ep_mesh,
        attn_implementation="flash_attention_2",
    )
    log("   from_pretrained OK!")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
