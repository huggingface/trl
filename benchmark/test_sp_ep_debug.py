# ruff: noqa
"""Debug: when does _SEQUENCE_PARALLEL_GROUP get initialized?"""

import os
import time

import torch
import torch.distributed as dist


rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
_start = time.time()


def log(msg):
    if rank == 0:
        print(f"[{time.time() - _start:6.1f}s] {msg}", flush=True)


def check_sp():
    import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu

    val = mpu._SEQUENCE_PARALLEL_GROUP
    return f"_SEQUENCE_PARALLEL_GROUP={'SET' if val is not None else 'None'}"


log("Step 1: init_process_group")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
log(f"  OK. world_size={dist.get_world_size()}, {check_sp()}")

log("Step 2: init_device_mesh")
ep_mesh = dist.init_device_mesh("cuda", (dist.get_world_size(),))
log(f"  OK. {check_sp()}")

log("Step 3: import transformers")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


log(f"  OK. {check_sp()}")

log("Step 4: from_pretrained WITHOUT EP (baseline)")
model_no_ep = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    device_map={"": local_rank},
    attn_implementation="flash_attention_2",
)
log(f"  OK. {check_sp()}")
del model_no_ep
torch.cuda.empty_cache()

log("Step 5: from_pretrained WITH EP")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    device_mesh=ep_mesh,
    attn_implementation="flash_attention_2",
)
log(f"  OK. {check_sp()}")

log(f"Step 6: dist.is_initialized()={dist.is_initialized()}")
log(f"  default PG: {dist.distributed_c10d._get_default_group() if dist.is_initialized() else 'N/A'}")

import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu


try:
    mpu.initialize_sequence_parallel(sequence_parallel_size=2)
    log("  SP init OK!")
except Exception as e:
    log(f"  FAILED: {type(e).__name__}: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()
    # Try re-init PG and retry
    log("  Retrying after checking dist state...")
    log(f"  dist.is_initialized()={dist.is_initialized()}")
    if not dist.is_initialized():
        log("  PG was destroyed! Re-initializing...")
        dist.init_process_group("nccl")
        mpu._SEQUENCE_PARALLEL_GROUP = None
        mpu._SEQUENCE_DATA_PARALLEL_GROUP = None
        mpu.initialize_sequence_parallel(sequence_parallel_size=2)
        log("  SP init OK after re-init!")

log("Done.")
dist.destroy_process_group()
