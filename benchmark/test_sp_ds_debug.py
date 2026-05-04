# ruff: noqa
"""
Debug: add DeepSpeed ZeRO-3 engine init to the working SP+EP flow.
This mimics what accelerator.prepare() does.
"""

import os
import time

import torch
import torch.distributed as dist


rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
SP_SIZE = int(os.environ.get("SP_SIZE", "2"))
USE_EP = os.environ.get("USE_EP", "0") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
_start = time.time()


def log(msg):
    if rank == 0:
        print(f"[{time.time() - _start:6.1f}s] {msg}", flush=True)


log("1. dist + ds_comm init")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
from deepspeed import comm as ds_comm


ds_comm.init_distributed("nccl")
log(f"   OK ws={world_size}")

log(f"2. from_pretrained({MODEL}) EP={USE_EP}")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


kwargs = {"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
if USE_EP:
    ep_mesh = dist.init_device_mesh("cuda", (world_size,))
    kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=True)
    kwargs["device_mesh"] = ep_mesh
else:
    kwargs["device_map"] = {"": local_rank}
model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
log("   OK")

log(f"3. UlyssesSPAttentionHF.register_with_transformers(sp={SP_SIZE})")
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF


mpu = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model,
    sequence_parallel_size=SP_SIZE,
    seq_length=None,
    seq_length_is_variable=True,
    core_attn_implementation="flash_attention_2",
    micro_batch_size=1,
)
log("   OK")

log("4. DeepSpeed ZeRO-3 engine init")
import deepspeed


ds_config = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "train_batch_size": world_size // SP_SIZE,  # DP size
    "zero_optimization": {"stage": 3},
    "bf16": {"enabled": True},
    "steps_per_print": 999999,
}
if mpu is not None:
    ds_config["train_batch_size"] = world_size // SP_SIZE
engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    mpu=mpu,
)
log("   OK")

log("5. forward pass")
x = torch.tensor([[1, 2, 3, 4]], device=f"cuda:{local_rank}")
try:
    with torch.no_grad():
        out = engine(x, use_cache=False)
    log(f"   OK logits nan={out.logits.isnan().any()}")
except Exception as e:
    log(f"   FAILED: {e}")

log("Done.")
dist.destroy_process_group()
