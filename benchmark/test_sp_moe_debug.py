# ruff: noqa
"""
Debug: why does Ulysses SP hang on 30B MoE but work on 4B dense?
Mimics the accelerate DeepSpeed + SP flow step by step.
"""

import os
import time

import torch
import torch.distributed as dist


rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
SP_SIZE = int(os.environ.get("SP_SIZE", "2"))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-30B-A3B")
_start = time.time()


def log(msg):
    if rank == 0:
        print(f"[{time.time() - _start:6.1f}s] {msg}", flush=True)


# Step 1: PyTorch dist init
log("1. dist.init_process_group")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
log(f"   OK ws={world_size}")

# Step 2: DeepSpeed comm init (needed for SP)
log("2. deepspeed.comm.init_distributed")
from deepspeed import comm as ds_comm


ds_comm.init_distributed("nccl")
log(f"   OK ds_comm.is_initialized()={ds_comm.is_initialized()}")

# Step 3: Load model WITH EP
log(f"3. AutoModelForCausalLM.from_pretrained({MODEL}) with EP")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


USE_EP = os.environ.get("USE_EP", "0") == "1"
kwargs = {"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
if USE_EP:
    ep_mesh = dist.init_device_mesh("cuda", (world_size,))
    kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=True)
    kwargs["device_mesh"] = ep_mesh
    log(f"   EP enabled, mesh={ep_mesh}")
else:
    kwargs["device_map"] = {"": local_rank}
    log("   EP disabled, device_map=local")

model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
if USE_EP:
    log(f"   OK. experts.num_experts={model.model.layers[0].mlp.experts.num_experts}")
else:
    log(f"   OK. type={type(model).__name__}")

# Step 4: Ulysses attention registration (this internally calls mpu.initialize_sequence_parallel)
log("4. UlyssesSPAttentionHF.register_with_transformers")
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF


mpu_obj = UlyssesSPAttentionHF.register_with_transformers(
    model_name_or_path=model,
    sequence_parallel_size=SP_SIZE,
    seq_length=None,
    seq_length_is_variable=True,
    core_attn_implementation="flash_attention_2",
    micro_batch_size=1,
)
log("   OK")

# Step 5: Forward pass
log("5. forward pass")
x = torch.tensor([[1, 2, 3, 4]], device=f"cuda:{local_rank}")
try:
    with torch.no_grad():
        out = model(x, use_cache=False)
    log(f"   OK logits nan={out.logits.isnan().any()}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
