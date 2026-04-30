# ruff: noqa: T201
"""
Full SP+EP test mimicking the accelerate DeepSpeed path.
Key insight: deepspeed.zero.Init() conflicts with EP's distribute_model().
Test: load with EP first (no zero.Init), then let DeepSpeed wrap later.
"""

import os
import time

import torch
import torch.distributed as dist


rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
SP_SIZE = int(os.environ.get("SP_SIZE", "2"))
_start = time.time()


def log(msg):
    if rank == 0:
        print(f"[{time.time() - _start:6.1f}s] {msg}", flush=True)


# Step 1: Init both PyTorch and DeepSpeed comm
log("1. init dist + ds_comm")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
from deepspeed import comm as ds_comm


ds_comm.init_distributed("nccl")
log(f"   OK ws={world_size} ds_comm={ds_comm.is_initialized()}")

# Step 2: Load model with EP (NO deepspeed.zero.Init)
log("2. from_pretrained with EP")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


ep_mesh = dist.init_device_mesh("cuda", (world_size,))
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-30B-A3B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    device_mesh=ep_mesh,
    attn_implementation="flash_attention_2",
)
log(f"   OK experts.num_experts={model.model.layers[0].mlp.experts.num_experts}")

# Step 3: Register SP attention
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

# Step 4: Wrap with DeepSpeed engine (ZeRO-3)
log("4. deepspeed.initialize (ZeRO-3)")
import deepspeed


ds_config = {
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "seq_parallel_communication_data_type": "bf16",
}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
    mpu=mpu,
)
log("   OK")

# Step 5: Forward pass
log("5. forward pass")
# SP requires seq_len divisible by SP_SIZE
seq_len = 4 * SP_SIZE
x = torch.randint(0, 1000, (1, seq_len), device=f"cuda:{local_rank}")
labels = x.clone()
try:
    out = model_engine(input_ids=x, labels=labels, use_cache=False)
    log(f"   OK loss={out.loss.item():.4f} logits_nan={out.logits.isnan().any()}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

# Step 6: Backward
log("6. backward")
try:
    model_engine.backward(out.loss)
    model_engine.step()
    log("   OK")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

log("Done.")
dist.destroy_process_group()
