# ruff: noqa
"""
Diagnostic v7: Test the fix end-to-end using the real from_pretrained path.
Changes applied to modeling_utils.py:
1. get_init_context: EP+DS uses meta device (not 'pass'/real tensors)
2. _load_pretrained_model: EP+DS skips zero3 loading, uses standard path
3. _move_missing_keys_from_meta_to_device: EP+DS uses standard move path
4. _initialize_missing_keys: EP+DS uses standard init path
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


# Step 1: Init
log("1. init dist")
dist.init_process_group("nccl")
torch.cuda.set_device(local_rank)
log(f"   ws={world_size}")

# Step 2: Set DS env (simulating accelerate)
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

# Step 3: Load model using the real from_pretrained path
log("2. from_pretrained with EP+DS fix")
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


ep_mesh = dist.init_device_mesh("cuda", (world_size,))
try:
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
    dist.destroy_process_group()
    import sys

    sys.exit(1)

# Step 4: Verify model
log("3. Verifying model")
expert_mod = model.model.layers[0].mlp.experts
log(f"   experts type: {type(expert_mod).__name__}")
for name, param in model.named_parameters():
    if "experts" in name:
        log(f"   Expert '{name}': shape={param.shape}, device={param.device}")
        break

meta_count = sum(1 for p in model.parameters() if p.device.type == "meta")
meta_buf_count = sum(1 for b in model.buffers() if b.device.type == "meta")
log(f"   Meta params: {meta_count}, meta buffers: {meta_buf_count}")

# Step 5: Move to GPU and forward
log("4. Forward pass")
try:
    model = model.to(f"cuda:{local_rank}")
    x = torch.randint(0, 1000, (1, 8), device=f"cuda:{local_rank}")
    with torch.no_grad():
        out = model(input_ids=x, use_cache=False)
    log(f"   Forward OK! logits={out.logits.shape}, nan={out.logits.isnan().any()}")
    log(f"   logits stats: max={out.logits.max().item():.3f}, min={out.logits.min().item():.3f}")
except Exception as e:
    log(f"   FAILED: {e}")
    import traceback

    if rank == 0:
        traceback.print_exc()

# Step 6: DeepSpeed initialize
log("5. DeepSpeed initialize")
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
    log(f"   Forward: loss={out.loss.item():.4f}")
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
