# ruff: noqa
"""Test if passing attention_mask changes compile performance.

The SFT trainer passes attention_mask (4D causal packed mask) while
our raw tests don't. This tests if the mask causes the slowdown.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_with_mask.py'
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM


SEQ_LEN = 16384
BATCH_SIZE = 1
N_WARMUP = 2
N_STEPS = 8


def benchmark(model, label, rank, world_size, device, use_mask=False):
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = input_ids.clone()

    kwargs = {"input_ids": input_ids, "labels": labels}
    if use_mask:
        # Create a 2D attention_mask (all ones = no padding, like packed sequences)
        kwargs["attention_mask"] = torch.ones(BATCH_SIZE, SEQ_LEN, device=device, dtype=torch.long)

    for _ in range(N_WARMUP):
        out = model(**kwargs)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        out = model(**kwargs)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms_per_step = (elapsed / N_STEPS) * 1000
    tps_gpu = (BATCH_SIZE * SEQ_LEN * N_STEPS) / elapsed

    if rank == 0:
        print(f"  {label}: {ms_per_step:.0f} ms/step, TPS/GPU={tps_gpu:.0f}", flush=True)
    return ms_per_step


def apply_compile(model):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    if getattr(model.config, "_experts_implementation", None) is None:
        if getattr(model.config, "num_local_experts", 0) > 0:
            model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)


def apply_fsdp(model):
    mp = MixedPrecisionPolicy()
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)


def load_model():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World: {world_size} GPUs, seq_len={SEQ_LEN}", flush=True)

    # Test 1: Compiled, no mask
    if rank == 0:
        print("\n--- Compiled, NO attention_mask ---", flush=True)
    m = load_model()
    apply_compile(m)
    apply_fsdp(m)
    t1 = benchmark(m, "Compiled no_mask", rank, world_size, device, use_mask=False)
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 2: Compiled, with mask
    if rank == 0:
        print("\n--- Compiled, WITH attention_mask ---", flush=True)
    m = load_model()
    apply_compile(m)
    apply_fsdp(m)
    t2 = benchmark(m, "Compiled with_mask", rank, world_size, device, use_mask=True)
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 3: Eager, no mask (baseline)
    if rank == 0:
        print("\n--- Eager, NO attention_mask ---", flush=True)
    m = load_model()
    apply_fsdp(m)
    t3 = benchmark(m, "Eager no_mask", rank, world_size, device, use_mask=False)
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 4: Eager, with mask
    if rank == 0:
        print("\n--- Eager, WITH attention_mask ---", flush=True)
    m = load_model()
    apply_fsdp(m)
    t4 = benchmark(m, "Eager with_mask", rank, world_size, device, use_mask=True)
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}", flush=True)
        print(f"SUMMARY ({world_size} GPUs):", flush=True)
        print(f"  Eager no_mask:      {t3:.0f} ms/step", flush=True)
        print(f"  Eager with_mask:    {t4:.0f} ms/step  ({t4 / t3:.2f}x)", flush=True)
        print(f"  Compiled no_mask:   {t1:.0f} ms/step  ({t1 / t3:.2f}x)", flush=True)
        print(f"  Compiled with_mask: {t2:.0f} ms/step  ({t2 / t3:.2f}x)", flush=True)
        print(f"  Compile effect no_mask:   {t3 / t1:.2f}x speedup", flush=True)
        print(f"  Compile effect with_mask: {t4 / t2:.2f}x speedup", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
