# ruff: noqa
"""Test if gradient checkpointing is what makes compile slow in SFT trainer.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_grad_ckpt.py'
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


def benchmark(model, label, rank, world_size, device):
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = input_ids.clone()

    for _ in range(N_WARMUP):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        out = model(input_ids=input_ids, labels=labels)
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


def load_model(grad_ckpt=True):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    if grad_ckpt:
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

    # === Test 1: Compiled + FSDP + grad ckpt ===
    if rank == 0:
        print("\n--- Test 1: Compiled + FSDP2 + grad_ckpt ---", flush=True)
    model = load_model(grad_ckpt=True)
    apply_compile(model)
    apply_fsdp(model)
    t1 = benchmark(model, "Compiled + grad_ckpt", rank, world_size, device)
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    # === Test 2: Compiled + FSDP, NO grad ckpt ===
    if rank == 0:
        print("\n--- Test 2: Compiled + FSDP2, NO grad_ckpt ---", flush=True)
    model2 = load_model(grad_ckpt=False)
    apply_compile(model2)
    apply_fsdp(model2)
    t2 = benchmark(model2, "Compiled, no grad_ckpt", rank, world_size, device)
    del model2
    torch.cuda.empty_cache()
    dist.barrier()

    # === Test 3: Eager + FSDP + grad ckpt (baseline) ===
    if rank == 0:
        print("\n--- Test 3: Eager + FSDP2 + grad_ckpt ---", flush=True)
    model3 = load_model(grad_ckpt=True)
    apply_fsdp(model3)
    t3 = benchmark(model3, "Eager + grad_ckpt", rank, world_size, device)
    del model3
    torch.cuda.empty_cache()
    dist.barrier()

    # === Test 4: Eager + FSDP, NO grad ckpt ===
    if rank == 0:
        print("\n--- Test 4: Eager + FSDP2, NO grad_ckpt ---", flush=True)
    model4 = load_model(grad_ckpt=False)
    apply_fsdp(model4)
    t4 = benchmark(model4, "Eager, no grad_ckpt", rank, world_size, device)
    del model4
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}", flush=True)
        print(f"SUMMARY ({world_size} GPUs):", flush=True)
        print(f"  Eager + grad_ckpt:      {t3:.0f} ms/step (baseline)", flush=True)
        print(f"  Eager, no grad_ckpt:    {t4:.0f} ms/step  ({t4 / t3:.2f}x)", flush=True)
        print(f"  Compiled + grad_ckpt:   {t1:.0f} ms/step  ({t1 / t3:.2f}x)", flush=True)
        print(f"  Compiled, no grad_ckpt: {t2:.0f} ms/step  ({t2 / t3:.2f}x)", flush=True)
        compile_speedup_gc = t3 / t1
        compile_speedup_no_gc = t4 / t2
        print(f"  Compile speedup WITH grad_ckpt:    {compile_speedup_gc:.2f}x", flush=True)
        print(f"  Compile speedup WITHOUT grad_ckpt: {compile_speedup_no_gc:.2f}x", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
