# ruff: noqa
"""Test if model.to('meta') after compile invalidates compiled graphs.

This simulates what accelerate does with cpu_ram_efficient_loading=True:
1. Load model on CPU
2. Compile layers
3. Move to meta device
4. Apply FSDP
5. Load state dict back

Run: srun --partition=hopper-prod --nodes=1 --gres=gpu:h100:8 --ntasks-per-node=1 --exclusive --time=00:30:00 --qos=normal \
  bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
  torchrun --nproc_per_node=8 benchmark/test_compile_meta_device.py'
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
N_STEPS = 5


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def benchmark(model, label, rank, world_size):
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
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
        print(f"  {label}: {ms_per_step:.0f} ms/step, TPS/GPU={tps_gpu:.0f}")
    return ms_per_step


def main():
    rank, world_size, local_rank = setup()
    mp_policy = MixedPrecisionPolicy()

    if rank == 0:
        print(f"GPUs: {world_size}, seq_len={SEQ_LEN}")

    # === Test 1: Compile then FSDP (direct, no meta device) ===
    if rank == 0:
        print("\n--- Test 1: Compile → FSDP (direct, our test script approach) ---")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)

    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

    t1 = benchmark(model, "Compile→FSDP (direct)", rank, world_size)
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    # === Test 2: Compile then meta then FSDP (accelerate flow) ===
    if rank == 0:
        print("\n--- Test 2: Compile → meta → FSDP (accelerate flow) ---")
    model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model2.train()
    model2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    sd = model2.state_dict()  # save state dict before meta

    for layer in model2.model.layers:
        layer.compile(backend="inductor", fullgraph=True)

    # Simulate accelerate's cpu_ram_efficient_loading
    model2 = model2.to(torch.device("meta"))
    if hasattr(model2, "tie_weights"):
        model2.tie_weights()

    for layer in model2.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model2, mp_policy=mp_policy)

    # Load state dict back (like accelerate does)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    set_model_state_dict(model2, sd, options=StateDictOptions(strict=False))

    t2 = benchmark(model2, "Compile→meta→FSDP", rank, world_size)
    del model2
    torch.cuda.empty_cache()
    dist.barrier()

    # === Test 3: FSDP only, no compile (baseline) ===
    if rank == 0:
        print("\n--- Test 3: FSDP only (no compile, baseline) ---")
    model3 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model3.train()
    model3.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    for layer in model3.model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model3, mp_policy=mp_policy)

    t3 = benchmark(model3, "FSDP only (eager)", rank, world_size)
    del model3
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"SUMMARY ({world_size} GPUs):")
        print(f"  FSDP only (eager):          {t3:.0f} ms/step")
        print(f"  Compile→FSDP (direct):      {t1:.0f} ms/step  ({t1 / t3:.2f}x vs eager)")
        print(f"  Compile→meta→FSDP (accel):  {t2:.0f} ms/step  ({t2 / t3:.2f}x vs eager)")
        print(f"{'=' * 60}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
