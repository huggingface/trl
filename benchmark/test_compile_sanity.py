# ruff: noqa
"""Sanity check: raw loop with REAL data from LongAlign-10k, tokenized + packed to 16384.
Tests compiled vs eager on the SAME data.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_sanity.py'
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQ_LEN = 16384
N_WARMUP = 2
N_STEPS = 10


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Prepare REAL data: tokenize LongAlign-10k samples, concat into 16384-length chunks
    if rank == 0:
        print("Preparing real tokenized data...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    ds = load_dataset("THUDM/LongAlign-10k", split="train")

    # Tokenize and pack: concatenate all tokens, split into SEQ_LEN chunks
    all_tokens = []
    for sample in ds:
        msgs = sample.get("messages", [])
        text = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
        if text:
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(tokens)
        if len(all_tokens) > SEQ_LEN * 20:
            break

    # Create packed batches
    packed_ids = torch.tensor(all_tokens[:SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)
    packed_labels = packed_ids.clone()

    if rank == 0:
        print(f"Packed input shape: {packed_ids.shape}, unique tokens: {packed_ids.unique().numel()}", flush=True)

    def load_model():
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

    def apply_compile(model):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        if getattr(model.config, "num_local_experts", 0) > 0:
            model.config._experts_implementation = "grouped_mm"
        for layer in model.model.layers:
            layer.compile(backend="inductor", fullgraph=True)

    def apply_fsdp(model):
        mp = MixedPrecisionPolicy()
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)

    def benchmark(model, input_ids, labels, label):
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
        ms = ((time.perf_counter() - t0) / N_STEPS) * 1000
        if rank == 0:
            print(f"  {label}: {ms:.0f} ms/step", flush=True)
        return ms

    # Test 1: EAGER + real data
    if rank == 0:
        print("\n--- EAGER + real packed data ---", flush=True)
    m = load_model()
    apply_fsdp(m)
    t_eager = benchmark(m, packed_ids, packed_labels, "Eager+real")
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 2: COMPILED + real data
    if rank == 0:
        print("\n--- COMPILED + real packed data ---", flush=True)
    m = load_model()
    apply_compile(m)
    apply_fsdp(m)
    t_compiled = benchmark(m, packed_ids, packed_labels, "Compiled+real")
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 3: COMPILED + random data (control)
    if rank == 0:
        print("\n--- COMPILED + random data (control) ---", flush=True)
    m = load_model()
    apply_compile(m)
    apply_fsdp(m)
    random_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN), device=device)
    t_rand = benchmark(m, random_ids, random_ids.clone(), "Compiled+random")
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}", flush=True)
        print(f"SANITY CHECK ({world_size} GPUs, real tokenized data):", flush=True)
        print(f"  Eager + real data:     {t_eager:.0f} ms/step", flush=True)
        print(f"  Compiled + real data:  {t_compiled:.0f} ms/step  ({t_compiled / t_eager:.2f}x vs eager)", flush=True)
        print(f"  Compiled + random:     {t_rand:.0f} ms/step  ({t_rand / t_eager:.2f}x vs eager)", flush=True)
        if t_compiled < t_eager:
            print(f"  ✓ Compile is {t_eager / t_compiled:.2f}x FASTER than eager with real data", flush=True)
        else:
            print(f"  ✗ Compile is {t_compiled / t_eager:.2f}x SLOWER than eager with real data", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
