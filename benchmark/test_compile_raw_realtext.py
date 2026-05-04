# ruff: noqa
"""Test: Raw loop + FSDP2 + compile, but with REAL tokenized text instead of random tokens.

If slow (~9s): the data content itself causes compile slowdown (MoE routing patterns)
If fast (~3s): something in SFT's data collator / packing creates the issue

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_raw_realtext.py'
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer


SEQ_LEN = 16384
BATCH_SIZE = 1
N_WARMUP = 2
N_STEPS = 10


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World: {world_size} GPUs", flush=True)

    # Tokenize real text to get realistic token distributions
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    # Create a long realistic text by repeating a common pattern
    real_text = "The quick brown fox jumps over the lazy dog. " * 5000
    tokens = tokenizer(real_text, return_tensors="pt", max_length=SEQ_LEN, truncation=True, padding="max_length")
    real_input_ids = tokens["input_ids"].to(device)  # (1, SEQ_LEN)
    real_labels = real_input_ids.clone()

    # Also create random tokens for comparison
    random_input_ids = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
    random_labels = random_input_ids.clone()

    if rank == 0:
        print(
            f"Real input_ids shape: {real_input_ids.shape}, unique tokens: {real_input_ids.unique().numel()}",
            flush=True,
        )
        print(
            f"Random input_ids shape: {random_input_ids.shape}, unique tokens: {random_input_ids.unique().numel()}",
            flush=True,
        )

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

    # Test 1: Compiled + random tokens
    if rank == 0:
        print("\n--- Compiled + random tokens ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    apply_compile(model)
    apply_fsdp(model)
    t1 = benchmark(model, random_input_ids, random_labels, "Compiled+random")
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    # Test 2: Compiled + real tokens
    if rank == 0:
        print("\n--- Compiled + real tokens ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    apply_compile(model)
    apply_fsdp(model)
    t2 = benchmark(model, real_input_ids, real_labels, "Compiled+real_text")
    del model
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}", flush=True)
        print("SUMMARY:", flush=True)
        print(f"  Compiled + random:    {t1:.0f} ms/step", flush=True)
        print(f"  Compiled + real text: {t2:.0f} ms/step  ({t2 / t1:.2f}x)", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
