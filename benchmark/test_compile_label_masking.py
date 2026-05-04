# ruff: noqa
"""Test: does -100 label masking cause the compile slowdown?

Experiment A: labels = input_ids (no masking) — known fast
Experiment B: labels with ~50% masked to -100 (simulating SFT prompt masking)

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_label_masking.py'
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

    # Prepare real tokenized data
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    ds = load_dataset("THUDM/LongAlign-10k", split="train")
    all_tokens = []
    for sample in ds:
        msgs = sample.get("messages", [])
        text = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
        if text:
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(toks)
        if len(all_tokens) > SEQ_LEN * 20:
            break

    input_ids = torch.tensor(all_tokens[:SEQ_LEN], dtype=torch.long, device=device).unsqueeze(0)

    # Labels A: no masking (labels = input_ids)
    labels_no_mask = input_ids.clone()

    # Labels B: ~50% masked with -100 (every other chunk of 512 tokens masked)
    labels_masked = input_ids.clone()
    for i in range(0, SEQ_LEN, 1024):
        labels_masked[0, i : i + 512] = -100

    n_masked = (labels_masked == -100).sum().item()
    if rank == 0:
        print(f"input_ids shape: {input_ids.shape}, unique: {input_ids.unique().numel()}", flush=True)
        print("labels_no_mask: 0 masked tokens", flush=True)
        print(f"labels_masked: {n_masked} masked tokens ({n_masked / SEQ_LEN * 100:.0f}%)", flush=True)

    def load_and_prepare():
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
        if getattr(model.config, "num_local_experts", 0) > 0:
            model.config._experts_implementation = "grouped_mm"
        for layer in model.model.layers:
            layer.compile(backend="inductor", fullgraph=True)
        mp = MixedPrecisionPolicy()
        for layer in model.model.layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
        return model

    def benchmark(model, ids, labels, label):
        for _ in range(N_WARMUP):
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
            for p in model.parameters():
                p.grad = None
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
            for p in model.parameters():
                p.grad = None
        torch.cuda.synchronize()
        ms = ((time.perf_counter() - t0) / N_STEPS) * 1000
        if rank == 0:
            print(f"  {label}: {ms:.0f} ms/step", flush=True)
        return ms

    # Experiment A: no label masking
    if rank == 0:
        print("\n--- A: Compiled + no label masking ---", flush=True)
    m = load_and_prepare()
    t_a = benchmark(m, input_ids, labels_no_mask, "no_mask")
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    # Experiment B: with label masking
    if rank == 0:
        print("\n--- B: Compiled + 50% label masking ---", flush=True)
    m = load_and_prepare()
    t_b = benchmark(m, input_ids, labels_masked, "50%_masked")
    del m
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print(f"\n{'=' * 60}", flush=True)
        print(f"LABEL MASKING TEST (compiled, {world_size} GPUs):", flush=True)
        print(f"  A (no masking):    {t_a:.0f} ms/step", flush=True)
        print(f"  B (50% masked):    {t_b:.0f} ms/step  ({t_b / t_a:.2f}x)", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
