# ruff: noqa
"""Incremental test: start from fast raw path, add SFT components one by one.

Raw FSDP2 + compile = 3s/step (fast). SFT trainer = 9s/step (slow).
This script adds components incrementally to find what causes the slowdown.

Test A: Raw loop (model + FSDP + compile + manual fwd/bwd) — BASELINE
Test B: HF Trainer loop (model + FSDP + compile + Trainer.train()) — adds training loop
Test C: SFT Trainer loop (model + FSDP + compile + SFTTrainer.train()) — adds SFT specifics

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=01:00:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) && export MASTER_PORT=29500 && \
    torchrun --nproc_per_node=8 --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    benchmark/test_compile_incremental.py'
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from datasets import Dataset
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


SEQ_LEN = 16384
BATCH_SIZE = 1
N_WARMUP = 2
N_STEPS = 10


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


def test_a_raw_loop(rank, world_size, device):
    """Test A: Raw training loop with manual fwd/bwd."""
    if rank == 0:
        print("\n=== Test A: Raw loop + FSDP2 + compile ===", flush=True)

    model = load_model()
    apply_compile(model)
    apply_fsdp(model)

    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=device)
    labels = input_ids.clone()

    # Warmup
    for _ in range(N_WARMUP):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = (elapsed / N_STEPS) * 1000

    if rank == 0:
        print(f"  Test A (raw loop): {ms:.0f} ms/step", flush=True)

    del model
    torch.cuda.empty_cache()
    dist.barrier()
    return ms


def test_b_hf_trainer(rank, world_size, device):
    """Test B: HF Trainer training loop with synthetic data."""
    if rank == 0:
        print("\n=== Test B: HF Trainer + FSDP2 + compile ===", flush=True)

    model = load_model()
    apply_compile(model)

    # Create synthetic dataset
    n_samples = 50
    data = {
        "input_ids": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "labels": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = TrainingArguments(
        output_dir="/tmp/test_b",
        max_steps=N_WARMUP + N_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        logging_steps=999,  # suppress logging
        torch_compile=False,  # already compiled per-layer
        tf32=True,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train — first N_WARMUP steps are warmup, then measure
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_steps = N_WARMUP + N_STEPS
    ms = (elapsed / total_steps) * 1000

    if rank == 0:
        print(f"  Test B (HF Trainer): {ms:.0f} ms/step (avg over {total_steps} steps incl warmup)", flush=True)

    del model, trainer
    torch.cuda.empty_cache()
    dist.barrier()
    return ms


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"World: {world_size} GPUs, seq_len={SEQ_LEN}", flush=True)

    t_a = test_a_raw_loop(rank, world_size, device)
    t_b = test_b_hf_trainer(rank, world_size, device)

    if rank == 0:
        print(f"\n{'=' * 60}", flush=True)
        print(f"SUMMARY ({world_size} GPUs, compiled):", flush=True)
        print(f"  Test A (raw loop):   {t_a:.0f} ms/step", flush=True)
        print(f"  Test B (HF Trainer): {t_b:.0f} ms/step  ({t_b / t_a:.2f}x)", flush=True)
        print(f"{'=' * 60}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
