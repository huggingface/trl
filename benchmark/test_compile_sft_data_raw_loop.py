# ruff: noqa
"""Use SFT's exact data pipeline but run with raw loop (no Trainer).

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_data_raw_loop.py'
"""

import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


SEQ_LEN = 16384
N_WARMUP = 2
N_STEPS = 10


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    dataset = load_dataset("THUDM/LongAlign-10k", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = SFTConfig(
        output_dir="/tmp/sft_data_raw",
        max_steps=N_WARMUP + N_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=999,
        torch_compile=True,
        tf32=True,
        max_length=SEQ_LEN,
        packing=True,
        packing_strategy="wrapped",
        include_num_input_tokens_seen="no",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    model = trainer.model
    dataloader = trainer.get_train_dataloader()
    rank = trainer.accelerator.process_index

    # Collect all batches first, move to GPU
    all_batches = []
    for batch in dataloader:
        all_batches.append(batch)
        if len(all_batches) >= N_WARMUP + N_STEPS:
            break

    # Move to GPU using trainer's prepare_inputs
    cuda_batches = []
    for b in all_batches:
        cuda_batches.append(trainer._prepare_inputs(b))

    if rank == 0:
        b = cuda_batches[0]
        details = {k: (tuple(v.shape), str(v.dtype), str(v.device)) for k, v in b.items()}
        print(f"[BATCH] {details}", flush=True)
        print("[TEST] Running raw loop with SFT data + compiled model", flush=True)

    # Warmup
    for i in range(N_WARMUP):
        out = model(**cuda_batches[i])
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_STEPS):
        out = model(**cuda_batches[N_WARMUP + i])
        out.loss.backward()
        for p in model.parameters():
            p.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = (elapsed / N_STEPS) * 1000

    if rank == 0:
        print(f"[RESULT] SFT data + raw loop + compiled: {ms:.0f} ms/step", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
