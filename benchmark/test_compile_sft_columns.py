# ruff: noqa
"""Test: does having only input_ids (no labels) in the dataset cause slowdown?

A: Dataset with input_ids + labels + attention_mask (our fast test)
B: Dataset with input_ids ONLY (what _prepare_dataset + pack produces)

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=01:00:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    export TEST_MODE=$TEST_MODE && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_columns.py'
"""

import os

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


SEQ_LEN = 16384
MAX_STEPS = 15


def prepare_packed_data():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    ds = load_dataset("THUDM/LongAlign-10k", split="train")
    all_tokens = []
    for sample in ds:
        msgs = sample.get("messages", [])
        text = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
        if text:
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(toks)
        if len(all_tokens) > SEQ_LEN * 100:
            break
    n_chunks = min(50, len(all_tokens) // SEQ_LEN)
    chunks = [all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_chunks)]
    return chunks, tokenizer


def main():
    chunks, tokenizer = prepare_packed_data()
    mode = os.environ.get("TEST_MODE", "with_labels")

    if mode == "with_labels":
        # A: dataset with input_ids + labels + attention_mask
        data = {
            "input_ids": chunks,
            "labels": [c[:] for c in chunks],
            "attention_mask": [[1] * SEQ_LEN for _ in chunks],
        }
        print(f"[TEST A] Dataset with input_ids + labels + attention_mask ({len(chunks)} samples)", flush=True)
    else:
        # B: dataset with input_ids ONLY (like _prepare_dataset produces)
        data = {
            "input_ids": chunks,
        }
        print(f"[TEST B] Dataset with input_ids ONLY ({len(chunks)} samples)", flush=True)

    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)

    args = SFTConfig(
        output_dir=f"/tmp/sft_columns_{mode}",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=999,
        torch_compile=True,
        tf32=True,
        include_num_input_tokens_seen="no",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] {mode}: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
