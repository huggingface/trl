# ruff: noqa
"""Test: SFTTrainer with real data but monkey-patch compute_loss to skip logits postprocessing.

This isolates whether the slowdown is in SFT's compute_loss override or elsewhere.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_nooverride.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from trl import SFTConfig, SFTTrainer


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    dataset = load_dataset("THUDM/LongAlign-10k", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = SFTConfig(
        output_dir="/tmp/sft_nooverride",
        max_steps=15,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,
        tf32=True,
        max_length=16384,
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

    # Monkey-patch: replace SFT's compute_loss with HF Trainer's default
    # This skips entropy/accuracy/token counting but keeps everything else (data, FSDP, compile)
    trainer.compute_loss = Trainer.compute_loss.__get__(trainer, SFTTrainer)

    result = trainer.train()
    ms = result.metrics["train_runtime"] / 15 * 1000
    if trainer.accelerator.is_main_process:
        print(
            f"[RESULT] SFT+realdata+packed+compile (no compute_loss override): {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
