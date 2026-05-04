# ruff: noqa
"""Test: SFTTrainer with real LongAlign-10k data + packing + compile, called from Python.

Same as the benchmark but invoked directly (not via trl/scripts/sft.py).
This isolates whether the issue is in sft.py's model loading or in the data.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_realdata_direct.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


SEQ_LEN = 16384


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    dataset = load_dataset("THUDM/LongAlign-10k")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = SFTConfig(
        output_dir="/tmp/sft_realdata_direct",
        max_steps=20,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,
        tf32=True,
        max_length=SEQ_LEN,
        packing=True,
        packing_strategy="wrapped",
        include_num_input_tokens_seen=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    result = trainer.train()
    ms = result.metrics["train_runtime"] / 20 * 1000
    if trainer.accelerator.is_main_process:
        print(
            f"[RESULT] SFTTrainer+realdata+packed+compile: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
            flush=True,
        )
        print(f"[RESULT] MFU={result.metrics.get('train_mfu', 'N/A')}", flush=True)


if __name__ == "__main__":
    main()
