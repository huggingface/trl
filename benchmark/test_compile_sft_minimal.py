# ruff: noqa
"""Minimal SFT trainer reproduce: uses SFTTrainer with synthetic data.

If this is slow (~9s/step compiled), the issue is in SFTTrainer/HF Trainer.
If fast (~3s/step), the issue is in the data pipeline.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_minimal.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


def main():
    # Create synthetic dataset (fixed-size packed sequences)
    seq_len = 16384
    n_samples = 100
    data = {
        "input_ids": [list(range(1000, 1000 + seq_len)) for _ in range(n_samples)],
    }
    dataset = Dataset.from_dict(data)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = SFTConfig(
        output_dir="/tmp/sft_compile_test",
        max_steps=20,
        logging_steps=5,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        max_length=seq_len,
        packing=True,
        packing_strategy="wrapped",
        include_num_input_tokens_seen=True,
        attn_implementation="sdpa",
        torch_compile=True,
        tf32=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # The SFTTrainer will do per-layer compile before super().__init__
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.accelerator.print("Done.")


if __name__ == "__main__":
    main()
