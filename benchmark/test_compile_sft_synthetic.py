# ruff: noqa
"""Test: actual SFTTrainer with synthetic data (no packing).

If slow (~9s): issue is in SFTTrainer setup (not data)
If fast (~3s): issue is in packed data / data collator

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_synthetic.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


SEQ_LEN = 16384


def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16)

    # Create synthetic pre-tokenized dataset
    n_samples = 50
    data = {
        "input_ids": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "labels": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "attention_mask": [[1] * SEQ_LEN for _ in range(n_samples)],
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = SFTConfig(
        output_dir="/tmp/sft_synthetic_compile",
        max_steps=15,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,  # per-layer compile via SFTTrainer
        tf32=True,
        max_length=SEQ_LEN,
        packing=True,
        packing_strategy="wrapped",
        include_num_input_tokens_seen=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    ms = result.metrics["train_runtime"] / 15 * 1000
    if trainer.accelerator.is_main_process:
        mfu = result.metrics.get("train_mfu", "N/A")
        print(
            f"[RESULT] SFTTrainer+synthetic+compile: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
