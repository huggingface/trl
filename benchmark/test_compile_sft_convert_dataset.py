# ruff: noqa
"""Test: let _prepare_dataset run normally, then convert dataset to simple format before training.

If fast → the Arrow/HF Dataset format from _prepare_dataset causes the slowdown
If slow → something else in the pipeline

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_convert_dataset.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


SEQ_LEN = 16384
MAX_STEPS = 15


def main():
    # Step 1: Create an SFTTrainer that processes the dataset normally
    model_tmp = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    dataset = load_dataset("THUDM/LongAlign-10k", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args_tmp = SFTConfig(
        output_dir="/tmp/sft_convert_tmp",
        max_steps=0,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        torch_compile=False,  # don't compile the throwaway model
        max_length=SEQ_LEN,
        packing=True,
        packing_strategy="wrapped",
    )

    # This processes the dataset via _prepare_dataset
    trainer_tmp = SFTTrainer(
        model=model_tmp,
        args=args_tmp,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Extract the processed dataset
    processed_ds = trainer_tmp.train_dataset
    print(f"[TEST] Processed dataset: {len(processed_ds)} samples, columns={processed_ds.column_names}", flush=True)

    # Convert to simple dict-based dataset (breaks Arrow backing)
    data_dict = {col: processed_ds[col] for col in processed_ds.column_names}
    simple_ds = Dataset.from_dict(data_dict)
    simple_ds.set_format("torch")
    print(
        f"[TEST] Converted dataset: {len(simple_ds)} samples, first input_ids len={len(simple_ds[0]['input_ids'])}",
        flush=True,
    )

    del model_tmp, trainer_tmp
    torch.cuda.empty_cache()

    # Step 2: Now create a NEW SFTTrainer with the converted dataset + compile
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)

    args = SFTConfig(
        output_dir="/tmp/sft_convert",
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
        train_dataset=simple_ds,
        processing_class=tokenizer,
    )

    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] SFT + converted _prepare_dataset output + compile: {ms:.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
