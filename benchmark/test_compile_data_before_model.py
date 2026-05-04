# ruff: noqa
"""Test: prepare data FIRST, then load model, then SFTTrainer with skip_prepare_dataset.

This matches the manual prep test flow but uses sft.py-style setup.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_data_before_model.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer
from trl.data_utils import maybe_convert_to_chatml, pack_dataset


SEQ_LEN = 16384
MAX_STEPS = 20


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    # === Step 1: Prepare data FIRST (no model in memory) ===
    ds = load_dataset("THUDM/LongAlign-10k", split="train")
    with PartialState().main_process_first():
        ds = ds.map(
            maybe_convert_to_chatml, remove_columns="conversations" if "conversations" in ds.column_names else None
        )

        def tokenize_fn(example):
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            return tokenizer(text, add_special_tokens=False)

        ds = ds.map(tokenize_fn, desc="Tokenizing")
        ds = ds.select_columns(["input_ids"])
        ds = pack_dataset(ds, SEQ_LEN, "wrapped")

    # Convert to simple format
    data_dict = {col: ds[col] for col in ds.column_names}
    simple_ds = Dataset.from_dict(data_dict)
    simple_ds.set_format("torch")
    del ds, data_dict
    print(f"[DATA] Prepared {len(simple_ds)} samples", flush=True)

    # === Step 2: NOW load model ===
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    # === Step 3: Create SFTTrainer with compile ===
    args = SFTConfig(
        output_dir="/tmp/data_before_model",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,
        tf32=True,
        include_num_input_tokens_seen=True,
        max_length=SEQ_LEN,
        packing=True,
        packing_strategy="wrapped",
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
        mfu = result.metrics.get("train_mfu", "N/A")
        print(
            f"[RESULT] Data-first + SFT compile: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
