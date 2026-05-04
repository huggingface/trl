# ruff: noqa
"""Run mode A (no max_length) and mode B (max_length=16384) SEQUENTIALLY on the SAME nodes.

This eliminates node variability as a confounding factor.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=01:00:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) --main_process_port 29500 \
    benchmark/test_compile_ab_same_nodes.py'
"""

import os

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer
from trl.data_utils import maybe_convert_to_chatml, pack_dataset


SEQ_LEN = 16384
MAX_STEPS = 15


def prepare_data():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
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
    data_dict = {col: ds[col] for col in ds.column_names}
    simple_ds = Dataset.from_dict(data_dict)
    simple_ds.set_format("torch")
    return simple_ds, tokenizer


def run_test(simple_ds, tokenizer, max_length, label):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    kwargs = dict(
        output_dir=f"/tmp/ab_{label}",
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
    if max_length is not None:
        kwargs["max_length"] = max_length

    args = SFTConfig(**kwargs)
    trainer = SFTTrainer(model=model, args=args, train_dataset=simple_ds, processing_class=tokenizer)
    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] {label} (max_length={max_length}): {ms:.0f} ms/step", flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    return ms


def main():
    simple_ds, tokenizer = prepare_data()

    # Run A first, then B on the SAME nodes
    # Note: can't create two Trainers sequentially due to Accelerator lifecycle.
    # So just run the one specified by TEST_MODE
    mode = os.environ.get("TEST_MODE", "A")
    if mode == "A":
        run_test(simple_ds, tokenizer, max_length=None, label="A")
    elif mode == "B":
        run_test(simple_ds, tokenizer, max_length=16384, label="B")


if __name__ == "__main__":
    main()
