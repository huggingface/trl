# ruff: noqa
"""Test: load model FIRST, then prepare data, then SFTTrainer with skip_prepare_dataset.

Same as data_before_model but reversed order — model loads first.
If this is slow, the model being in memory during data prep causes the issue.

Run (2 nodes):
  srun ... benchmark/test_compile_model_before_data.py
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

    # === Step 1: Load model FIRST ===
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    print("[MODEL] Loaded", flush=True)

    # === Step 2: THEN prepare data (model in memory) ===
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
    del ds, data_dict
    print(f"[DATA] Prepared {len(simple_ds)} samples", flush=True)

    # === Step 3: Create SFTTrainer with compile ===
    args = SFTConfig(
        output_dir="/tmp/model_before_data",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,
        tf32=True,
        include_num_input_tokens_seen=True,
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
        print(
            f"[RESULT] Model-first + SFT compile: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
