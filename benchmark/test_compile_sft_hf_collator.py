# ruff: noqa
"""Test: SFTTrainer + compile but with HF's DataCollatorWithPadding instead of SFT's collator.

Run (2 nodes): srun ... benchmark/test_compile_sft_hf_collator.py
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer

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
    data_dict = {"input_ids": ds["input_ids"], "labels": ds["input_ids"]}
    simple_ds = Dataset.from_dict(data_dict)
    simple_ds.set_format("torch")
    return simple_ds, tokenizer


def main():
    simple_ds, tokenizer = prepare_data()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    args = SFTConfig(
        output_dir="/tmp/sft_hf_collator",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,
        tf32=True,
        max_length=SEQ_LEN,
        include_num_input_tokens_seen=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Pass HF's collator instead of letting SFT create its own
    hf_collator = DataCollatorWithPadding(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=simple_ds,
        processing_class=tokenizer,
        data_collator=hf_collator,
    )

    # Also strip compute_loss to isolate collator effect
    trainer.compute_loss = Trainer.compute_loss.__get__(trainer, type(trainer))

    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] SFT+compile+HF_collator: {ms:.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
