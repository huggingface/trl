# ruff: noqa
"""Test: Does adding attention_mask to SFT's signature columns fix the compile regression?

A: SFTTrainer default signature (no attention_mask) — should be slow
B: SFTTrainer with attention_mask added to signature — should be fast if this is the cause

Run (2 nodes): TEST_MODE=X srun ... benchmark/test_compile_attn_mask_sig.py
"""

import os

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

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
    mode = os.environ.get("TEST_MODE", "default")
    simple_ds, tokenizer = prepare_data()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    args = SFTConfig(
        output_dir=f"/tmp/sig_{mode}",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=True,
        tf32=True,
        max_length=SEQ_LEN,
        include_num_input_tokens_seen="no",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(model=model, args=args, train_dataset=simple_ds, processing_class=tokenizer)
    trainer.compute_loss = Trainer.compute_loss.__get__(trainer, type(trainer))

    if mode == "with_attn_mask":
        # Add attention_mask to signature columns
        trainer._signature_columns = [
            "input_ids",
            "labels",
            "attention_mask",
            "seq_lengths",
            "completion_mask",
            "assistant_masks",
        ]
        print("[TEST] Added attention_mask to signature columns", flush=True)
    else:
        print(f"[TEST] Default signature columns: {trainer._signature_columns}", flush=True)

    # Print what the collator actually produces for the first batch
    if trainer.accelerator.is_main_process:
        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        print(
            f"[BATCH] keys={list(batch.keys())}, shapes={{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in batch.items() if hasattr(v, 'shape'))}}}",
            flush=True,
        )

    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] {mode}: {ms:.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
