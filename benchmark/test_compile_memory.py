# ruff: noqa
"""Measure peak GPU memory for mode A vs B.

Run (2 nodes):
  TEST_MODE=X srun ... benchmark/test_compile_memory.py
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
MAX_STEPS = 5


def main():
    mode = os.environ.get("TEST_MODE", "A")
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
    del ds, data_dict

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    kwargs = dict(
        output_dir=f"/tmp/mem_{mode}",
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
    if mode == "B":
        kwargs["max_length"] = SEQ_LEN

    args = SFTConfig(**kwargs)

    # Reset memory stats before trainer creation
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.max_memory_allocated() / 1e9

    trainer = SFTTrainer(model=model, args=args, train_dataset=simple_ds, processing_class=tokenizer)

    mem_after_init = torch.cuda.max_memory_allocated() / 1e9

    result = trainer.train()

    mem_after_train = torch.cuda.max_memory_allocated() / 1e9

    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(
            f"[MEM] Mode {mode}: before={mem_before:.1f}GiB after_init={mem_after_init:.1f}GiB after_train={mem_after_train:.1f}GiB",
            flush=True,
        )
        print(f"[RESULT] Mode {mode}: {ms:.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
