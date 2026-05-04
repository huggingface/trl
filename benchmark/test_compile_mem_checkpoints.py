# ruff: noqa
"""Measure GPU memory at each stage of training step.

Run (2 nodes):
  TEST_MODE=X srun ... benchmark/test_compile_mem_checkpoints.py
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


class MemSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.accelerator.is_main_process and self.state.global_step < 3:
            torch.cuda.synchronize()
            m0 = torch.cuda.memory_allocated() / 1e9

        # SFT's compute_loss sets use_cache=False and calls super
        mode = "train" if self.model.training else "eval"
        inputs["use_cache"] = False
        labels = inputs["labels"]

        (loss, outputs) = super(SFTTrainer, self).compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        if self.accelerator.is_main_process and self.state.global_step < 3:
            torch.cuda.synchronize()
            m1 = torch.cuda.memory_allocated() / 1e9
            logits_size = (
                outputs.logits.numel() * outputs.logits.element_size() / 1e9 if outputs.logits is not None else 0
            )
            print(
                f"[MEM step={self.state.global_step}] before_fwd={m0:.1f}G after_fwd={m1:.1f}G delta={m1 - m0:.1f}G logits={logits_size:.1f}G",
                flush=True,
            )

        # Skip entropy/accuracy to avoid OOM — just return loss
        return (loss, outputs) if return_outputs else loss


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
        output_dir=f"/tmp/memck_{mode}",
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
    trainer = MemSFTTrainer(model=model, args=args, train_dataset=simple_ds, processing_class=tokenizer)
    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000

    peak = torch.cuda.max_memory_allocated() / 1e9
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] Mode {mode}: {ms:.0f} ms/step, peak_mem={peak:.1f}G", flush=True)


if __name__ == "__main__":
    main()
