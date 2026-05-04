# ruff: noqa
"""Detailed per-step timing: A (no max_length) vs B (max_length=16384).

Measures: data loading, forward, backward, step overhead.
Runs on same process, one after another (restarts for each).

Run (2 nodes):
  TEST_MODE=X srun ... benchmark/test_compile_detailed_timing.py
"""

import os
import time

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer
from trl.data_utils import maybe_convert_to_chatml, pack_dataset


SEQ_LEN = 16384
MAX_STEPS = 12  # 2 warmup + 10 measured


class TimedSFTTrainer(SFTTrainer):
    """SFTTrainer with detailed per-step timing."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step

        # Time data-to-GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward + loss (inside compute_loss)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        # Backward
        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss = loss / self.current_gradient_accumulation_steps
        self.accelerator.backward(loss)

        torch.cuda.synchronize()
        t_bwd = time.perf_counter()

        fwd_ms = (t_fwd - t0) * 1000
        bwd_ms = (t_bwd - t_fwd) * 1000
        total_ms = (t_bwd - t0) * 1000

        if self.accelerator.is_main_process and step >= 2 and step % 2 == 0:
            print(f"[STEP {step}] fwd={fwd_ms:.0f}ms bwd={bwd_ms:.0f}ms total={total_ms:.0f}ms", flush=True)

        return loss.detach()


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
        output_dir=f"/tmp/timing_{mode}",
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

    print(f"[TEST] Mode {mode}, max_length={'16384' if mode == 'B' else 'None'}", flush=True)
    args = SFTConfig(**kwargs)
    trainer = TimedSFTTrainer(model=model, args=args, train_dataset=simple_ds, processing_class=tokenizer)
    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] Mode {mode}: {ms:.0f} ms/step total", flush=True)


if __name__ == "__main__":
    main()
