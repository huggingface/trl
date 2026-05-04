# ruff: noqa
"""Print actual input tensor shapes for mode A vs B.

Run (2 nodes):
  TEST_MODE=X srun ... benchmark/test_compile_input_shapes.py
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
MAX_STEPS = 3


class ShapeSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.accelerator.is_main_process and self.state.global_step < 3:
            shapes = {k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__ for k, v in inputs.items()}
            print(f"[SHAPES step={self.state.global_step}] {shapes}", flush=True)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )


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

    # Check raw dataset lengths
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_rank() == 0:
        lens = [len(ds[i]["input_ids"]) for i in range(min(5, len(ds)))]
        print(f"[DATA] First 5 sample lengths: {lens}", flush=True)

    data_dict = {col: ds[col] for col in ds.column_names}
    simple_ds = Dataset.from_dict(data_dict)
    simple_ds.set_format("torch")
    del ds, data_dict

    if dist.get_rank() == 0:
        s = simple_ds[0]["input_ids"]
        print(f"[DATA] After from_dict+set_format: type={type(s)}, len={len(s)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    kwargs = dict(
        output_dir=f"/tmp/shapes_{mode}",
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
    trainer = ShapeSFTTrainer(model=model, args=args, train_dataset=simple_ds, processing_class=tokenizer)

    # Also check collator output directly
    if dist.get_rank() == 0:
        sample = [simple_ds[0]]
        collator_out = trainer.data_collator(sample)
        shapes = {k: tuple(v.shape) for k, v in collator_out.items()}
        print(f"[COLLATOR] Mode {mode}: {shapes}", flush=True)

    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] Mode {mode}: {ms:.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
