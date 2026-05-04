# ruff: noqa
"""Direct A/B: HF Trainer vs SFTTrainer, SAME manually packed real data, same compile.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=01:00:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_vs_hf_same_data.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from trl import SFTConfig, SFTTrainer


SEQ_LEN = 16384
MAX_STEPS = 15


def prepare_real_packed_data():
    """Tokenize LongAlign-10k and pack into fixed-size chunks."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    ds = load_dataset("THUDM/LongAlign-10k", split="train")
    all_tokens = []
    for sample in ds:
        msgs = sample.get("messages", [])
        text = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
        if text:
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(toks)
        if len(all_tokens) > SEQ_LEN * 100:
            break
    n_chunks = min(50, len(all_tokens) // SEQ_LEN)
    data = {
        "input_ids": [all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_chunks)],
        "labels": [all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_chunks)],
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")
    return dataset, tokenizer


def apply_compile(model):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    if getattr(model.config, "num_local_experts", 0) > 0:
        model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)


def run_hf_trainer(dataset, tokenizer, label):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()
    apply_compile(model)
    args = TrainingArguments(
        output_dir=f"/tmp/{label}",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        logging_steps=999,
        torch_compile=False,
        tf32=True,
        dataloader_pin_memory=False,
        include_num_input_tokens_seen="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] {label}: {ms:.0f} ms/step", flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    return ms


def run_sft_trainer(dataset, tokenizer, label):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    args = SFTConfig(
        output_dir=f"/tmp/{label}",
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
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000
    if trainer.accelerator.is_main_process:
        print(f"[RESULT] {label}: {ms:.0f} ms/step", flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    return ms


def main():
    dataset, tokenizer = prepare_real_packed_data()

    print(f"[TEST] Dataset: {len(dataset)} samples, seq_len={SEQ_LEN}", flush=True)

    import os

    mode = os.environ.get("TEST_MODE", "hf")

    if mode == "hf":
        run_hf_trainer(dataset, tokenizer, "HF_Trainer_compile")
    elif mode == "sft":
        run_sft_trainer(dataset, tokenizer, "SFT_Trainer_compile")

    import torch.distributed as dist

    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"\n{'=' * 60}", flush=True)
        print("SAME DATA, SAME COMPILE:", flush=True)
        print(f"  HF Trainer:  {t_hf:.0f} ms/step", flush=True)
        print(f"  SFT Trainer: {t_sft:.0f} ms/step  ({t_sft / t_hf:.2f}x)", flush=True)
        print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
