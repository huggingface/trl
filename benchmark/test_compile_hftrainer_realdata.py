# ruff: noqa
"""HF Trainer (NOT SFTTrainer) with real LongAlign-10k data, tokenized + packed manually.

The fast HF Trainer test used synthetic data (3.8s). This uses real packed data.
If fast → SFTTrainer setup is the issue. If slow → the data pipeline creates slow inputs.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:45:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_hftrainer_realdata.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


SEQ_LEN = 16384


def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()

    # Per-layer compile
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    if getattr(model.config, "num_local_experts", 0) > 0:
        model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)
    print(f"[TEST] Compiled {len(model.model.layers)} layers", flush=True)

    # Tokenize real data and pack manually into SEQ_LEN chunks
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

    # Pack into fixed-size chunks
    n_chunks = min(50, len(all_tokens) // SEQ_LEN)
    packed_data = {
        "input_ids": [all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_chunks)],
        "labels": [all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN] for i in range(n_chunks)],
    }
    dataset = Dataset.from_dict(packed_data)
    dataset.set_format("torch")

    print(f"[TEST] Created {n_chunks} packed chunks of {SEQ_LEN} tokens from real data", flush=True)

    args = TrainingArguments(
        output_dir="/tmp/test_hf_realdata",
        max_steps=15,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        torch_compile=False,
        tf32=True,
        dataloader_pin_memory=False,
        include_num_input_tokens_seen="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    ms = result.metrics["train_runtime"] / 15 * 1000
    if trainer.accelerator.is_main_process:
        print(
            f"[RESULT] HF Trainer + real packed data + compile: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
