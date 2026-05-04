# ruff: noqa
"""Test B: HF Trainer + per-layer compile with accelerate FSDP2.

Uses accelerate launch so Trainer picks up the FSDP2 config properly.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_hftrainer.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


SEQ_LEN = 16384


def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()

    # Per-layer compile BEFORE Trainer (which will apply FSDP via accelerate)
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    if getattr(model.config, "num_local_experts", 0) > 0:
        model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)
    print(f"[TEST] Compiled {len(model.model.layers)} layers", flush=True)

    # Create synthetic dataset with pre-tokenized fixed-size sequences
    n_samples = 50
    data = {
        "input_ids": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "labels": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = TrainingArguments(
        output_dir="/tmp/test_hftrainer_compile",
        max_steps=15,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        report_to="none",
        logging_steps=5,
        logging_first_step=True,
        torch_compile=False,  # already compiled per-layer
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
    print(
        f"[TEST] train_runtime={result.metrics['train_runtime']:.1f}s, "
        f"steps_per_second={result.metrics['train_steps_per_second']:.3f}",
        flush=True,
    )
    ms_per_step = result.metrics["train_runtime"] / 15 * 1000
    print(f"[TEST] avg ms/step={ms_per_step:.0f}", flush=True)


if __name__ == "__main__":
    main()
