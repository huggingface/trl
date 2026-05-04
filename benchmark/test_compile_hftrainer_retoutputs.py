# ruff: noqa
"""Test: HF Trainer with return_outputs=True forced in compute_loss.

If this is slow (~9s), return_outputs=True is the root cause.
If fast (~3.8s), something else in SFT's compute_loss causes it.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=00:30:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_hftrainer_retoutputs.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


SEQ_LEN = 16384


class TrainerReturnOutputs(Trainer):
    """Trainer that forces return_outputs=True in compute_loss, like SFT does."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Force return_outputs=True to materialize logits (like SFT does)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        # Access logits to ensure they're materialized
        if outputs is not None and hasattr(outputs, "logits"):
            _ = outputs.logits.shape
        return (loss, outputs) if return_outputs else loss


def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", torch_dtype=torch.bfloat16)
    model.train()

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    if getattr(model.config, "num_local_experts", 0) > 0:
        model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)
    print(f"[TEST] Compiled {len(model.model.layers)} layers", flush=True)

    n_samples = 50
    data = {
        "input_ids": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "labels": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")

    args = TrainingArguments(
        output_dir="/tmp/test_retoutputs",
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

    trainer = TrainerReturnOutputs(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    result = trainer.train()
    ms_per_step = result.metrics["train_runtime"] / 15 * 1000
    print(
        f"[TEST] return_outputs=True: {ms_per_step:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
