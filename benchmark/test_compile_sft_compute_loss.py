# ruff: noqa
"""Test: HF Trainer with SFT's exact compute_loss additions, one by one.

Isolates which specific SFT compute_loss change causes the slowdown.

Run (2 nodes):
  srun --partition=hopper-prod --nodes=2 --gres=gpu:h100:8 --ntasks-per-node=1 --cpus-per-task=64 --mem=0 --exclusive --time=01:00:00 --qos=normal \
    bash -c 'source /fsx/amine_dirhoussi/trl/.venv/bin/activate && export HF_HOME=/fsx/amine_dirhoussi/.cache && export HF_HUB_OFFLINE=1 && \
    accelerate launch --config_file /fsx/amine_dirhoussi/trl/benchmark/generated/qwen3_30b_a3b_ctx16k_n2_fsdp2_dp16_tp1_pp1_cp1_ep1_compile/accelerate_config.yaml \
    --num_processes 16 --num_machines 2 --machine_rank $SLURM_PROCID \
    --main_process_ip $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --main_process_port 29500 \
    benchmark/test_compile_sft_compute_loss.py'
"""

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from trl.trainer.utils import entropy_from_logits


SEQ_LEN = 16384
MAX_STEPS = 15


class TrainerA(Trainer):
    """Baseline: default HF Trainer (return_outputs=False)"""

    pass


class TrainerB(Trainer):
    """+ use_cache=False in inputs"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["use_cache"] = False
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )


class TrainerC(Trainer):
    """+ use_cache=False + return_outputs=True + entropy + accuracy (full SFT compute_loss)"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["use_cache"] = False
        labels = inputs["labels"]

        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        # Entropy (same as SFT)
        with torch.no_grad():
            per_token_entropy = entropy_from_logits(outputs.logits)
            if "attention_mask" in inputs:
                entropy = torch.sum(per_token_entropy * inputs["attention_mask"]) / inputs["attention_mask"].sum()
            else:
                entropy = torch.mean(per_token_entropy)
            entropy = self.accelerator.gather_for_metrics(entropy).mean().item()

        # Token accuracy (same as SFT)
        with torch.no_grad():
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            correct = (predictions == shift_labels) & mask
            correct = self.accelerator.gather_for_metrics(correct.sum())
            total = self.accelerator.gather_for_metrics(mask.sum())
            acc = (correct.sum() / total.sum()).item() if total.sum() > 0 else 0.0

        return (loss, outputs) if return_outputs else loss


def apply_compile(model):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    if getattr(model.config, "num_local_experts", 0) > 0:
        model.config._experts_implementation = "grouped_mm"
    for layer in model.model.layers:
        layer.compile(backend="inductor", fullgraph=True)


def run_test(trainer_cls, label, model_name, dataset, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.train()
    apply_compile(model)

    args = TrainingArguments(
        output_dir=f"/tmp/test_{label}",
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

    trainer = trainer_cls(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train()
    ms = result.metrics["train_runtime"] / MAX_STEPS * 1000

    if trainer.accelerator.is_main_process:
        print(f"[RESULT] {label}: {ms:.0f} ms/step, runtime={result.metrics['train_runtime']:.1f}s", flush=True)

    del model, trainer
    torch.cuda.empty_cache()
    return ms


def main():
    n_samples = 50
    data = {
        "input_ids": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "labels": [torch.randint(0, 1000, (SEQ_LEN,)).tolist() for _ in range(n_samples)],
        "attention_mask": [[1] * SEQ_LEN for _ in range(n_samples)],
    }
    dataset = Dataset.from_dict(data)
    dataset.set_format("torch")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    model_name = "Qwen/Qwen3-30B-A3B"

    print("[START] Running incremental SFT compute_loss tests", flush=True)

    t_a = run_test(TrainerA, "A_baseline", model_name, dataset, tokenizer)
    t_b = run_test(TrainerB, "B_use_cache", model_name, dataset, tokenizer)
    t_c = run_test(TrainerC, "C_full_sft", model_name, dataset, tokenizer)

    import torch.distributed as dist

    if dist.get_rank() == 0:
        print(f"\n{'=' * 60}", flush=True)
        print("SUMMARY (compiled, per-layer):", flush=True)
        print(f"  A (baseline HF Trainer):     {t_a:.0f} ms/step", flush=True)
        print(f"  B (+ use_cache=False):        {t_b:.0f} ms/step  ({t_b / t_a:.2f}x)", flush=True)
        print(f"  C (+ entropy + accuracy):     {t_c:.0f} ms/step  ({t_c / t_a:.2f}x)", flush=True)
        print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()
