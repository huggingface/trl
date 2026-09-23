# Diagnostic for the DPO chunked-vs-full lm_head.weight gradient mismatch seen only on the min-versions CI job.
# Not a test: run with `python tests/_diag_dpo_lm_head_parity.py`.
import sys
import tempfile

import accelerate
import datasets
import torch
import transformers
from datasets import load_dataset

from trl import DPOConfig, DPOTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
CONFIGS = {
    "loss_types0": dict(
        loss_type=["sigmoid", "hinge", "ipo", "exo_pair", "robust"], loss_weights=[0.2] * 5, label_smoothing=0.1
    ),
    "loss_types1": dict(
        loss_type=["nca_pair", "bco_pair", "sppo_hard", "aot", "aot_unpaired"],
        loss_weights=[0.2] * 5,
        label_smoothing=0.1,
    ),
    "forward_kl": dict(f_divergence_type="forward_kl"),
}
RTOL, ATOL = 1e-3, 5e-4


def describe_patching(model):
    fwd = model.forward
    fwd_mod = getattr(getattr(fwd, "__func__", fwd), "__module__", "?")
    return f"forward from {fwd_mod}; final norm {type(model.model.norm).__module__}.{type(model.model.norm).__name__}"


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def run(name, config_kwargs, patch, tmp_dir):
    dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
    # Build without `use_liger_kernel` so no transformers version patches the model on its own, then patch explicitly.
    args = DPOConfig(
        output_dir=tmp_dir, bf16=False, per_device_train_batch_size=2, report_to="none", **config_kwargs
    )
    trainer = DPOTrainer(model=MODEL_ID, args=args, train_dataset=dataset)
    if patch:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        _apply_liger_kernel_to_instance(model=trainer.model)
    trainer.model.train()
    inputs = trainer._prepare_inputs(next(iter(trainer.get_train_dataloader())))

    results = {}
    for label, chunked in [("chunked", True), ("full", False), ("full_again", False)]:
        trainer.model.zero_grad()
        trainer._metrics["train"].clear()
        trainer.use_liger_kernel = chunked
        loss = trainer.compute_loss(trainer.model, inputs)
        loss.backward()
        results[label] = (loss.detach(), grads_of(trainer.model))

    print(f"\n### {name} | liger_patched={patch} | {describe_patching(trainer.model)}")
    print(f"loss chunked={results['chunked'][0].item():.8f} full={results['full'][0].item():.8f}")
    for other in ["full_again", "chunked"]:
        worst = []
        for pname, g in results["full"][1].items():
            h = results[other][1][pname]
            diff = (h - g).abs()
            bad = ~torch.isclose(h, g, rtol=RTOL, atol=ATOL)
            worst.append((diff.max().item(), pname, int(bad.sum()), g.abs().max().item()))
        worst.sort(reverse=True)
        print(f"-- full vs {other}: top params by max abs diff (max_diff, name, n_mismatch, max|grad|)")
        for row in worst[:4]:
            print(f"   {row[0]:.3e}  {row[1]}  mismatched={row[2]}  max|grad|={row[3]:.3e}")

    g_full = results["full"][1]["lm_head.weight"]
    g_chunk = results["chunked"][1]["lm_head.weight"]
    bad = ~torch.isclose(g_chunk, g_full, rtol=RTOL, atol=ATOL)
    rows = bad.any(dim=1).nonzero().flatten().tolist()
    labels = inputs["input_ids"][:, 1:][inputs["completion_mask"][:, 1:].bool()]
    prompt_ids = inputs["input_ids"][~inputs["completion_mask"].bool() & inputs["attention_mask"].bool()]
    print(f"-- lm_head.weight mismatched rows: {len(rows)}")
    for r in rows[:15]:
        d = (g_chunk[r] - g_full[r]).abs().max().item()
        print(
            f"   row {r}: max_diff={d:.3e} |grad_row|max={g_full[r].abs().max().item():.3e} "
            f"label_count={(labels == r).sum().item()} prompt_count={(prompt_ids == r).sum().item()}"
        )


def main():
    print(
        f"python {sys.version.split()[0]} torch {torch.__version__} transformers {transformers.__version__} "
        f"accelerate {accelerate.__version__} datasets {datasets.__version__}"
    )
    try:
        import liger_kernel

        print(f"liger-kernel {getattr(liger_kernel, '__version__', 'unknown')}")
    except ImportError:
        print("liger-kernel not installed")
    print(f"tf32 matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32}")
    for name, config_kwargs in CONFIGS.items():
        for patch in [False, True]:
            with tempfile.TemporaryDirectory() as tmp_dir:
                run(name, config_kwargs, patch, tmp_dir)


if __name__ == "__main__":
    main()
