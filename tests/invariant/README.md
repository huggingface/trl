# Training invariant tests

Catches silent training bugs that don't fail unit tests but shift the training trajectory. Runs on real models, opt-in only.

## How it works

Configs are grouped into **equivalence classes**: configs in the same class must produce the same trajectory (e.g. PDB=1×GAS=8 must equal PDB=8×GAS=1, FA2 must equal eager). Each class has one **canonical** config (the first one) that owns the class's reference snapshot. Every config in the class — canonical included — is asserted to match the saved reference. This catches both invariant breakage (a non-canonical config drifting away from the canonical's pinned trajectory) and numerical regressions (the canonical itself drifting from its committed snapshot across versions).

Recording the references is a separate concern from testing them, so it's a separate entry point (`python tests/invariant/test_invariant.py`).

Each config is a `trl <method>` CLI invocation with a fixed set of args. The harness shells out (`subprocess.run(["trl", method, ...])`), the CLI runs end to end, writes `trainer_state.json` to its `--output_dir`, and the harness parses the `log_history` into a `Trajectory`.

This means the suite tests the actual user-facing entry point, not the Python API. Catches CLI-only bugs (arg parsing, defaults, dispatch) for free. Distributed runs are an additive change: prepend `accelerate launch --config_file <strategy>.yaml` to the same command.

## Scope (initial)

- Trainers: `trl sft`, `trl dpo`
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (pinned revision)
- Equivalence classes:
  - `sft`: `sft_default` (canonical), `sft_pdb1_gas8` (gradient accumulation), `sft_attn_fa2_kernels` (FA2 via kernels)
  - `dpo`: `dpo_default` (canonical), `dpo_pdb1_gas8` (gradient accumulation)
- Single GPU, fp32, fixed seed, ~50 optimizer steps.

Other axes (sharding, DDP, more trainers) are deferred and will be additive.

## Trajectory

Per optimizer step: `loss`, `grad_norm`. One JSON per equivalence class in `references/` (`sft.json`, `dpo.json`):

```json
{
  "config": {"name": "sft_default", "method": "sft", "args": {...}},
  "env":    {"accelerate": "...", "torch": "...", "transformers": "...", "trl": "...", "gpu": "H100-80GB"},
  "steps":  [{"step": 1, "loss": 1.234, "grad_norm": 0.567}, ...]
}
```

## Comparison

Scalar series with absolute tolerance + zero-mean-residual. The residual check is what flags bugs like GAS-dropping — they show up as a one-sided systematic shift in the loss curve, not as point-wise outliers.

## Hardware

Reference snapshots are recorded on **H100 80GB** (pinned in `references/env.lock`).

## Running

```bash
# test
pytest tests/invariant/ -m invariant

# record references
python tests/invariant/test_invariant.py      # all classes
python tests/invariant/test_invariant.py sft  # one class
```

Snapshot updates must be justified in the PR description.
