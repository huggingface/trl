# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import accelerate
import pytest
import torch
import transformers

from ..testing_utils import is_bf16_supported


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

SFT_DATASET = "trl-lib/Capybara"
DPO_DATASET = "trl-lib/ultrafeedback_binarized"

REFERENCES_DIR = Path(__file__).parent / "references"

NUM_STEPS = 50
SEED = 42
MAX_LENGTH = 512


def _trl_commit() -> str:
    """Return the current trl commit SHA (with `-dirty` suffix if the working tree has uncommitted changes).

    Assumes the suite is run from a `pip install -e .` checkout — the only intended setup.
    """
    cwd = Path(__file__).parent
    sha = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(cwd), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def env_snapshot() -> dict:
    return {
        "accelerate": accelerate.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": _trl_commit(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@dataclass
class StepRecord:
    step: int
    loss: float
    grad_norm: float


@dataclass
class Trajectory:
    config: dict
    env: dict
    steps: list[StepRecord]


@dataclass
class CorrectnessConfig:
    name: str
    method: str  # "sft" | "dpo"
    args: dict[str, str]
    num_processes: int = 1

    def cli_args(self) -> list[str]:
        out: list[str] = []
        for k, v in self.args.items():
            out.extend([f"--{k}", v])
        return out


def run(config: CorrectnessConfig) -> Trajectory:
    """Invoke the trl CLI as a subprocess; parse its trainer_state.json into a Trajectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["trl", config.method]
        cmd += ["--num_processes", str(config.num_processes)]
        cmd += ["--output_dir", tmpdir, *config.cli_args()]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(config.num_processes))}
        subprocess.run(cmd, check=True, env=env)

        state_paths = list(Path(tmpdir).glob("**/trainer_state.json"))
        if not state_paths:
            raise RuntimeError(f"trainer_state.json not produced in {tmpdir}")
        state = json.loads(state_paths[0].read_text())

    steps = [
        StepRecord(
            step=int(log["step"]),
            loss=float(log["loss"]),
            grad_norm=float(log["grad_norm"]),
        )
        for log in state["log_history"]
        if "loss" in log  # skip eval and final-summary entries
    ]

    return Trajectory(
        config={"name": config.name, "method": config.method, "args": config.args},
        env=env_snapshot(),
        steps=steps,
    )


def save(trajectory: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(trajectory), indent=2))


def load(path: Path) -> Trajectory:
    data = json.loads(path.read_text())
    return Trajectory(
        config=data["config"],
        env=data["env"],
        steps=[StepRecord(**s) for s in data["steps"]],
    )


def compare_scalars(a: Trajectory, b: Trajectory, tol: dict[str, float], residual_tol: dict[str, float]) -> list[str]:
    """Compare scalar series (loss, grad_norm). `tol` and `residual_tol` are per-field dicts keyed by `'loss'` and
    `'grad_norm'`."""
    errors: list[str] = []
    if len(a.steps) != len(b.steps):
        return [f"length mismatch: {len(a.steps)} vs {len(b.steps)}"]

    for field in ("loss", "grad_norm"):
        sa = [getattr(s, field) for s in a.steps]
        sb = [getattr(s, field) for s in b.steps]
        diffs = [x - y for x, y in zip(sa, sb, strict=False)]
        max_abs = max(abs(d) for d in diffs)
        if max_abs > tol[field]:
            i = max(range(len(diffs)), key=lambda k: abs(diffs[k]))
            step = a.steps[i].step
            errors.append(
                f"{field}: max |Δ|={max_abs:.3e} at step {step} (a={sa[i]:.6e}, b={sb[i]:.6e}, tol={tol[field]:.1e})"
            )

        mean = sum(diffs) / len(diffs)
        if abs(mean) > residual_tol[field]:
            errors.append(f"{field}: systematic drift, mean Δ={mean:.3e} (tol={residual_tol[field]:.1e})")

    return errors


def _build(
    name: str, method: str, dataset: str, attn: str = "eager", num_processes: int = 1, **overrides
) -> CorrectnessConfig:
    args: dict[str, str] = {
        "model_name_or_path": MODEL,
        "model_revision": MODEL_REVISION,
        "attn_implementation": attn,
        "dataset_name": dataset,
        "max_steps": str(NUM_STEPS),
        "max_length": str(MAX_LENGTH),
        "logging_steps": "1",
        "report_to": "none",
        "seed": str(SEED),
        "data_seed": str(SEED),
        "full_determinism": "True",
        # Force pure fp32 training for maximal determinism and to avoid bfloat16-induced divergences.
        "bf16": "False",
    }
    args.update({k: str(v) for k, v in overrides.items()})
    return CorrectnessConfig(name=name, method=method, args=args, num_processes=num_processes)


# Equivalence classes: each maps to a `members` list plus per-field `tol` (max |Δ|) and `residual_tol` (mean Δ)
# dicts. The first member is the canonical config — it owns the class's reference snapshot and is the only one
# re-recorded under `--update-references`. Every other member is asserted to match that snapshot.
# Tuning tip: run `python tests/invariant/test_invariant.py <klass> --report` to see actual Δs and set tolerances
# to ~1.5–2× the observed noise.
EQUIVALENCE_CLASSES: dict[str, dict] = {
    "sft": {
        "tol": {"loss": 1e-3, "grad_norm": 1e-1},
        "residual_tol": {"loss": 1e-5, "grad_norm": 1e-3},
        "members": [
            _build("sft_default", "sft", SFT_DATASET),
            _build("sft_pdb1_gas8", "sft", SFT_DATASET, per_device_train_batch_size=1, gradient_accumulation_steps=8),
            _build("sft_no_grad_ckpt", "sft", SFT_DATASET, gradient_checkpointing=False),
            _build("sft_ddp2", "sft", SFT_DATASET, per_device_train_batch_size=4, num_processes=2),
        ],
    },
    "sft_fa2": {
        # loss_type not pinned; this class exercises the current SFTConfig default ("chunked_nll").
        # Loss is much tighter than grad_norm under FA2+bf16 (grad_norm absorbs bf16 + FA varlen kernel noise).
        # The grad_norm tol (5.0) is intentionally ~50× looser than the non-FA2 sft class (0.1): it is sized to the
        # FA2 varlen kernel noise observed in practice, not a regression budget. Do not tighten it without re-running
        # the class and confirming the new gap; see https://github.com/huggingface/trl/pull/5842#issuecomment-4539190615
        "tol": {"loss": 1.5e-2, "grad_norm": 5.0},
        "residual_tol": {"loss": 1e-3, "grad_norm": 2.5e-1},
        "members": [
            _build(
                "sft_fa2",
                "sft",
                SFT_DATASET,
                attn="kernels-community/flash-attn2",  # to avoid cross-contamination between samples when padding_free=True
                bf16=True,  # required for FA2 kernels, which are bfloat16-only
                max_length=None,  # Required when padding_free=True
                per_device_train_batch_size=2,
            ),
            _build(
                "sft_fa2_padfree",
                "sft",
                SFT_DATASET,
                attn="kernels-community/flash-attn2",  # to avoid cross-contamination between samples when padding_free=True
                bf16=True,  # required for FA2 kernels, which are bfloat16-only
                max_length=None,  # Required when padding_free=True
                per_device_train_batch_size=2,
                padding_free=True,
            ),
        ],
    },
    "dpo": {
        "tol": {"loss": 1e-4, "grad_norm": 1e-2},
        "residual_tol": {"loss": 1e-5, "grad_norm": 1e-3},
        "members": [
            _build("dpo_default", "dpo", DPO_DATASET),
            _build("dpo_pdb1_gas8", "dpo", DPO_DATASET, per_device_train_batch_size=1, gradient_accumulation_steps=8),
            _build("dpo_no_grad_ckpt", "dpo", DPO_DATASET, gradient_checkpointing=False),
            _build("dpo_ddp2", "dpo", DPO_DATASET, per_device_train_batch_size=4, num_processes=2),
        ],
    },
}


_ALL = [(klass, c) for klass, ec in EQUIVALENCE_CLASSES.items() for c in ec["members"]]


@pytest.mark.invariant
@pytest.mark.parametrize("klass,config", _ALL, ids=[c.name for _, c in _ALL])
def test_invariant(klass, config):
    ref_path = REFERENCES_DIR / f"{klass}.json"
    if not ref_path.exists():
        pytest.fail(f"no reference at {ref_path}; record it with `python {Path(__file__).name}`")

    if config.num_processes > 1 and torch.cuda.device_count() < config.num_processes:
        pytest.skip(f"requires {config.num_processes} GPUs, got {torch.cuda.device_count()}")

    # FA2 members require bf16 (the kernels are bfloat16-only), and bf16=True raises on a device that
    # does not support it (CPU, pre-Ampere GPU). Skip rather than error on such devices.
    if config.args.get("bf16") == "True" and not is_bf16_supported():
        pytest.skip("config requires bf16, which the current device does not support")

    trajectory = run(config)
    reference = load(ref_path)
    ec = EQUIVALENCE_CLASSES[klass]
    errors = compare_scalars(trajectory, reference, tol=ec["tol"], residual_tol=ec["residual_tol"])
    assert not errors, f"'{config.name}' diverges from class '{klass}' reference:\n  " + "\n  ".join(errors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record canonical reference trajectories for the invariant tests.")
    parser.add_argument(
        "klass",
        nargs="*",
        choices=list(EQUIVALENCE_CLASSES),
        help="Equivalence class(es) to record. Default: all.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow recording from a dirty working tree (snapshot will pin an irreproducible state).",
    )
    cli_args = parser.parse_args()

    if _trl_commit().endswith("-dirty") and not cli_args.allow_dirty:
        sys.exit(
            "Refusing to record from a dirty working tree: the snapshot would pin a state that can't be "
            "reproduced from a commit SHA. Commit your changes first, or pass --allow-dirty to override."
        )

    classes = cli_args.klass or list(EQUIVALENCE_CLASSES)
    for klass in classes:
        canonical = EQUIVALENCE_CLASSES[klass]["members"][0]
        print(f"recording '{klass}' from canonical config '{canonical.name}'")  # noqa: T201
        trajectory = run(canonical)
        ref_path = REFERENCES_DIR / f"{klass}.json"
        save(trajectory, ref_path)
        print(f"  → {ref_path}")  # noqa: T201
