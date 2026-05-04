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

    def cli_args(self) -> list[str]:
        out: list[str] = []
        for k, v in self.args.items():
            out.extend([f"--{k}", v])
        return out


def run(config: CorrectnessConfig) -> Trajectory:
    """Invoke the trl CLI as a subprocess; parse its trainer_state.json into a Trajectory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["trl", config.method, "--output_dir", tmpdir, *config.cli_args()]
        subprocess.run(cmd, check=True)

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


def compare_scalars(a: Trajectory, b: Trajectory, tol: float, residual_tol: float) -> list[str]:
    """Compare scalar series (loss, grad_norm). Returns a list of error messages, empty if equal."""
    errors: list[str] = []
    if len(a.steps) != len(b.steps):
        return [f"length mismatch: {len(a.steps)} vs {len(b.steps)}"]

    for field in ("loss", "grad_norm"):
        sa = [getattr(s, field) for s in a.steps]
        sb = [getattr(s, field) for s in b.steps]
        diffs = [x - y for x, y in zip(sa, sb, strict=False)]
        max_abs = max(abs(d) for d in diffs)
        if max_abs > tol:
            i = max(range(len(diffs)), key=lambda k: abs(diffs[k]))
            step = a.steps[i].step
            errors.append(
                f"{field}: max |Δ|={max_abs:.3e} at step {step} (a={sa[i]:.6e}, b={sb[i]:.6e}, tol={tol:.1e})"
            )

        mean = sum(diffs) / len(diffs)
        if abs(mean) > residual_tol:
            errors.append(f"{field}: systematic drift, mean Δ={mean:.3e} (tol={residual_tol:.1e})")

    return errors


def _build(name: str, method: str, dataset: str, attn: str = "eager", **overrides) -> CorrectnessConfig:
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
    return CorrectnessConfig(name=name, method=method, args=args)


# Equivalence classes: each maps to a `members` list and a tolerance pair. The first member is the canonical
# config — it owns the class's reference snapshot and is the only one re-recorded under `--update-references`.
# Every other member is asserted to match that snapshot.
EQUIVALENCE_CLASSES: dict[str, dict] = {
    "sft": {
        "tol": 5e-2,
        "residual_tol": 1e-2,
        "members": [
            _build("sft_default", "sft", SFT_DATASET),
            _build("sft_pdb1_gas8", "sft", SFT_DATASET, per_device_train_batch_size=1, gradient_accumulation_steps=8),
            _build("sft_no_grad_ckpt", "sft", SFT_DATASET, gradient_checkpointing=False),
        ],
    },
    "dpo": {
        "tol": 5e-2,
        "residual_tol": 1e-2,
        "members": [
            _build("dpo_default", "dpo", DPO_DATASET),
            _build("dpo_pdb1_gas8", "dpo", DPO_DATASET, per_device_train_batch_size=1, gradient_accumulation_steps=8),
            _build("dpo_no_grad_ckpt", "dpo", DPO_DATASET, gradient_checkpointing=False),
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
