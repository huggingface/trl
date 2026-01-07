import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tests" / "accelerate_configs" / "2gpu.yaml"

def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, env=env, cwd=ROOT, check=False)
    assert result.returncode == 0

def test_sft():
    accelerate = shutil.which("accelerate")
    with tempfile.TemporaryDirectory() as tmpdir:
        run_command(
            [
                # fmt: off
                accelerate, "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/sft.py",
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen", "--dataset_config", "standard_language_modeling",
                "--output_dir", tmpdir,
                # fmt: on
            ],
            os.environ.copy(),
        )


def test_dpo():
    accelerate = shutil.which("accelerate")
    with tempfile.TemporaryDirectory() as tmpdir:
        run_command(
            [
                # fmt: off
                accelerate, "launch", "--config_file", str(CONFIG_PATH), "trl/scripts/dpo.py",
                "--model_name_or_path", "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name", "trl-internal-testing/zen", "--dataset_config", "standard_preerence",
                "--output_dir", tmpdir,
                # fmt: on
            ],
            os.environ.copy(),
        )
