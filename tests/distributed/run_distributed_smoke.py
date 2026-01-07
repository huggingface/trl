import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "tests" / "accelerate_configs" / "2gpu.yaml"


def run_command(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, env=env, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def ensure_non_empty_dir(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Expected output directory was not created: {path}")
    if not any(path.iterdir()):
        raise SystemExit(f"Expected output directory to contain files: {path}")


def main() -> None:
    accelerate = shutil.which("accelerate")
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sft_output = tmp_path / "sft_out"
        dpo_output = tmp_path / "dpo_out"

        run_command(
            [
                accelerate,
                "launch",
                "--config_file",
                str(CONFIG_PATH),
                "trl/scripts/sft.py",
                "--model_name_or_path",
                "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name",
                "trl-internal-testing/zen",
                "--dataset_config",
                "standard_prompt_only",
                "--output_dir",
                str(sft_output),
            ],
            env,
        )
        ensure_non_empty_dir(sft_output)

        run_command(
            [
                accelerate,
                "launch",
                "--config_file",
                str(CONFIG_PATH),
                "trl/scripts/dpo.py",
                "--model_name_or_path",
                "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                "--dataset_name",
                "trl-internal-testing/zen",
                "--dataset_config",
                "standard_preference",
                "--eval_strategy",
                "no",
                "--no_remove_unused_columns",
                "--output_dir",
                str(dpo_output),
            ],
            env,
        )
        ensure_non_empty_dir(dpo_output)


if __name__ == "__main__":
    main()
