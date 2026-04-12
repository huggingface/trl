# ruff: noqa: T201
#!/usr/bin/env python3
"""
Benchmark runner for TRL SFT scaling tests.

Default mode (dry-run): prints rendered sbatch scripts to stdout.
With --submit: writes generated files and submits jobs via sbatch.

Usage:
    # Dry-run — review generated scripts
    python benchmark/run_benchmark.py --config benchmark/configs/qwen3_4b.yaml

    # Submit jobs
    python benchmark/run_benchmark.py --config benchmark/configs/qwen3_4b.yaml --submit
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


def build_run_id(config: dict, run: dict) -> str:
    model_short = config["model_name_or_path"].split("/")[-1].lower().replace("-", "_")
    ctx_k = run["context_length"] // 1024
    parts = [
        model_short,
        f"ctx{ctx_k}k",
        f"n{run['nodes']}",
        run["backend"],
        f"dp{run['dp']}",
        f"tp{run['tp']}",
        f"pp{run['pp']}",
        f"cp{run['cp']}",
        f"ep{run['ep']}",
    ]
    if run.get("sp", 1) > 1:
        parts.append(f"sp{run['sp']}")
    attn = run.get("attn_implementation", config.get("attn_implementation", "sdpa"))
    if attn != "sdpa":
        parts.append(attn.replace("_", ""))
    if run.get("cpu_offload", False):
        parts.append("offload")
    return "_".join(parts)


def render_accelerate_config(env: Environment, run: dict, config: dict) -> str:
    backend = run["backend"]
    gpus_per_node = config.get("gpus_per_node", 8)
    num_processes = run["nodes"] * gpus_per_node
    cpu_offload = run.get("cpu_offload", False)

    template_vars = {
        "nodes": run["nodes"],
        "num_processes": num_processes,
        "tp": run["tp"],
        "cp": run["cp"],
        "pp": run.get("pp", 1),
        "ep": run.get("ep", 1),
        "sp": run.get("sp", 1),
        "cpu_offload": cpu_offload,
        "cpu_ram_efficient_loading": run.get(
            "cpu_ram_efficient_loading", config.get("cpu_ram_efficient_loading", True)
        ),
    }

    # Compute DP shard/replicate sizes for parallelism_config
    # Total GPUs = dp_replicate * dp_shard * tp * cp * pp * sp
    # Note: EP is NOT included — it's handled by transformers' distribute_model, not accelerate's mesh
    non_dp = run["tp"] * run["cp"] * run.get("pp", 1) * run.get("sp", 1)
    if non_dp > 0 and num_processes // non_dp > 0:
        dp_total = num_processes // non_dp
        # dp_shard = dp (FSDP shards), dp_replicate = dp_total // dp_shard
        template_vars["dp_shard"] = min(run["dp"], dp_total)
        template_vars["dp_replicate"] = max(1, dp_total // template_vars["dp_shard"])

    if backend == "fsdp2":
        template = env.get_template("accelerate/fsdp2.yaml.j2")
    elif backend in ("deepspeed", "deepspeed_zero3"):
        template = env.get_template("accelerate/deepspeed_zero3.yaml.j2")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return template.render(**template_vars)


def render_launch_script(env: Environment, run: dict, config: dict, run_id: str, accel_config_path: str) -> str:
    gpus_per_node = config.get("gpus_per_node", 8)
    num_processes = run["nodes"] * gpus_per_node
    use_reentrant_gc = run.get("use_reentrant_gc", config.get("use_reentrant_gc", False))

    template = env.get_template("launch.sh.j2")
    return template.render(
        nodes=run["nodes"],
        num_processes=num_processes,
        model_name_or_path=config["model_name_or_path"],
        dataset_name=config.get("dataset_name", "THUDM/LongAlign-10k"),
        context_length=run["context_length"],
        accelerate_config_path=accel_config_path,
        output_dir=f"benchmark/outputs/{run_id}",
        per_device_train_batch_size=run.get("per_device_train_batch_size", 1),
        max_steps=run.get("max_steps", 20),
        logging_steps=run.get("logging_steps", 5),
        use_reentrant_gc=use_reentrant_gc,
        attn_implementation=run.get("attn_implementation", config.get("attn_implementation", "sdpa")),
        extra_args=run.get("extra_args", config.get("extra_args", "")),
    )


def render_sbatch_script(env: Environment, run: dict, config: dict, run_id: str, launch_script_path: str) -> str:
    gpus_per_node = config.get("gpus_per_node", 8)

    template = env.get_template("sft.sbatch.j2")
    return template.render(
        job_name=f"bench-{run_id}",
        nodes=run["nodes"],
        gpus_per_node=gpus_per_node,
        qos=run.get("qos", "normal"),
        time_limit=run.get("time_limit", "4:00:00"),
        model_name_or_path=config["model_name_or_path"],
        context_length=run["context_length"],
        wandb_project=config.get("wandb_project", "trl-sft-benchmark"),
        wandb_run_name=run_id,
        backend=run["backend"],
        dp=run["dp"],
        tp=run["tp"],
        pp=run.get("pp", 1),
        cp=run["cp"],
        ep=run.get("ep", 1),
        launch_script_path=launch_script_path,
    )


def main():
    parser = argparse.ArgumentParser(description="TRL SFT Benchmark Runner")
    parser.add_argument("--config", required=True, help="Path to benchmark config YAML")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Write files and submit jobs (default: dry-run)",
    )
    parser.add_argument("--run-index", type=int, nargs="*", help="Only run specific indices (0-based)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), keep_trailing_newline=True)

    runs = config["runs"]
    if args.run_index is not None:
        runs = [(i, runs[i]) for i in args.run_index]
    else:
        runs = list(enumerate(runs))

    for idx, run in runs:
        run_id = build_run_id(config, run)

        # Render accelerate config
        accel_yaml = render_accelerate_config(env, run, config)

        if args.submit:
            # Write files
            run_dir = GENERATED_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            accel_path = run_dir / "accelerate_config.yaml"
            accel_path.write_text(accel_yaml)

            launch_script = render_launch_script(env, run, config, run_id, str(accel_path))
            launch_path = run_dir / "launch.sh"
            launch_path.write_text(launch_script)
            launch_path.chmod(0o755)

            sbatch_script = render_sbatch_script(env, run, config, run_id, str(launch_path))
            sbatch_path = run_dir / "job.sbatch"
            sbatch_path.write_text(sbatch_script)

            # Submit
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            if result.returncode == 0:
                print(f"[{idx}] {run_id}: {result.stdout.strip()}")
            else:
                print(
                    f"[{idx}] {run_id}: FAILED — {result.stderr.strip()}",
                    file=sys.stderr,
                )
        else:
            # Dry-run: print to stdout
            accel_path_placeholder = f"benchmark/generated/{run_id}/accelerate_config.yaml"
            launch_script = render_launch_script(env, run, config, run_id, accel_path_placeholder)
            launch_path_placeholder = f"benchmark/generated/{run_id}/launch.sh"
            sbatch_script = render_sbatch_script(env, run, config, run_id, launch_path_placeholder)

            print(f"{'=' * 80}")
            print(f"RUN [{idx}]: {run_id}")
            print(f"{'=' * 80}")
            print()
            print("--- accelerate_config.yaml ---")
            print(accel_yaml)
            print("--- launch.sh ---")
            print(launch_script)
            print("--- job.sbatch ---")
            print(sbatch_script)
            print()


if __name__ == "__main__":
    main()
