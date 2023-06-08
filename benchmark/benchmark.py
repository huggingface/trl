import argparse
import math
import os
import shlex
import subprocess
import uuid
from distutils.util import strtobool


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="",
        help="the command to run")
    parser.add_argument("--num-seeds", type=int, default=3,
        help="the number of random seeds")
    parser.add_argument("--start-seed", type=int, default=1,
        help="the number of the starting seed")
    parser.add_argument("--workers", type=int, default=0,
        help="the number of workers to run benchmark experimenets")
    parser.add_argument("--auto-tag", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, the runs will be tagged with git tags, commit, and pull request number if possible")
    parser.add_argument("--slurm-template-path", type=str, default=None,
        help="the path to the slurm template file (see docs for more details)")
    parser.add_argument("--slurm-gpus-per-task", type=int, default=1,
        help="the number of gpus per task to use for slurm jobs")
    parser.add_argument("--slurm-total-cpus", type=int, default=50,
        help="the number of gpus per task to use for slurm jobs")
    parser.add_argument("--slurm-ntasks", type=int, default=1,
        help="the number of tasks to use for slurm jobs")
    parser.add_argument("--slurm-nodes", type=int, default=None,
        help="the number of nodes to use for slurm jobs")
    args = parser.parse_args()
    # fmt: on
    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = parse_args()

    commands = []
    for seed in range(0, args.num_seeds):
        commands += [" ".join([args.command, "--seed", str(args.start_seed + seed)])]

    print("======= commands to run:")
    for command in commands:
        print(command)

    if args.workers > 0 and args.slurm_template_path is None:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="cleanrl-benchmark-worker-")
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print("not running the experiments because --workers is set to 0; just printing the commands to run")

    # SLURM logic
    if args.slurm_template_path is not None:
        if not os.path.exists("slurm"):
            os.makedirs("slurm")
        if not os.path.exists("slurm/logs"):
            os.makedirs("slurm/logs")
        print("======= slurm commands to run:")
        with open(args.slurm_template_path) as f:
            slurm_template = f.read()
        slurm_template = slurm_template.replace("{{array}}", f"0-{len(commands) - 1}%{args.workers}")
        slurm_template = slurm_template.replace(
            "{{seeds}}", f"({' '.join([str(args.start_seed + int(seed)) for seed in range(args.num_seeds)])})"
        )
        slurm_template = slurm_template.replace("{{len_seeds}}", f"{args.num_seeds}")
        slurm_template = slurm_template.replace("{{command}}", args.command)
        slurm_template = slurm_template.replace("{{gpus_per_task}}", f"{args.slurm_gpus_per_task}")
        total_gpus = args.slurm_gpus_per_task * args.slurm_ntasks
        slurm_cpus_per_gpu = math.ceil(args.slurm_total_cpus / total_gpus)
        slurm_template = slurm_template.replace("{{cpus_per_gpu}}", f"{slurm_cpus_per_gpu}")
        slurm_template = slurm_template.replace("{{ntasks}}", f"{args.slurm_ntasks}")
        if args.slurm_nodes is not None:
            slurm_template = slurm_template.replace("{{nodes}}", f"#SBATCH --nodes={args.slurm_nodes}")
        else:
            slurm_template = slurm_template.replace("{{nodes}}", "")
        filename = str(uuid.uuid4())
        open(os.path.join("slurm", f"{filename}.slurm"), "w").write(slurm_template)
        slurm_path = os.path.join("slurm", f"{filename}.slurm")
        print(f"saving command in {slurm_path}")
        if args.workers > 0:
            run_experiment(f"sbatch {slurm_path}")
