import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    # slurm config
    parser.add_argument("--account", type=str, default="nvr_elm_llm")
    parser.add_argument("--job_name", type=str, default="nvr_elm_llm-ellm:tokenization")
    parser.add_argument("--partition", type=str, default="cpu,cpu_short")
    parser.add_argument("--time", type=str, default="04:00:00")
    parser.add_argument("--project_dir", type=str, default="$HOME/workspace/code/ellm-clean")
    parser.add_argument("--execute", action="store_true")

    # tokenization config
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=5038344)
    parser.add_argument("--postfix", type=str, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--eos_id", type=int, default=151643)
    parser.add_argument("--bos_id", type=int, default=-1)
    parser.add_argument("--chunks", type=int, default=500)
    parser.add_argument("--max_parallel", type=int, default=25)
    parser.add_argument("--doc_token_num_threshold", type=int, default=None)
    parser.add_argument("--formatting_func", type=str, default=None)
    parser.add_argument("--data_source", type=str, default=None)
    parser.add_argument("--data_source_key", type=str, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--file_type", type=str, default=None)

    environment_dir = os.path.dirname(os.popen("which python").read())

    args = parser.parse_args()
    os.makedirs(args.target_path, exist_ok=True)

    slurm_script = (
        f"#!/bin/bash\n"
        + f"#SBATCH -A {args.account}\n"
        + f"#SBATCH -p {args.partition}\n"
        + f"#SBATCH -t {args.time}\n"
        + f"#SBATCH -N 1\n"
        + f"#SBATCH -J {args.job_name}\n"
        + f"#SBATCH --array=0-{args.chunks-1}%{min(args.max_parallel, args.chunks)}\n"
        + f"#SBATCH --exclusive\n"
    )

    if args.mem is not None:
        slurm_script += f"#SBATCH --mem={args.mem}\n"
    slurm_script += "\n"

    slurm_script += f"export LOGLEVEL=INFO\n\n"
    slurm_script += f"export PATH={environment_dir}:$PATH\n"
    slurm_script += f"cd {args.project_dir}\n\n"

    slurm_script += (
        "read -r -d '' cmd <<EOF\n"
        + f"python tokenization.py \\\n"
        + f"\t--src_path {args.src_path} \\\n"
        + f"\t--target_path {args.target_path} \\\n"
        + f"\t--seed {args.seed} --workers {args.workers} \\\n"
        + f"\t--tokenizer {args.tokenizer} --eos_id {args.eos_id} --bos_id {args.bos_id} \\\n"
        + f"\t--chunks $SLURM_ARRAY_TASK_COUNT --index $SLURM_ARRAY_TASK_ID \\\n"
        + f"\t--file_type {args.file_type}"
    )

    if args.doc_token_num_threshold is not None:
        slurm_script += f" \\\n\t--doc_token_num_threshold {args.doc_token_num_threshold}"
    if args.formatting_func is not None:
        slurm_script += f" \\\n\t--formatting_func {args.formatting_func}"
    if args.data_source is not None:
        slurm_script += f" \\\n\t--data_source {args.data_source}"
    if args.data_source_key is not None:
        slurm_script += f" \\\n\t--data_source_key {args.data_source_key}"
    if args.postfix is not None:
        slurm_script += " \\\n\t--postfix " + " ".join(args.postfix)

    slurm_script += "\nEOF\n\n"
    slurm_script += 'srun bash -c "${cmd}"\n'

    slurm_path = os.path.join(args.target_path, "slurm.sh")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)

    slurm_command = f"sbatch {slurm_path}"
    print(slurm_command)
    if args.execute:
        os.system(slurm_command)


if __name__ == "__main__":
    main()
