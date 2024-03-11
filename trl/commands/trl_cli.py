import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser("HuggingFace TRL CLI tool", usage="trl <command> [<args>]")

    # Step 3: Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Choose a command")

    # Subparser for 'sft' command
    sft_parser = subparsers.add_parser("sft", help="Description of sft command")
    sft_parser.add_argument("--model_name", type=str, help="Name of the model", required=True)
    sft_parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)

    # Subparser for 'dpo' command
    dpo_parser = subparsers.add_parser("dpo", help="Description of dpo command")
    dpo_parser.add_argument("--model_name", type=str, help="Name of the model", required=True)
    dpo_parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)

    args = parser.parse_args()

    # Get the required args
    model_name = args.model_name
    dataset_name = args.dataset_name

    current_dir = os.path.dirname(__file__)
    trl_root_dir = "/".join(current_dir.split("/")[:-2])

    command = f"""
    python {trl_root_dir}/examples/scripts/{args.command}.py \
        --model_name_or_path {model_name} \
        --dataset_name {dataset_name}
    """

    subprocess.run(
        command.split(),
        text=True,
        check=True,
        encoding="utf-8",
        cwd=os.getcwd(),
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    main()
