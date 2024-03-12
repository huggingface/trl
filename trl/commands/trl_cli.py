import argparse
import os
import subprocess
from subprocess import CalledProcessError

from rich.console import Console

from .config_parser import YamlConfigParser


def main():
    console = Console()

    with console.status("[bold green]Initializing the CLI..."):
        parser = argparse.ArgumentParser("HuggingFace TRL CLI tool", usage="trl <command> [<args>]")

        # Step 3: Create subparsers for different commands
        subparsers = parser.add_subparsers(dest="command", help="Choose a command")

        # Subparser for 'sft' command
        sft_parser = subparsers.add_parser("sft", help="Description of sft command")
        sft_parser.add_argument("--model_name", type=str, help="Name of the model", required=True)
        sft_parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)
        sft_parser.add_argument("--config", type=str, help="Path to the config file", required=False, default=None)

        # Subparser for 'dpo' command
        dpo_parser = subparsers.add_parser("dpo", help="Description of dpo command")
        dpo_parser.add_argument("--model_name", type=str, help="Name of the model", required=True)
        dpo_parser.add_argument("--dataset_name", type=str, help="Name of the dataset", required=True)
        dpo_parser.add_argument("--config", type=str, help="Path to the config file", required=False, default=None)

        args = parser.parse_args()

    # Get the required args
    model_name = args.model_name
    dataset_name = args.dataset_name
    config = args.config

    # if the configuration is None, create a new `output_dir` variable
    config_parser = YamlConfigParser(config)
    output_dir = config_parser.output_dir

    current_dir = os.path.dirname(__file__)

    command = f"""
    python {current_dir}/{args.command}.py \
        --model_name_or_path {model_name} \
        --dataset_name {dataset_name}
        --output_dir {output_dir}
    """

    with console.status(f"[bold green]Running {args.command.upper()}..."):
        try:
            subprocess.run(
                command.split(),
                text=True,
                check=True,
                encoding="utf-8",
                cwd=os.getcwd(),
                env=os.environ.copy(),
            )
        except (CalledProcessError, ChildProcessError):
            console.log(f"{args.command.upper()} failed ! See the logs in `output.log` for further details.")


if __name__ == "__main__":
    main()
