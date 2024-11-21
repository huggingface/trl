import argparse

from transformers import HfArgumentParser, TrainingArguments
import sys
from pathlib import Path

# Add the root of the project to the python path so that we can import examples scripts
path = Path(__file__).parent.parent
sys.path.append(str(path))


def main():
    parser = argparse.ArgumentParser(prog="trl", description="A CLI tool for training and fine-tuning")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=HfArgumentParser)

    # 'dpo' subcommand
    dpo_parser = subparsers.add_parser("dpo", help="Run the DPO training process", dataclass_types=TrainingArguments)

    args = parser.parse_args()
    sys.argv = sys.argv[1:]  # Remove 'trl' from sys.argv

    if args.command == "dpo":
        from examples.scripts.dpo import main as dpo_main

        (training_args,) = dpo_parser.parse_args_into_dataclasses()
        dpo_main(training_args)

