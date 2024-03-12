import os
import subprocess
from dataclasses import dataclass, field
from subprocess import CalledProcessError

from rich.console import Console

from .config_parser import YamlConfigParser


SUPPORTED_COMMANDS = ["sft"]


@dataclass
class ScriptArguments:
    dataset_name: str = field(metadata={"help": "the dataset name"})
    dataset_text_field: str = field(default="text", metadata={"help": "the text field of the dataset"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    packing: bool = field(default=False, metadata={"help": "Whether to apply data packing or not during training"})


def main():
    console = Console()

    with console.status("[bold purple]Initializing the CLI..."):
        from transformers import HfArgumentParser, TrainingArguments

        from trl import ModelConfig

        parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))

        (args, training_args, model_config, command_name) = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )

        command_name = command_name[0]

        if command_name not in SUPPORTED_COMMANDS:
            raise ValueError(
                f"Please use one of the supported commands, got {command_name} - supported commands are {SUPPORTED_COMMANDS}"
            )

        # Get the required args
        model_name = model_config.model_name_or_path
        dataset_name = args.dataset_name
        dataset_text_field = args.dataset_text_field
        max_seq_length = args.max_seq_length
        config = args.config

        # if the configuration is None, create a new `output_dir` variable
        config_parser = YamlConfigParser(config)
        output_dir = training_args.output_dir
        report_to = config_parser.report_to

        current_dir = os.path.dirname(__file__)

    EXTRA_AGRS = """
    """

    if args.packing:
        EXTRA_AGRS += """--packing yes """

    command = f"""
    python {current_dir}/{command_name}.py \
        --model_name_or_path {model_name} \
        --dataset_name {dataset_name} \
        --output_dir {output_dir} \
        --report_to {report_to} \
        --dataset_text_field {dataset_text_field} \
        --max_seq_length {max_seq_length} \
    """

    command += EXTRA_AGRS

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
        console.log(f"TRL - {command_name.upper()} failed on {model_name}! See the logs above for further details.")


if __name__ == "__main__":
    main()
