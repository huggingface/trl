import os
import subprocess
from subprocess import CalledProcessError

from rich.console import Console

from .config_parser import YamlConfigParser


SUPPORTED_COMMANDS = ["sft"]


def main():
    console = Console()

    with console.status("[bold purple]Initializing the CLI..."):
        from transformers import HfArgumentParser, TrainingArguments

        from trl import ModelConfig

        from .sft import ScriptArguments

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
        config = args.config

        # if the configuration is None, create a new `output_dir` variable
        config_parser = YamlConfigParser(config, [args, training_args, model_config])
        current_dir = os.path.dirname(__file__)

        model_name = model_config.model_name_or_path

    command = f"""
    python {current_dir}/{command_name}.py {config_parser.to_string()}
    """

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
