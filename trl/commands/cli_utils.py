# This file is a copy of trl/examples/scripts/sft.py so that we could
# use it together with rich and the TRL CLI in a more customizable manner.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import yaml
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils.deprecation import deprecate_kwarg


logger = logging.getLogger(__name__)


class YamlConfigParser:
    """ """

    def __init__(self) -> None:
        warnings.warn(
            "The `YamlConfigParser` class is deprecated and will be removed in version 0.14. "
            "If you need to use this class, please copy the code to your own project.",
            DeprecationWarning,
        )

    def parse_and_set_env(self, config_path: str) -> dict:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)

        if "env" in config:
            env_vars = config.pop("env")
            if isinstance(env_vars, dict):
                for key, value in env_vars.items():
                    os.environ[key] = str(value)
            else:
                raise ValueError("`env` field should be a dict in the YAML file.")

        return config

    def to_string(self, config):
        final_string = ""
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                if len(value) != 0:
                    value = str(value)
                    value = value.replace("'", '"')
                    value = f"'{value}'"
                else:
                    continue

            final_string += f"--{key} {value} "
        return final_string


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make
    """
    import logging
    import warnings

    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


@dataclass
class ChatArguments:
    # general settings
    model_name_or_path: str = field(metadata={"help": "Name of the pre-trained model"})
    user: str = field(default=None, metadata={"help": "Username to display in chat interface"})
    system_prompt: str = field(default=None, metadata={"help": "System prompt"})
    save_folder: str = field(default="./chat_history/", metadata={"help": "Folder to save chat history"})
    device: str = field(
        default="cpu",
        metadata={"help": "device to use for inference."},
    )
    config: str = field(
        default="default",
        metadata={
            "help": "Config file used for setting the configs. If `default` uses examples/scripts/config/default_chat_config.yaml"
        },
    )
    examples: str = field(default=None, metadata={"help": "Empty placeholder needs to be set via config."})
    # generation settings
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum number of tokens to generate"})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample outputs during generation"})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search"})
    temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for generation"})
    top_k: int = field(default=50, metadata={"help": "Value of k for top-k sampling"})
    top_p: float = field(default=1.0, metadata={"help": "Value of p for nucleus sampling"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty"})
    eos_tokens: str = field(
        default=None,
        metadata={"help": "EOS tokens to stop the generation. If multiple they should be comma separated"},
    )
    eos_token_ids: str = field(
        default=None,
        metadata={"help": "EOS token IDs to stop the generation. If multiple they should be comma separated"},
    )
    # model loading
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: str = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: str = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "use 8 bit precision for the base model - works only with LoRA"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "use 4 bit precision for the base model - works only with LoRA"},
    )

    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})


class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]`):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser

    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"

    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    @deprecate_kwarg(
        "ignore_extra_args",
        "0.14.0",
        warn_if_greater_or_equal_version=True,
        additional_message="Use the `return_remaining_strings` in the `parse_args_and_config` method instead.",
    )
    def __init__(
        self,
        dataclass_types: Union[DataClassType, Iterable[DataClassType]],
        ignore_extra_args: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(dataclass_types=dataclass_types, **kwargs)
        self._ignore_extra_args = ignore_extra_args

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

    def post_process_dataclasses(self, dataclasses):
        """
        Post process dataclasses to merge the TrainingArguments with the SFTScriptArguments or DPOScriptArguments.
        """
        warnings.warn(
            "The `post_process_dataclasses` method is deprecated and will be removed in version 0.14. "
            "It is no longer functional and can be safely removed from your code.",
            DeprecationWarning,
        )
        return dataclasses

    def parse_args_and_config(
        self, args: Optional[Iterable[str]] = None, return_remaining_strings: bool = False
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        if self._ignore_extra_args is not None:
            return_remaining_strings = not self._ignore_extra_args

        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments.

        Any argument with an updated default will also be marked as not required
        if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """
        # If an argument is in the kwargs, update its default and set it as not required
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs.pop(action.dest)
                action.required = False
        remaining_strings = [item for key, value in kwargs.items() for item in [f"--{key}", str(value)]]
        return remaining_strings


def get_git_commit_hash(package_name):
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"
