# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml
from datasets import load_dataset
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset name or path. Can be a HuggingFace dataset name or a local file path "
            "(starting with './', '/', '~', or Windows drive letter)."
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode."},
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]` or `None`, *optional*, defaults to `None`):
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

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self,
        args: Optional[Iterable[str]] = None,
        return_remaining_strings: bool = False,
        fail_with_unknown_args: bool = True,
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
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
        elif fail_with_unknown_args and config_remaining_strings:
            raise ValueError(
                f"Unknown arguments from config file: {config_remaining_strings}. Please remove them, add them to the "
                "dataclass, or set `fail_with_unknown_args=False`."
            )
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        def apply_defaults(parser, kw):
            used_keys = set()
            for action in parser._actions:
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    for subparser in action.choices.values():
                        used_keys.update(apply_defaults(subparser, kw))
                elif action.dest in kw:
                    action.default = kw[action.dest]
                    action.required = False
                    used_keys.add(action.dest)
            return used_keys

        used_keys = apply_defaults(self, kwargs)
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        return remaining


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


def is_local_dataset(dataset_name: str) -> bool:
    """
    Detect if dataset_name refers to a local file or directory.

    Only detects paths that are explicitly intended to be local:
    - Start with './' (current directory)
    - Start with '/' (absolute path on Unix)
    - Start with '~' (home directory)
    - Start with drive letter on Windows (e.g., 'C:')

    Args:
        dataset_name (`str`):
            Dataset name to check.

    Returns:
        `bool`: True if the dataset name appears to be a local path, False otherwise.

    Examples:
    ```python
    >>> is_local_dataset("./data/train.json")
    True
    >>> is_local_dataset("/home/user/data.csv")
    True
    >>> is_local_dataset("~/datasets/my_data.parquet")
    True
    >>> is_local_dataset("C:\\data\\train.json")  # Windows
    True
    >>> is_local_dataset("huggingface/dataset")
    False
    >>> is_local_dataset("my-dataset")
    False
    ```
    """
    if dataset_name.startswith(("./", "/", "~")):
        return True
    # Windows drive letter detection (C:, D:, etc.)
    if len(dataset_name) >= 2 and dataset_name[1] == ":" and dataset_name[0].isalpha():
        return True
    return False


def _infer_dataset_format(file_path: str) -> Optional[str]:
    """
    Infer the dataset format from the file extension.

    Args:
        file_path (`str`):
            Path to the dataset file.

    Returns:
        `Optional[str]`: The inferred format for datasets.load_dataset(), or None if unknown.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return "json"
    elif suffix == ".csv":
        return "csv"
    elif suffix == ".parquet":
        return "parquet"
    elif suffix == ".txt":
        return "text"
    elif suffix == ".jsonl":
        return "json"
    else:
        # For unknown extensions, let datasets.load_dataset() handle it
        return None


def load_dataset_with_local_support(dataset_name: str, name: Optional[str] = None, streaming: bool = False, **kwargs):
    """
    Load dataset with support for local files.

    This function extends the standard datasets.load_dataset() to support local files
    while maintaining full backward compatibility with HuggingFace datasets.

    Args:
        dataset_name (`str`):
            Dataset name or path. If it appears to be a local path (starts with './', '/', '~',
            or Windows drive letter), it will be loaded as a local dataset. Otherwise, it will
            be loaded as a HuggingFace dataset.
        name (`str`, *optional*):
            Dataset configuration name. Only used for HuggingFace datasets.
        streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset.
        **kwargs:
            Additional arguments passed to datasets.load_dataset().

    Returns:
        `Dataset` or `DatasetDict`: The loaded dataset.

    Examples:
    ```python
    >>> # Load local dataset
    >>> dataset = load_dataset_with_local_support("./data/train.json")

    >>> # Load HuggingFace dataset (unchanged behavior)
    >>> dataset = load_dataset_with_local_support("squad", name="v1.1")
    ```
    """
    if is_local_dataset(dataset_name):
        # Handle local dataset
        expanded_path = os.path.expanduser(dataset_name)

        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Local dataset file not found: {expanded_path}")

        # Infer format from file extension
        inferred_format = _infer_dataset_format(expanded_path)

        if inferred_format:
            return load_dataset(inferred_format, data_files=expanded_path, streaming=streaming, **kwargs)
        else:
            # Let datasets.load_dataset() try to infer the format
            return load_dataset(expanded_path, streaming=streaming, **kwargs)
    else:
        # Handle HuggingFace dataset (unchanged behavior)
        return load_dataset(dataset_name, name=name, streaming=streaming, **kwargs)
