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
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import datasets
import yaml
from datasets import DatasetDict, concatenate_datasets
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """

    Args:

        path (`str`):
            Path or name of the dataset.
        name (`str`, *optional*, defaults to `None`):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*, defaults to `None`):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders(csv, text
            etc.) or the Hub datasets and `data_files` is `None`, the behavior is equal to passing
            `os.path.join(data_dir, **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*, defaults to `None`):
            Path(s) to source data file(s).
        split (`str`, *optional*, defaults to `"train"`):
            Which split of the data to load.
        columns (`list[str]`, *optional*, defaults to `None`):
            List of column names to select from the dataset. If `None`, all columns are selected.
    """

    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Union[str, list[str], dict[str, str]]] = None
    split: str = "train"
    columns: Optional[list[str]] = None


@dataclass
class DatasetMixtureConfig:
    """
    Configuration for a mixture of datasets.

    Args:
        datasets (`list[DatasetConfig]`):
            List of dataset configurations to include in the mixture.
        streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the datasets. If `True`, the datasets will be loaded in streaming mode.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Seed for random operations, such as splitting the dataset into train and test sets.
        test_split_size (`float` or `None`, *optional*, defaults to `None`):
            Size of the test split. Refer to the `test_size` parameter in the [`~datasets.train_test_split`] function
            for more details. If `None`, the dataset will not be split into train and test sets.
    """

    datasets: list[DatasetConfig]
    streaming: bool = False
    seed: Optional[int] = None
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`, or `None`, *optional*, defaults to `None`):
            Path or name of the dataset to load. If `dataset_mixture` is provided, this will be ignored.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
            If `dataset_mixture` is provided, this will be ignored.
        dataset_data_dir (`str` or `None`, *optional*, defaults to `None`):
            Dataset data directory. Corresponds to the `data_dir` argument of the [`~datasets.load_dataset`] function.
            If `dataset_mixture` is provided, this will be ignored.
        dataset_data_files (`str`, `list[str]`, or `dict[str, str]`, *optional*, defaults to `None`):
            Path(s) to source data file(s). Corresponds to the `data_files` argument of the [`~datasets.load_dataset`]
            function. If `dataset_mixture` is provided, this will be ignored.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training. If `dataset_mixture` is provided, this will be ignored.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation. If `dataset_mixture` is provided, this will be ignored.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If `dataset_mixture`
            is provided, this will be ignored.
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
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
            "help": "Path or name of the dataset to load. If `dataset_mixture` is provided, this will be ignored."
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function. If `dataset_mixture` is provided, this will be ignored."
        },
    )
    dataset_train_split: str = field(
        default="train",
        metadata={
            "help": "Dataset split to use for training. If `dataset_mixture` is provided, this will be ignored."
        },
    )
    dataset_test_split: str = field(
        default="test",
        metadata={
            "help": "Dataset split to use for evaluation. If `dataset_mixture` is provided, this will be ignored."
        },
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If "
            "`dataset_mixture` is provided, this will be ignored."
        },
    )
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures."},
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

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_name is not None and self.dataset_mixture is not None:
            raise ValueError("Cannot provide both `dataset_name` and `dataset_mixture`. Please specify only one.")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging

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


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """
    Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (`ScriptArguments`):
            Script arguments containing dataset configuration.

    Returns:
        `DatasetDict`:
            The loaded datasets.

    Example:
    ```python
    >>> from trl import get_dataset, ScriptArguments
    ```
    """
    if args.dataset_name and args.dataset_mixture:
        logger.warning(
            "Both `dataset_path` and `dataset_mixture` are provided. `dataset_mixture` will be used for loading "
            "datasets and `dataset_path` will be ignored.",
        )
    elif args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config)
    elif args.dataset_mixture and not args.dataset_name:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        datasets_list = []
        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.path} (config name: {dataset_config.name})")
            dataset = datasets.load_dataset(
                path=dataset_config.path,
                name=dataset_config.name,
                data_dir=dataset_config.data_dir,
                data_files=dataset_config.data_files,
                split=dataset_config.split,
                streaming=args.dataset_mixture.streaming,
            )
            if dataset_config.columns is not None:
                dataset = dataset.select_columns(dataset_config.columns)
            datasets_list.append(dataset)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                logger.info(
                    f"Spliting dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=args.dataset_mixture.seed
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
