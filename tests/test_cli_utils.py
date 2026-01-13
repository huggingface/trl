# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import tempfile
from dataclasses import dataclass
from unittest.mock import mock_open, patch

import pytest
from datasets import DatasetDict, load_dataset

from trl import DatasetMixtureConfig, TrlParser, get_dataset
from trl.scripts.utils import DatasetConfig

from .testing_utils import TrlTestCase


@dataclass
class MyDataclass:
    arg1: int
    arg2: str = "default"


@dataclass
class InvalidDataclass:
    config: str  # This should raise an error in the TrlParser


class TestTrlParser(TrlTestCase):
    def test_init_without_config_field(self):
        """Test initialization without 'config' field in the dataclasses."""
        parser = TrlParser(dataclass_types=[MyDataclass])
        assert isinstance(parser, TrlParser)

    def test_init_with_config_field(self):
        """Test initialization with a 'config' field in the dataclass (should raise ValueError)."""
        with pytest.raises(ValueError, match="has a field named 'config'"):
            TrlParser(dataclass_types=[InvalidDataclass])

    @patch("builtins.open", mock_open(read_data="env:\n VAR1: value1\n VAR2: value2\narg1: 2"))
    @patch("yaml.safe_load")
    @patch("os.environ", new_callable=dict)  # Mock os.environ as a dictionary
    def test_parse_args_and_config_with_valid_config(self, mock_environ, mock_yaml_load):
        """Test parse_args_and_config method with valid arguments and config."""
        mock_yaml_load.return_value = {"env": {"VAR1": "value1", "VAR2": "value2"}, "arg1": 2}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg2", "value", "--config", "config.yaml"]  # don't set arg1 to test default value

        # Simulate the config being loaded and environment variables being set
        result_args = parser.parse_args_and_config(args)

        # Set the environment variables using the mock
        mock_environ["VAR1"] = "value1"
        mock_environ["VAR2"] = "value2"

        # Ensure that the environment variables were set correctly
        assert mock_environ.get("VAR1") == "value1"
        assert mock_environ.get("VAR2") == "value2"

        # Check the parsed arguments
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[0].arg2 == "value"

    @patch("builtins.open", mock_open(read_data="arg1: 2"))
    @patch("yaml.safe_load")
    def test_parse_args_and_arg_override_config(self, mock_yaml_load):
        """Test parse_args_and_config method and check that arguments override the config."""
        mock_yaml_load.return_value = {"arg1": 2}  # this arg is meant to be overridden

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "3", "--config", "config.yaml"]  # override arg1 default with 3

        # Simulate the config being loaded and arguments being passed
        result_args = parser.parse_args_and_config(args)

        # Check the parsed arguments
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 3

    @patch("builtins.open", mock_open(read_data="env: not_a_dict"))
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_invalid_env(self, mock_yaml_load):
        """Test parse_args_and_config method when the 'env' field is not a dictionary."""
        mock_yaml_load.return_value = {"env": "not_a_dict"}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value", "--config", "config.yaml"]

        with pytest.raises(ValueError, match="`env` field should be a dict in the YAML file."):
            parser.parse_args_and_config(args)

    def test_parse_args_and_config_without_config(self):
        """Test parse_args_and_config without the `--config` argument."""
        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value"]

        # Simulate no config, just parse args normally
        result_args = parser.parse_args_and_config(args)

        # Check that the arguments are parsed as is
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[0].arg2 == "value"

    def test_set_defaults_with_config(self):
        """Test set_defaults_with_config updates the defaults."""
        parser = TrlParser(dataclass_types=[MyDataclass])

        # Update defaults
        parser.set_defaults_with_config(arg1=42)

        # Ensure the default value is updated
        result_args = parser.parse_args_and_config([])
        assert len(result_args) == 1
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 42

    def test_parse_args_and_config_with_remaining_strings(self):
        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value", "remaining"]

        # Simulate no config, just parse args normally
        result_args = parser.parse_args_and_config(args, return_remaining_strings=True)

        # Check that the arguments are parsed as is
        assert len(result_args) == 2
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[0].arg2 == "value"
        assert result_args[1] == ["remaining"]

    @patch("builtins.open", mock_open(read_data="remaining_string_in_config: abc"))
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_remaining_strings_in_config_and_args(self, mock_yaml_load):
        mock_yaml_load.return_value = {"remaining_string_in_config": "abc"}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--remaining_string_in_args", "def", "--config", "config.yaml"]

        # Simulate the config being loaded and arguments being passed
        result_args = parser.parse_args_and_config(args, return_remaining_strings=True)

        # Check that the arguments are parsed as is
        assert len(result_args) == 2
        assert isinstance(result_args[0], MyDataclass)
        assert result_args[0].arg1 == 2
        assert result_args[1] == ["--remaining_string_in_config", "abc", "--remaining_string_in_args", "def"]

    @patch("builtins.open", mock_open(read_data="arg1: 2\narg2: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_with_config_defaults(self, mock_yaml_load):
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "arg2": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand", dataclass_types=[MyDataclass])

        # Parse with config file
        args = ["subcommand", "--config", "config.yaml"]
        result_args = parser.parse_args_and_config(args)

        # Check main parser arguments
        assert len(result_args) == 1

        # Check that config values were applied to the subparser
        assert result_args[0].arg1 == 2  # Default from config
        assert result_args[0].arg2 == "config_value"  # Default from config

    @patch("builtins.open", mock_open(read_data="arg1: 2\narg2: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_with_config_defaults_and_arg_override(self, mock_yaml_load):
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "arg2": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand", dataclass_types=[MyDataclass])

        # Test with command line arguments overriding config
        args = ["subcommand", "--arg1", "3", "--config", "config.yaml"]
        result_args = parser.parse_args_and_config(args)

        # Command line arguments should override config
        assert result_args[0].arg1 == 3
        assert result_args[0].arg2 == "config_value"  # Still from config

    @patch("builtins.open", mock_open(read_data="arg1: 2\nthis_arg_does_not_exist: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_with_config_defaults_and_arg_override_wrong_name(self, mock_yaml_load):
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "this_arg_does_not_exist": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand", dataclass_types=[MyDataclass])

        # Test with command line arguments overriding config
        args = ["subcommand", "--arg1", "3", "--config", "config.yaml"]
        with pytest.raises(ValueError):
            parser.parse_args_and_config(args)

        parser.parse_args_and_config(args, fail_with_unknown_args=False)

    @patch("builtins.open", mock_open(read_data="arg1: 2\narg2: config_value"))
    @patch("yaml.safe_load")
    def test_subparsers_multiple_with_config_defaults(self, mock_yaml_load):
        """Test that config defaults are applied to all subparsers."""
        mock_yaml_load.return_value = {"arg1": 2, "arg2": "config_value"}

        # Create the main parser
        parser = TrlParser()

        # Add subparsers
        subparsers = parser.add_subparsers(dest="command", parser_class=TrlParser)

        # Create a subparser for a specific command
        subparsers.add_parser("subcommand0", dataclass_types=[MyDataclass])
        subparsers.add_parser("subcommand1", dataclass_types=[MyDataclass])

        for idx in range(2):
            # Parse with config file
            args = [f"subcommand{idx}", "--config", "config.yaml"]
            result_args = parser.parse_args_and_config(args)

            # Check main parser arguments
            assert len(result_args) == 1

            # Check that config values were applied to the subparser
            assert result_args[0].arg1 == 2  # Default from config
            assert result_args[0].arg2 == "config_value"  # Default from config


class TestGetDataset:
    def test_single_dataset_with_config(self):
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling")]
        )
        result = get_dataset(mixture_config)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")
        assert expected["train"][:] == result["train"][:]

    def test_single_dataset_preference_config(self):
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_preference")]
        )
        result = get_dataset(mixture_config)
        expected = load_dataset("trl-internal-testing/zen", "standard_preference")
        assert expected["train"][:] == result["train"][:]

    def test_single_dataset_streaming(self):
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling")],
            streaming=True,
        )
        result = get_dataset(mixture_config)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")
        assert expected["train"].to_list() == list(result["train"])

    def test_dataset_mixture_basic(self):
        dataset_config1 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_prompt_completion", split="train", columns=["prompt"]
        )
        dataset_config2 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_preference", split="train", columns=["prompt"]
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2])
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        train_dataset = result["train"]
        assert train_dataset.column_names == ["prompt"]
        prompts = train_dataset["prompt"]
        expected_first_half = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        assert prompts[: len(prompts) // 2] == expected_first_half["prompt"]
        expected_second_half = load_dataset("trl-internal-testing/zen", "standard_prompt_completion", split="train")
        assert prompts[len(prompts) // 2 :] == expected_second_half["prompt"]

    def test_dataset_mixture_with_weights(self):
        dataset_config1 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_prompt_completion", split="train[:50%]", columns=["prompt"]
        )
        dataset_config2 = DatasetConfig(
            path="trl-internal-testing/zen", name="standard_preference", split="train[:50%]", columns=["prompt"]
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2])
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        train_dataset = result["train"]
        assert train_dataset.column_names == ["prompt"]
        prompts = train_dataset["prompt"]
        expected_first_half = load_dataset("trl-internal-testing/zen", "standard_preference", split="train[:50%]")
        assert prompts[: len(prompts) // 2] == expected_first_half["prompt"]
        expected_second_half = load_dataset(
            "trl-internal-testing/zen", "standard_prompt_completion", split="train[:50%]"
        )
        assert prompts[len(prompts) // 2 :] == expected_second_half["prompt"]

    def test_dataset_mixture_with_test_split(self):
        mixture_config = DatasetMixtureConfig(
            datasets=[DatasetConfig(path="trl-internal-testing/zen", name="standard_language_modeling")],
            test_split_size=2,
        )
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "test" in result
        assert len(result["train"]) == 15
        assert len(result["test"]) == 2

    def test_empty_dataset_mixture_raises_error(self):
        mixture_config = DatasetMixtureConfig(datasets=[])

        with pytest.raises(ValueError, match="No datasets were loaded"):
            get_dataset(mixture_config)

    def test_mixture_multiple_different_configs(self):
        dataset_config1 = DatasetConfig(
            path="trl-internal-testing/zen", name="conversational_preference", split="train", columns=["prompt"]
        )
        dataset_config2 = DatasetConfig(
            path="trl-internal-testing/zen", name="conversational_prompt_only", split="test"
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2])
        result = get_dataset(mixture_config)
        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert len(result["train"]) > 0

    def test_trlparser_parses_yaml_config_correctly(self):
        # Prepare YAML content exactly like your example
        # docstyle-ignore
        yaml_content = """
        datasets:
        - path: trl-internal-testing/zen
          name: standard_prompt_only
        - path: trl-internal-testing/zen
          name: standard_preference
          columns:
          - prompt
        """

        # Write YAML to a temporary file
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as tmpfile:
            tmpfile.write(yaml_content)
            tmpfile.flush()
            parser = TrlParser((DatasetMixtureConfig,))
            args = parser.parse_args_and_config(args=["--config", tmpfile.name])[0]

        # Assert that we got DatasetMixtureConfig instance
        assert isinstance(args, DatasetMixtureConfig)

        # Assert datasets list length
        assert len(args.datasets) == 2

        # Check first dataset
        dataset_config1 = args.datasets[0]
        assert isinstance(dataset_config1, DatasetConfig)
        assert dataset_config1.path == "trl-internal-testing/zen"
        assert dataset_config1.name == "standard_prompt_only"
        assert dataset_config1.columns is None  # No columns specified

        # Check second dataset
        dataset_config2 = args.datasets[1]
        assert isinstance(dataset_config2, DatasetConfig)
        assert dataset_config2.path == "trl-internal-testing/zen"
        assert dataset_config2.name == "standard_preference"
        assert dataset_config2.columns == ["prompt"]  # Columns specified

    def test_trlparser_parses_yaml_and_loads_dataset(self):
        # Prepare YAML content exactly like your example
        # docstyle-ignore
        yaml_content = """
        datasets:
        - path: trl-internal-testing/zen
          name: standard_language_modeling
        """

        # Write YAML to a temporary file
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml") as tmpfile:
            tmpfile.write(yaml_content)
            tmpfile.flush()
            parser = TrlParser((DatasetMixtureConfig,))
            args = parser.parse_args_and_config(args=["--config", tmpfile.name])[0]

        # Load the dataset using get_dataset
        result = get_dataset(args)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")
        assert expected["train"][:] == result["train"][:]
