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

import unittest
from dataclasses import asdict, dataclass
from unittest.mock import mock_open, patch

from datasets import DatasetDict, load_dataset

from trl import DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset
from trl.scripts.utils import DatasetConfig


@dataclass
class MyDataclass:
    arg1: int
    arg2: str = "default"


@dataclass
class InvalidDataclass:
    config: str  # This should raise an error in the TrlParser


class TestTrlParser(unittest.TestCase):
    def test_init_without_config_field(self):
        """Test initialization without 'config' field in the dataclasses."""
        parser = TrlParser(dataclass_types=[MyDataclass])
        self.assertIsInstance(parser, TrlParser)

    def test_init_with_config_field(self):
        """Test initialization with a 'config' field in the dataclass (should raise ValueError)."""
        with self.assertRaises(ValueError) as context:
            TrlParser(dataclass_types=[InvalidDataclass])
        self.assertTrue("has a field named 'config'" in str(context.exception))

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
        self.assertEqual(mock_environ.get("VAR1"), "value1")
        self.assertEqual(mock_environ.get("VAR2"), "value2")

        # Check the parsed arguments
        self.assertEqual(len(result_args), 1)
        self.assertIsInstance(result_args[0], MyDataclass)
        self.assertEqual(result_args[0].arg1, 2)
        self.assertEqual(result_args[0].arg2, "value")

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
        self.assertEqual(len(result_args), 1)
        self.assertIsInstance(result_args[0], MyDataclass)
        self.assertEqual(result_args[0].arg1, 3)

    @patch("builtins.open", mock_open(read_data="env: not_a_dict"))
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_invalid_env(self, mock_yaml_load):
        """Test parse_args_and_config method when the 'env' field is not a dictionary."""
        mock_yaml_load.return_value = {"env": "not_a_dict"}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value", "--config", "config.yaml"]

        with self.assertRaises(ValueError) as context:
            parser.parse_args_and_config(args)

        self.assertEqual(str(context.exception), "`env` field should be a dict in the YAML file.")

    def test_parse_args_and_config_without_config(self):
        """Test parse_args_and_config without the `--config` argument."""
        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value"]

        # Simulate no config, just parse args normally
        result_args = parser.parse_args_and_config(args)

        # Check that the arguments are parsed as is
        self.assertEqual(len(result_args), 1)
        self.assertIsInstance(result_args[0], MyDataclass)
        self.assertEqual(result_args[0].arg1, 2)
        self.assertEqual(result_args[0].arg2, "value")

    def test_set_defaults_with_config(self):
        """Test set_defaults_with_config updates the defaults."""
        parser = TrlParser(dataclass_types=[MyDataclass])

        # Update defaults
        parser.set_defaults_with_config(arg1=42)

        # Ensure the default value is updated
        result_args = parser.parse_args_and_config([])
        self.assertEqual(len(result_args), 1)
        self.assertIsInstance(result_args[0], MyDataclass)
        self.assertEqual(result_args[0].arg1, 42)

    def test_parse_args_and_config_with_remaining_strings(self):
        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--arg2", "value", "remaining"]

        # Simulate no config, just parse args normally
        result_args = parser.parse_args_and_config(args, return_remaining_strings=True)

        # Check that the arguments are parsed as is
        self.assertEqual(len(result_args), 2)
        self.assertIsInstance(result_args[0], MyDataclass)
        self.assertEqual(result_args[0].arg1, 2)
        self.assertEqual(result_args[0].arg2, "value")
        self.assertEqual(result_args[1], ["remaining"])

    @patch("builtins.open", mock_open(read_data="remaining_string_in_config: abc"))
    @patch("yaml.safe_load")
    def test_parse_args_and_config_with_remaining_strings_in_config_and_args(self, mock_yaml_load):
        mock_yaml_load.return_value = {"remaining_string_in_config": "abc"}

        parser = TrlParser(dataclass_types=[MyDataclass])

        args = ["--arg1", "2", "--remaining_string_in_args", "def", "--config", "config.yaml"]

        # Simulate the config being loaded and arguments being passed
        result_args = parser.parse_args_and_config(args, return_remaining_strings=True)

        # Check that the arguments are parsed as is
        self.assertEqual(len(result_args), 2)
        self.assertIsInstance(result_args[0], MyDataclass)
        self.assertEqual(result_args[0].arg1, 2)
        self.assertEqual(result_args[1], ["--remaining_string_in_config", "abc", "--remaining_string_in_args", "def"])

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
        self.assertEqual(len(result_args), 1)

        # Check that config values were applied to the subparser
        self.assertEqual(result_args[0].arg1, 2)  # Default from config
        self.assertEqual(result_args[0].arg2, "config_value")  # Default from config

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
        self.assertEqual(result_args[0].arg1, 3)
        self.assertEqual(result_args[0].arg2, "config_value")  # Still from config

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
        with self.assertRaises(ValueError):
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
            self.assertEqual(len(result_args), 1)

            # Check that config values were applied to the subparser
            self.assertEqual(result_args[0].arg1, 2)  # Default from config
            self.assertEqual(result_args[0].arg2, "config_value")  # Default from config


class TestGetDataset(unittest.TestCase):
    def test_single_dataset_with_config(self):
        args = ScriptArguments(
            dataset_name="trl-internal-testing/zen", dataset_config="standard_language_modeling", dataset_mixture=None
        )

        result = get_dataset(args)
        expected = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        self.assertEqual(expected["train"][:], result["train"][:])
        self.assertEqual(expected["test"][:], result["test"][:])

    def test_single_dataset_preference_config(self):
        args = ScriptArguments(
            dataset_name="trl-internal-testing/zen", dataset_config="standard_preference", dataset_mixture=None
        )

        result = get_dataset(args)
        expected = load_dataset("trl-internal-testing/zen", "standard_preference")

        self.assertEqual(expected["train"][:], result["train"][:])
        self.assertEqual(expected["test"][:], result["test"][:])

    def test_dataset_mixture_basic(self):
        dataset_config1 = DatasetConfig(
            id="trl-internal-testing/zen", config="standard_language_modeling", split="train"
        )
        dataset_config2 = DatasetConfig(id="trl-internal-testing/zen", config="standard_preference", split="train")
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2], seed=42)

        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(mixture_config))

        result = get_dataset(args)

        self.assertIsInstance(result, DatasetDict)
        self.assertIn("train", result)
        self.assertGreater(len(result["train"]), 0)

    def test_dataset_mixture_with_columns(self):
        dataset_config = DatasetConfig(
            id="trl-internal-testing/zen", config="standard_language_modeling", split="train", columns=["text"]
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config], seed=42)

        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(mixture_config))

        result = get_dataset(args)

        self.assertIsInstance(result, DatasetDict)
        self.assertIn("train", result)
        self.assertGreater(len(result["train"]), 0)

        # Check that only specified column is present
        sample = result["train"][0]
        self.assertIn("text", sample)

    def test_dataset_mixture_with_weights(self):
        dataset_config = DatasetConfig(
            id="trl-internal-testing/zen", config="standard_language_modeling", split="train", weight=0.5
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config], seed=42)

        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(mixture_config))

        result = get_dataset(args)

        self.assertIsInstance(result, DatasetDict)
        self.assertIn("train", result)

        # Load the original dataset to compare sizes
        original_args = ScriptArguments(
            dataset_name="trl-internal-testing/zen", dataset_config="standard_language_modeling", dataset_mixture=None
        )
        original_result = get_dataset(original_args)

        # With weight=0.5, result should have roughly half the examples
        original_size = len(original_result["train"])
        weighted_size = len(result["train"])
        expected_size = int(original_size * 0.5)

        self.assertEqual(weighted_size, expected_size)

    def test_dataset_mixture_with_test_split(self):
        dataset_config = DatasetConfig(
            id="trl-internal-testing/zen", config="standard_language_modeling", split="train"
        )
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config], seed=42, test_split_size=2)

        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(mixture_config))

        result = get_dataset(args)

        self.assertIsInstance(result, DatasetDict)
        self.assertIn("train", result)
        self.assertIn("test", result)
        self.assertGreater(len(result["train"]), 0)
        self.assertGreater(len(result["test"]), 0)
        self.assertEqual(len(result["test"]), 2)

    def test_empty_dataset_mixture_raises_error(self):
        mixture_config = DatasetMixtureConfig(datasets=[], seed=42)
        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(mixture_config))

        with self.assertRaises(ValueError) as context:
            get_dataset(args)

        self.assertIn("No datasets were loaded", str(context.exception))

    def test_no_dataset_name_or_mixture_raises_error(self):
        with self.assertRaises(ValueError) as context:
            ScriptArguments(dataset_name=None, dataset_mixture=None)

        self.assertIn("Either `dataset_name` or `dataset_mixture` must be provided", str(context.exception))

    def test_mixture_multiple_different_configs(self):
        dataset_config1 = DatasetConfig(
            id="trl-internal-testing/zen", config="standard_language_modeling", split="train"
        )
        dataset_config2 = DatasetConfig(id="trl-internal-testing/zen", config="standard_prompt_only", split="train")
        mixture_config = DatasetMixtureConfig(datasets=[dataset_config1, dataset_config2], seed=42)

        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(mixture_config))

        result = get_dataset(args)

        self.assertIsInstance(result, DatasetDict)
        self.assertIn("train", result)
        self.assertGreater(len(result["train"]), 0)
