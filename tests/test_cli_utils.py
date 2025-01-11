# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from unittest.mock import mock_open, patch

from trl import TrlParser


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
