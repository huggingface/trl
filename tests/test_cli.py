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


import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from trl.cli import main


class TestCLI(unittest.TestCase):
    def test_dpo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:  # Create a temporary directory
            command = f"trl dpo --output_dir {tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_preference --report_to none"
            with patch("sys.argv", command.split(" ")):
                main()

    @patch("sys.stdout", new_callable=StringIO)
    def test_env(self, mock_stdout):
        command = "trl env"
        with patch("sys.argv", command.split(" ")):
            main()
        self.assertIn("TRL version: ", mock_stdout.getvalue().strip())

    def test_kto(self):
        with tempfile.TemporaryDirectory() as tmp_dir:  # Create a temporary directory
            command = f"trl kto --output_dir {tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_unpaired_preference --report_to none"
            with patch("sys.argv", command.split(" ")):
                main()

    def test_sft(self):
        with tempfile.TemporaryDirectory() as tmp_dir:  # Create a temporary directory
            command = f"trl sft --output_dir {tmp_dir} --model_name_or_path trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --dataset_name trl-internal-testing/zen --dataset_config standard_language_modeling --report_to none"
            with patch("sys.argv", command.split(" ")):
                main()


if __name__ == "__main__":
    unittest.main()
