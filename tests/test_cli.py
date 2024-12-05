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
import unittest
from io import StringIO
from unittest.mock import patch

from trl.cli import main


class TestCLI(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    @patch("sys.argv", ["trl", "env"])
    def test_env(self, mock_stdout):
        main()
        self.assertIn("TRL version: ", mock_stdout.getvalue().strip())

    @patch(
        "sys.argv",
        [
            "trl",
            "dpo",
            "--output_dir",
            "output_dir",
            "--model_name_or_path",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "--dataset_name trl-internal-testing/zen",
            "--report_to none",
        ],
    )
    def test_dpo(self):
        main()


if __name__ == "__main__":
    unittest.main()
