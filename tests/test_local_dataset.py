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

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from trl.scripts.utils import _infer_dataset_format, is_local_dataset, load_dataset_with_local_support


class TestLocalDataset(unittest.TestCase):
    def test_is_local_dataset_relative_paths(self):
        """Test detection of relative paths."""
        self.assertTrue(is_local_dataset("./data/train.json"))
        self.assertTrue(is_local_dataset("./my_dataset.csv"))
        self.assertTrue(is_local_dataset("./folder/subfolder/data.parquet"))

    def test_is_local_dataset_absolute_paths(self):
        """Test detection of absolute paths."""
        self.assertTrue(is_local_dataset("/home/user/data.json"))
        self.assertTrue(is_local_dataset("/data/train.csv"))
        self.assertTrue(is_local_dataset("/tmp/dataset.parquet"))

    def test_is_local_dataset_home_paths(self):
        """Test detection of home directory paths."""
        self.assertTrue(is_local_dataset("~/data/train.json"))
        self.assertTrue(is_local_dataset("~user/data.csv"))
        self.assertTrue(is_local_dataset("~/datasets/my_data.parquet"))

    def test_is_local_dataset_windows_paths(self):
        """Test detection of Windows drive letter paths."""
        self.assertTrue(is_local_dataset("C:\\data\\train.json"))
        self.assertTrue(is_local_dataset("D:\\datasets\\data.csv"))
        self.assertTrue(is_local_dataset("C:/data/train.json"))

    def test_is_local_dataset_huggingface_names(self):
        """Test that HuggingFace dataset names are not detected as local."""
        self.assertFalse(is_local_dataset("squad"))
        self.assertFalse(is_local_dataset("trl-lib/Capybara"))
        self.assertFalse(is_local_dataset("huggingface/dataset"))
        self.assertFalse(is_local_dataset("my-dataset"))
        self.assertFalse(is_local_dataset("organization/dataset-name"))

    def test_is_local_dataset_edge_cases(self):
        """Test edge cases that should not be detected as local."""
        # These might look like they could be local but don't match our patterns
        self.assertFalse(is_local_dataset("data.json"))  # No path prefix
        self.assertFalse(is_local_dataset("train.csv"))  # No path prefix
        self.assertFalse(is_local_dataset("folder/data.parquet"))  # No path prefix

    def test_infer_dataset_format_json(self):
        """Test format inference for JSON files."""
        self.assertEqual(_infer_dataset_format("data.json"), "json")
        self.assertEqual(_infer_dataset_format("./data/train.json"), "json")
        self.assertEqual(_infer_dataset_format("/home/user/data.JSON"), "json")

    def test_infer_dataset_format_csv(self):
        """Test format inference for CSV files."""
        self.assertEqual(_infer_dataset_format("data.csv"), "csv")
        self.assertEqual(_infer_dataset_format("./data/train.csv"), "csv")
        self.assertEqual(_infer_dataset_format("/home/user/data.CSV"), "csv")

    def test_infer_dataset_format_parquet(self):
        """Test format inference for Parquet files."""
        self.assertEqual(_infer_dataset_format("data.parquet"), "parquet")
        self.assertEqual(_infer_dataset_format("./data/train.parquet"), "parquet")
        self.assertEqual(_infer_dataset_format("/home/user/data.PARQUET"), "parquet")

    def test_infer_dataset_format_text(self):
        """Test format inference for text files."""
        self.assertEqual(_infer_dataset_format("data.txt"), "text")
        self.assertEqual(_infer_dataset_format("./data/train.txt"), "text")
        self.assertEqual(_infer_dataset_format("/home/user/data.TXT"), "text")

    def test_infer_dataset_format_jsonl(self):
        """Test format inference for JSONL files."""
        self.assertEqual(_infer_dataset_format("data.jsonl"), "json")
        self.assertEqual(_infer_dataset_format("./data/train.jsonl"), "json")
        self.assertEqual(_infer_dataset_format("/home/user/data.JSONL"), "json")

    def test_infer_dataset_format_unknown(self):
        """Test format inference for unknown file types."""
        self.assertIsNone(_infer_dataset_format("data.unknown"))
        self.assertIsNone(_infer_dataset_format("./data/train.xyz"))
        self.assertIsNone(_infer_dataset_format("/home/user/data"))

    def test_load_dataset_with_local_support_local_json(self):
        """Test loading a local JSON dataset."""
        # Create a temporary JSON file
        sample_data = [
            {"text": "This is a test sentence.", "label": 0},
            {"text": "Another test sentence.", "label": 1},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name

        try:
            # Test loading with local path
            dataset = load_dataset_with_local_support(temp_file)

            # Verify it's a dataset and contains our data
            self.assertIsNotNone(dataset)
            self.assertEqual(len(dataset["train"]), 2)
            self.assertEqual(dataset["train"][0]["text"], "This is a test sentence.")
            self.assertEqual(dataset["train"][1]["label"], 1)

        finally:
            os.unlink(temp_file)

    def test_load_dataset_with_local_support_local_csv(self):
        """Test loading a local CSV dataset."""
        # Create a temporary CSV file
        csv_content = "text,label\nThis is a test sentence.,0\nAnother test sentence.,1"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_file = f.name

        try:
            # Test loading with local path
            dataset = load_dataset_with_local_support(temp_file)

            # Verify it's a dataset and contains our data
            self.assertIsNotNone(dataset)
            self.assertEqual(len(dataset["train"]), 2)
            self.assertEqual(dataset["train"][0]["text"], "This is a test sentence.")
            self.assertEqual(dataset["train"][1]["label"], 1)  # CSV loads as integers for numeric values

        finally:
            os.unlink(temp_file)

    def test_load_dataset_with_local_support_file_not_found(self):
        """Test error handling when local file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            load_dataset_with_local_support("./nonexistent_file.json")

    def test_load_dataset_with_local_support_home_expansion(self):
        """Test that home directory paths are expanded."""
        # Create a temporary file in home directory
        home_dir = os.path.expanduser("~")
        temp_file = os.path.join(home_dir, "temp_dataset.json")

        sample_data = [{"text": "Home test", "label": 0}]

        with open(temp_file, "w") as f:
            json.dump(sample_data, f)

        try:
            # Test loading with ~ path
            dataset = load_dataset_with_local_support("~/temp_dataset.json")

            # Verify it's a dataset and contains our data
            self.assertIsNotNone(dataset)
            self.assertEqual(len(dataset["train"]), 1)
            self.assertEqual(dataset["train"][0]["text"], "Home test")

        finally:
            os.unlink(temp_file)

    @patch("trl.scripts.utils.load_dataset")
    def test_load_dataset_with_local_support_huggingface_dataset(self, mock_load_dataset):
        """Test that HuggingFace datasets are loaded normally."""
        # Mock the datasets.load_dataset function
        mock_load_dataset.return_value = {"train": [{"text": "HF data"}]}

        # Test loading HuggingFace dataset
        dataset = load_dataset_with_local_support("squad", name="v1.1", streaming=False)

        # Verify the mock was called with correct parameters
        mock_load_dataset.assert_called_once_with("squad", name="v1.1", streaming=False)
        self.assertEqual(dataset["train"][0]["text"], "HF data")

    def test_load_dataset_with_local_support_streaming(self):
        """Test that streaming parameter is passed correctly."""
        # Create a temporary JSON file
        sample_data = [{"text": "Streaming test", "label": 0}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name

        try:
            # Test loading with streaming=True
            dataset = load_dataset_with_local_support(temp_file, streaming=True)

            # Verify it's a dataset (streaming datasets work differently)
            self.assertIsNotNone(dataset)

        finally:
            os.unlink(temp_file)

    def test_load_dataset_with_local_support_unknown_format(self):
        """Test loading a local file with unknown format."""
        # Create a temporary file with unknown extension
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("unknown format content")
            temp_file = f.name

        try:
            # This should fail because datasets.load_dataset cannot handle unknown formats
            with self.assertRaises((ValueError, OSError)):
                load_dataset_with_local_support(temp_file)

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
