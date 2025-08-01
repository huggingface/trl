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

import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import load_dataset
from parameterized import parameterized
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.testing_utils import require_liger_kernel, require_peft, require_vision
from transformers.utils import is_peft_available

from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import (
    RepeatSampler,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    truncate_with_protected_tokens,
    unsplit_pixel_values_by_grid,
)

from .testing_utils import require_vllm


if is_peft_available():
    from peft import LoraConfig, PeftModel


class SplitTensorDictTester(unittest.TestCase):
    def test_split_equal_chunks(self):
        x = torch.arange(12).reshape(6, 2)
        y = torch.arange(6).reshape(6, 1)
        tensor_dict = {"x": x, "y": y}

        result = split_tensor_dict(tensor_dict, 3)

        expected_x_chunks = torch.chunk(x, 3, dim=0)
        expected_y_chunks = torch.chunk(y, 3, dim=0)
        self.assertEqual(len(result), 3)
        for i in range(3):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertTrue(torch.equal(result[i]["y"], expected_y_chunks[i]))

    def test_with_none_tensor(self):
        x = torch.arange(12).reshape(6, 2)
        tensor_dict = {"x": x, "y": None}

        result = split_tensor_dict(tensor_dict, 2)

        expected_x_chunks = torch.chunk(x, 2, dim=0)
        self.assertEqual(len(result), 2)
        for i in range(2):
            self.assertTrue(torch.equal(result[i]["x"], expected_x_chunks[i]))
            self.assertIsNone(result[i]["y"])


class ShuffleSequenceDictTester(unittest.TestCase):
    def test_shuffle_preserves_shape(self):
        x = torch.arange(6).reshape(3, 2)
        y = torch.arange(3).reshape(3, 1)
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_sequence_dict(tensor_dict)

        self.assertEqual(shuffled["x"].shape, x.shape)
        self.assertEqual(shuffled["y"].shape, y.shape)

    def test_shuffle_consistent_across_tensors(self):
        # Use known patterns to check alignment
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = torch.tensor([[1], [2], [3]])
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_sequence_dict(tensor_dict)

        # Build a reverse map from shuffled x rows to y values
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i].item()

            if torch.equal(x_row, torch.tensor([10, 11])):
                self.assertEqual(y_val, 1)
            elif torch.equal(x_row, torch.tensor([20, 21])):
                self.assertEqual(y_val, 2)
            elif torch.equal(x_row, torch.tensor([30, 31])):
                self.assertEqual(y_val, 3)
            else:
                self.fail("Unexpected x row in shuffled output.")

    def test_none_tensor_remains_none(self):
        x = torch.arange(6).reshape(3, 2)
        tensor_dict = {"x": x.clone(), "y": None}

        shuffled = shuffle_sequence_dict(tensor_dict)

        self.assertIsNone(shuffled["y"])
        self.assertEqual(shuffled["x"].shape, x.shape)

    def test_shuffle_with_list(self):
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = ["a", "b", "c"]

        sequence_dict = {"x": x.clone(), "y": y}

        shuffled = shuffle_sequence_dict(sequence_dict)

        # Check that the list y is shuffled in the same order as x
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i]

            if torch.equal(x_row, torch.tensor([10, 11])):
                self.assertEqual(y_val, "a")
            elif torch.equal(x_row, torch.tensor([20, 21])):
                self.assertEqual(y_val, "b")
            elif torch.equal(x_row, torch.tensor([30, 31])):
                self.assertEqual(y_val, "c")
            else:
                self.fail("Unexpected x row in shuffled output.")


class RepeatRandomSamplerTester(unittest.TestCase):
    def test_sampler(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 1, 1, 2, 2, 6, 6, 5, 5]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated twice
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))

    def test_sampler_no_shuffle(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, shuffle=False)
        sampled = list(sampler)
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        self.assertEqual(sampled, expected)

    def test_sampler_no_repeat(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1)
        # Should output something like [4, 3, 0, 1, 2, 6, 5]
        sampled = list(sampler)
        # Check that the length is the same
        assert len(sampled) == len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))

    def test_sampler_with_batch_size(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g", "h"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6, 5, 7, 5, 7]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_batch_size_and_drop(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * (
            len(dataset) - 1
        )  # one element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_mini_repeat_count_and_batch_size_1(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=3, repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 4, 4, 3, 3, 0, 0,
        #                               1, 1, 2, 2, 6, 6, 1, 1, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is quadrupled
        assert len(sampled) == 4 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]

    def test_sampler_with_mini_repeat_count_and_batch_size_2(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=3, batch_size=2, repeat_count=2)
        # Should output something like [4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 3, 3,
        #                               0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        #                               2, 2, 2, 6, 6, 6, 2, 2, 2, 6, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] == sampled[i + 2] for i in range(0, len(sampled), 3))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]
        assert sampled[24:30] == sampled[30:36]

    def test_sampler_with_mini_repeat_count_and_batch_size_3(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=2, repeat_count=3)
        # Should output something like [4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3,
        #                               0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        #                               2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:4] == sampled[4:8] == sampled[8:12]
        assert sampled[12:16] == sampled[16:20] == sampled[20:24]
        assert sampled[24:28] == sampled[28:32] == sampled[32:36]


class TruncateWithProtectedTokensTester(unittest.TestCase):
    def test_basic_example(self):
        """Test the basic example from the problem description."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2, 3, 6]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[2, 3, 5], [6, 9, 10]])
        expected_mask = torch.ones_like(expected_ids)

        self.assertTrue(torch.equal(new_ids, expected_ids))
        self.assertTrue(torch.equal(new_mask, expected_mask))

    def test_no_truncation_needed(self):
        """Test when target length equals current length."""
        prompt_ids = torch.tensor([[1, 2, 3]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        self.assertTrue(torch.equal(new_ids, prompt_ids))
        self.assertTrue(torch.equal(new_mask, prompt_mask))

    def test_no_protected_tokens(self):
        """Test truncation with no protected tokens (normal right truncation)."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = []
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[3, 4, 5]])  # Last 3 tokens
        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_all_tokens_protected(self):
        """Test when all remaining tokens are protected."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [3, 4, 5]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[3, 4, 5]])
        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_too_many_protected_tokens(self):
        """Test error when too many protected tokens for target length."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [1, 2, 3, 4]
        target_length = 3

        with self.assertRaises(ValueError):
            truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

    def test_single_batch_single_token(self):
        """Test edge case with single batch and single token."""
        prompt_ids = torch.tensor([[5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [5]
        target_length = 1

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        self.assertTrue(torch.equal(new_ids, prompt_ids))

    def test_mask_preservation(self):
        """Test that mask values are correctly preserved."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.tensor([[1, 0, 1, 0, 1]])  # Mixed mask values
        protected_tokens = [2, 4]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[2, 4, 5]])
        expected_mask = torch.tensor([[0, 0, 1]])  # Corresponding mask values

        self.assertTrue(torch.equal(new_ids, expected_ids))
        self.assertTrue(torch.equal(new_mask, expected_mask))

    def test_multiple_batches_different_protected(self):
        """Test multiple batches where protected tokens appear differently."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5], [2, 6, 7, 8, 9], [10, 11, 12, 2, 13]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2]
        target_length = 3

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor(
            [
                [2, 4, 5],  # 2 is protected, keep last 2 non-protected (4,5)
                [2, 8, 9],  # 2 is protected, keep last 2 non-protected (8,9)
                [12, 2, 13],  # 2 is protected, keep last 2 non-protected (12,13)
            ]
        )

        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_order_preservation(self):
        """Test that relative order is preserved."""
        prompt_ids = torch.tensor([[10, 2, 20, 3, 30, 40]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = [2, 3]
        target_length = 4

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        # Should keep protected tokens 2,3 and last 2 non-protected tokens 30,40
        # Order should be: 2, 3, 30, 40 (maintaining original relative positions)
        expected_ids = torch.tensor([[2, 3, 30, 40]])

        self.assertTrue(torch.equal(new_ids, expected_ids))

    def test_empty_protected_tokens_list(self):
        """Test with empty protected tokens list."""
        prompt_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_mask = torch.ones_like(prompt_ids)
        protected_tokens = []
        target_length = 2

        new_ids, new_mask = truncate_with_protected_tokens(prompt_ids, prompt_mask, target_length, protected_tokens)

        expected_ids = torch.tensor([[4, 5]])  # Last 2 tokens
        self.assertTrue(torch.equal(new_ids, expected_ids))


class GetHighEntropyMaskTester(unittest.TestCase):
    def get_high_entropy_mask(self, entropies, mask, threshold):
        """Helper method to test the get_high_entropy_mask functionality."""
        # Create a mock trainer with minimal setup
        from unittest.mock import Mock

        # Create a mock accelerator
        mock_accelerator = Mock()
        mock_accelerator.num_processes = 1  # Single process for testing

        # Create a minimal trainer instance just to access the method
        trainer = Mock(spec=GRPOTrainer)
        trainer.accelerator = mock_accelerator
        trainer.accelerator.gather = lambda x: x  # Mock gather to return the input directly

        # Call the actual method from GRPOTrainer
        return GRPOTrainer.get_high_entropy_mask(trainer, entropies, mask, threshold)

    def test_compute_entropy_mask_0(self):
        # We have a total of 12 tokens out of which 10 are non-pad.
        # for a top_entropy_quantile of 0.8, we expect the top 20% i.e 2 non-pad tokens corresponding to
        # the highest entropy to be unmasked.
        # In our example these will be the tokens corresponding to the entropies 0.9 and 1.0 since 1.1 and 1.2 are pad
        # tokens they are excluded from the entropy threshold calculation.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.8)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_mask_1(self):
        # Another example with a different set of entropies and a different mask.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 1.4, 0.5, 0.14], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.8)
        expected_mask = torch.tensor([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_mask_lower_threshold(self):
        # For a threshold of 0.5 we expect the top half of the non-pad tokens to be unmasked.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.5)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_threshold_0(self):
        # If the threshold is 0.0 then we expect the mask to be all ones for non-pad tokens.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.0)
        expected_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_threshold_1(self):
        # If the threshold is 1.0 then we expect the mask to be all zeros BUT ONE VALUE.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=1.0)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)

    def test_compute_entropy_all_masked(self):
        # If there are no non-pad tokens we expect the mask to be all zeros.
        entropies = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        entropy_mask = self.get_high_entropy_mask(entropies, mask, threshold=0.5)
        expected_mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        torch.testing.assert_close(entropy_mask, expected_mask)


class SplitPixelValuesByGridTester(unittest.TestCase):
    def test_split_correctly_0(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 4]]),  # Products: [4, 4]
            "pixel_values": torch.arange(8 * 3).reshape(8, 3),  # Shape: [8, 3]
        }
        result = split_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], list)
        self.assertEqual(len(result["pixel_values"]), 2)
        self.assertTrue(torch.equal(result["pixel_values"][0], batch["pixel_values"][:4]))
        self.assertTrue(torch.equal(result["pixel_values"][1], batch["pixel_values"][4:]))

    def test_split_correctly_1(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),  # Products: [4, 8]
            "pixel_values": torch.arange(12 * 3).reshape(12, 3),  # Shape: [12, 3]
        }
        result = split_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], list)
        self.assertEqual(len(result["pixel_values"]), 2)
        self.assertTrue(torch.equal(result["pixel_values"][0], batch["pixel_values"][:4]))
        self.assertTrue(torch.equal(result["pixel_values"][1], batch["pixel_values"][4:12]))

    def test_missing_keys(self):
        batch = {"pixel_values": torch.tensor([1.0])}
        result = split_pixel_values_by_grid(batch)
        self.assertEqual(result, batch)

    def test_mismatched_length(self):
        batch = {
            "image_grid_thw": torch.tensor([[2, 1, 1], [2, 1, 1]]),  # Total = 4
            "pixel_values": torch.randn(3, 5),  # Only 3 rows
        }
        with self.assertRaises(ValueError):
            split_pixel_values_by_grid(batch)


class UnsplitPixelValuesByGridTester(unittest.TestCase):
    def test_unsplit_correctly(self):
        split = [torch.randn(4, 5), torch.randn(2, 5)]
        merged = torch.cat(split, dim=0)
        batch = {"pixel_values": split, "other_key": torch.tensor([1])}
        result = unsplit_pixel_values_by_grid(batch)
        self.assertIsInstance(result["pixel_values"], torch.Tensor)
        self.assertTrue(torch.allclose(result["pixel_values"], merged))
        self.assertIn("other_key", result)

    def test_no_op_if_not_list(self):
        original = torch.randn(5, 3)
        batch = {"pixel_values": original}
        result = unsplit_pixel_values_by_grid(batch)
        self.assertTrue(torch.equal(result["pixel_values"], original))


class GRPOTrainerTester(unittest.TestCase):
    def test_init_minimal(self):
        # Test that GRPOTrainer can be instantiated with only model, reward_model and train_dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            train_dataset=dataset,
        )

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @parameterized.expand([("bnpo",), ("dr_grpo",)])
    def test_training_loss_types(self, loss_type):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=32,  # reduce the completion length to reduce memory usage
                loss_type=loss_type,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_eval(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                per_device_eval_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                eval_strategy="steps",
                eval_steps=2,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
            )

            trainer.train()

    def test_training_multiple_iterations(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                num_iterations=2,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_peft
    def test_training_peft(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(),
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the peft params have changed and the base model params have not changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if n in base_param_names:  # We expect the base model params to be the same
                    self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
                elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                    self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_peft
    def test_training_peft_with_gradient_checkpointing(self):
        """Test that training works with PEFT and gradient checkpointing enabled."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            torch_dtype=torch.float32,  # Use float32 for testing to avoid precision issues
            use_cache=False,  # Required for gradient checkpointing
        )

        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                gradient_checkpointing=True,  # Enable gradient checkpointing
                report_to="none",
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

            # Verify gradient checkpointing is enabled
            self.assertIsInstance(trainer.model, PeftModel)

            # Store initial parameters to check which ones change
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that only LoRA parameters have changed, base model parameters remain unchanged
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if "lora" in n.lower():  # LoRA parameters should change
                    self.assertFalse(torch.equal(param, new_param), f"LoRA parameter {n} has not changed.")
                else:  # Base model parameters should not change
                    self.assertTrue(torch.equal(param, new_param), f"Base parameter {n} has changed.")

    def test_training_different_reward_model(self):
        # Use a reward model different from the model: different chat template, tokenization, etc.
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")
        reward_model_id = "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id)
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        # By default, the trainer uses the eos token as the padding token. However, for Llama models, the eos token
        # appears in the chat template. Using it as a pad token disrupts the reward calculation, as the calculation
        # considers the score of the last token before the first pad token. To ensure correct reward calculations,
        # we use a separate pad token instead.
        reward_tokenizer.pad_token = "<|finetune_right_pad_id|>"

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_model,
                args=training_args,
                train_dataset=dataset,
                reward_processing_classes=reward_tokenizer,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_reward_func_standard(self):
        # Test if trainer can handle reward function with standard format
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_func,
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_reward_func_conversational(self):
        # Test if trainer can handle reward function with conversational format
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that gives higher scores to longer completion content."""
            completion_contents = [completion[0]["content"] for completion in completions]
            return [float(len(content)) for content in completion_contents]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_func,
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_reward_funcs(self):
        # Test that GRPOTrainer can be instantiated with multiple reward functions
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=[reward_func1, reward_func2],
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_reward_funcs_with_None_output(self):
        """Test that a valid math reward function is processed correctly while the code reward function returns None."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def applicable_reward_func(completions, **kwargs):
            """A reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def non_applicable_reward_func(completions, **kwargs):
            """A reward function that returns None for all inputs, as it is not applicable to this sample."""
            return [None] * len(completions)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                report_to="none",
            )

            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=[
                    applicable_reward_func,
                    non_applicable_reward_func,
                ],  # One applicable, one non applicable
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {
                n: param.clone() for n, param in trainer.model.named_parameters() if param.requires_grad
            }

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_reward_funcs_with_weights(self):
        """Test that GRPOTrainer can handle multiple reward functions with weights."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func1(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        def reward_func2(completions, **kwargs):
            """Reward function that rewards completions with more unique letters."""
            return [float(len(set(completion))) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                reward_weights=[0.7, 0.3],  # weight of reward_func1 and reward_func2 respectively
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=[reward_func1, reward_func2],
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            # Check that training logs contain both reward metrics
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
            self.assertIn("rewards/reward_func1/mean", trainer.state.log_history[-1])
            self.assertIn("rewards/reward_func1/std", trainer.state.log_history[-1])
            self.assertIn("rewards/reward_func2/mean", trainer.state.log_history[-1])
            self.assertIn("rewards/reward_func2/std", trainer.state.log_history[-1])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_mixed_reward_funcs(self):
        # Test if the trainer can handle a mix of reward functions and reward models
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            """Reward function that rewards longer completions."""
            return [float(len(completion)) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=[reward_func, "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"],
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_reward_func_additional_column(self):
        # Test if trainer can handle reward function that rely on additional columns in the dataset
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        # Add a column to the dataset (dummy example, the column could be anything)
        some_values = list(range(len(dataset)))
        dataset = dataset.add_column("some_values", some_values)

        def reward_func(completions, some_values, **kwargs):
            """Reward function that rewards completions with lengths closer to the values in some_values."""
            return [float(abs(len(completion) - value)) for completion, value in zip(completions, some_values)]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_func,
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_sync_ref_model(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                sync_ref_model=True,
                ref_model_sync_steps=2,  # reduce sync steps to ensure a sync happens
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_beta_non_zero(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_entropy_filter(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                top_entropy_quantile=0.2,
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @unittest.skip("We should add a mock for the vLLM server.")
    @require_peft
    @require_vllm
    def test_training_vllm_and_peft(self):
        """Test that training works with vLLM for generation."""
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # tiny model is too small for vLLM
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
            )
            lora_config = LoraConfig(
                target_modules="all-linear",
                # test with non-default modules as it adds extra keys in state_dict that we need to handle
                modules_to_save=["embed_tokens", "lm_head"],
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=lora_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the peft params have changed and the base model params have not changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if n in base_param_names:  # We expect the base model params to be the same
                    self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
                elif "base_layer" not in n and "original_module" not in n:
                    # We expect the peft params to be different (except for the base layer)
                    self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_guided_decoding(self):
        """Test that training works with vLLM for generation with guided decoding."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
                vllm_guided_decoding_regex=r"<reasoning>\n.*\n</reasoning>\n<answer>\n.*\n</answer>",
            )
            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_additional_generation_kwargs(self):
        """Test that training works with additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                top_p=0.9,
                top_k=10,
                min_p=0.01,
                repetition_penalty=1.1,
            )

            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vllm
    @unittest.skip("We should add a mock for the vLLM server.")
    def test_training_vllm_with_additional_generation_kwargs(self):
        """Test that training works with vLLM and additional generation kwargs."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
                use_vllm=True,
                top_p=0.9,
                top_k=10,
                min_p=0.01,
                repetition_penalty=1.1,
            )

            trainer = GRPOTrainer(
                model="Qwen/Qwen2.5-0.5B-Instruct",  # tiny model is too small for vLLM
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_no_scale_rewards(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                scale_rewards=False,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @patch("transformers.generation.utils.GenerationMixin.generate")
    def test_training_with_mask_truncated_completions(self, mock_generate):
        """Test that training works with mask_truncated_completions=True parameter."""

        # We mock the generate method because the model's random weights make it extremely unlikely to produce a
        # sequence containing the EOS token within the allowed max_completion_length. As a result, all tokens are
        # masked in the loss, the model doesn't update, and the final check (which verifies the update) fails.
        def fake_generate(input_ids, **kwargs):
            # pad_token_id = 151643; eos_token_id = 151645
            completions_ids = torch.tensor(
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],  # this one is truncated
                    [9, 10, 11, 151645, 151643, 151643, 151643, 151643],  # this one contains eos
                    [12, 13, 14, 15, 16, 17, 18, 151645],  # particular case, eos is generated just within the limit
                ],
                device=input_ids.device,
            )
            return torch.cat([input_ids, completions_ids], dim=1)

        mock_generate.side_effect = fake_generate

        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                mask_truncated_completions=True,  # Enable masking of truncated completions
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_mask_truncated_completions_all_masked(self):
        """
        Test that when all generated completions are truncated (i.e., none contain an EOS token), and
        mask_truncated_completions=True, the model receives no effective learning signal and therefore does not update
        its parameters.

        Here, we don't mock the generate method, be we rely on the fact that the model the probability of generating
        the EOS token is extremely low, so all generated completions are truncated.
        """
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                mask_truncated_completions=True,  # Enable masking of truncated completions
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertTrue(torch.equal(param, new_param), f"Parameter {n} has changed.")

    def test_training_num_generations_larger_than_batch_size(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                num_generations=6,  # the number of generations is larger than the batch size, but
                gradient_accumulation_steps=2,  # gradient accumulation should allow that
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_delta_clipping(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                delta=2.0,  # set delta to a non-None value
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_multiple_dataloader_workers(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                dataloader_num_workers=2,  # use multiple dataloader workers
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_generation_kwargs(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                generation_kwargs={"do_sample": True, "top_k": 50, "length_penalty": -0.1},  # Add some gen kwargs
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_with_reward_func_accessing_trainer_state(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def reward_func(completions, **kwargs):
            trainer_state = kwargs.get("trainer_state")
            assert trainer_state is not None
            # transformers.TrainerState instance should have a `global_step` property.
            assert hasattr(trainer_state, "global_step")
            return [float(len(set(completion))) for completion in completions]

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                num_generations=2,
                max_completion_length=8,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=reward_func,
                args=training_args,
                train_dataset=dataset,
            )
            trainer.train()

    def test_prepare_input_called_with_correct_data(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                gradient_accumulation_steps=3,  # can be anything in this test
                # steps_per_generation*per_device_train_batch_size=24 is divisible by num_generations=4
                steps_per_generation=4,
                num_generations=4,
                per_device_train_batch_size=6,  # reduce the batch size to reduce memory usage
                num_iterations=2,
                shuffle_dataset=False,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )
            # steps_per_generation=4, per_device_train_batch_size=6 and num_generations=4, so we expect a
            # generation batch of 24 samples (steps_per_generation * per_device_train_batch_size), containing 6
            # different prompts (steps_per_generation * per_device_train_batch_size // num_generations), each repeated
            # 4 times (num_generations).
            expected_first_generation_batch = (
                [{"prompt": "Beautiful is better than"}] * 4
                + [{"prompt": "Explicit is"}] * 4
                + [{"prompt": "Simple is better"}] * 4
                + [{"prompt": "Complex"}] * 4
                + [{"prompt": "Flat is better than"}] * 4
                + [{"prompt": "Sparse is better"}] * 4
            )
            expected_second_generation_batch = (
                [{"prompt": "Readability"}] * 4
                + [{"prompt": "Special cases aren't special"}] * 4
                + [{"prompt": "Although practicality beats"}] * 4
                + [{"prompt": "Errors should never"}] * 4
                + [{"prompt": "Unless explicitly"}] * 4
                + [{"prompt": "In the face of ambiguity, refuse"}] * 4
            )

            with patch.object(GRPOTrainer, "training_step", wraps=trainer.training_step) as mock_prepare:
                trainer.train()
                # 3 epochs * 2 iterations * 2 generation batches to cover the dataset * 4 steps_per_generation
                self.assertEqual(mock_prepare.call_count, 48)
                for i in range(0, 8):  # Generation batch repeated 8 times (steps_per_generation*num_iterations)
                    assert mock_prepare.call_args_list[i].args[1] == expected_first_generation_batch
                for i in range(8, 16):
                    assert mock_prepare.call_args_list[i].args[1] == expected_second_generation_batch

    @parameterized.expand(
        [
            ("trl-internal-testing/tiny-Gemma3ForConditionalGeneration",),
            ("trl-internal-testing/tiny-LlavaNextForConditionalGeneration",),
            ("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",),
            ("trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",),
            # ("trl-internal-testing/tiny-SmolVLMForConditionalGeneration",), seems not to support bf16 properly
        ]
    )
    @require_vision
    def test_training_vlm(self, model_id):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                max_prompt_length=None,  # disable prompt truncation, because usually, models don't support it
                report_to="none",
            )
            trainer = GRPOTrainer(
                model=model_id,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            # Because of the way the tiny models are initialized, the gradient does not flow properly through the
            # vision parts of the model, so we skip them. Ideally, we should fix the init of these models.
            params_to_skip = (
                "model.vision_tower.",
                "model.multi_modal_projector.",
                "model.vision_model.",
                "model.connector.modality_projection.",
            )
            for n, param in previous_trainable_params.items():
                if n.startswith(params_to_skip):
                    continue
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_beta_non_zero(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                beta=0.1,  # set beta to non-zero value to test the case where the reference model is used
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    @require_peft
    def test_training_vlm_peft(self):
        model = AutoModelForImageTextToText.from_pretrained(
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
        )
        base_param_names = [f"base_model.model.{n}" for n, _ in model.named_parameters()]
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                report_to="none",
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
                peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]),
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the peft params have changed and the base model params have not changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                if n in base_param_names:  # We expect the base model params to be the same
                    self.assertTrue(torch.allclose(param, new_param), f"Parameter {n} has changed.")
                elif "base_layer" not in n:  # We expect the peft params to be different (except for the base layer)
                    self.assertFalse(torch.allclose(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_and_importance_sampling(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                steps_per_generation=2,  # increase the steps per generation to trigger IS
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    @require_liger_kernel
    def test_training_vlm_and_liger(self):
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                use_liger_loss=True,  # Enable Liger loss
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    @require_vision
    def test_training_vlm_and_prompt_truncation(self):
        # If not handled properly, prompt truncation may truncate image token
        dataset = load_dataset("trl-internal-testing/zen-image", "conversational_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                max_prompt_length=18,
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")

    def test_training_sequence_importance_sampling(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = GRPOConfig(
                output_dir=tmp_dir,
                learning_rate=0.1,  # increase the learning rate to speed up the test
                per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
                num_generations=3,  # reduce the number of generations to reduce memory usage
                max_completion_length=8,  # reduce the completion length to reduce memory usage
                num_iterations=2,  # the importance sampling weights won't be 0 in this case
                importance_sampling_level="sequence",
                report_to="none",
            )
            trainer = GRPOTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

            # Check that the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                self.assertFalse(torch.equal(param, new_param), f"Parameter {n} has not changed.")


if __name__ == "__main__":
    unittest.main()
