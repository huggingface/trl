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
from unittest.mock import MagicMock, patch

import torch
from transformers.testing_utils import require_torch, torch_device

from trl.models.ring_attn import (
    get_cu_seqlens_from_pos_ids,
    get_ring_attn_group,
    reset_ring_attn_position_ids,
    set_ring_attn_group,
    update_ring_attn_params,
)

from .testing_utils import require_ring_attn


class TestRingAttnHelpers(unittest.TestCase):
    """Test suite for ring attention helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch_device

    def test_get_cu_seqlens_from_pos_ids_single_sequence(self):
        """Test cu_seqlens creation for a single sequence."""
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=self.device)

        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        # For a single sequence, we expect [0, seq_len]
        expected = torch.tensor([[0, 8]], dtype=torch.int32, device=self.device)
        self.assertEqual(cu_seqlens.shape, expected.shape)
        # Since get_cu_seqlens_from_pos_ids returns a padded 2D tensor, we check the values
        self.assertEqual(cu_seqlens[0, 0].item(), 0)
        self.assertEqual(cu_seqlens[0, 1].item(), 8)

    def test_get_cu_seqlens_from_pos_ids_multiple_sequences(self):
        """Test cu_seqlens creation for multiple packed sequences."""
        # Three sequences: [0,1,2], [0,1,2,3], [0,1]
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 0, 1]], dtype=torch.long, device=self.device)

        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        # Should detect sequence boundaries at positions 0, 3, 7, and end at 9
        # So cu_seqlens should be [0, 3, 7, 9]
        self.assertEqual(cu_seqlens[0, 0].item(), 0)
        self.assertEqual(cu_seqlens[0, 1].item(), 3)
        self.assertEqual(cu_seqlens[0, 2].item(), 7)
        self.assertEqual(cu_seqlens[0, 3].item(), 9)

    def test_get_cu_seqlens_from_pos_ids_with_padding(self):
        """Test cu_seqlens creation with right-side padding."""
        # Sequence with padding: [0,1,2,3,0,0,0] where last 3 zeros are padding
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 0, 0]], dtype=torch.long, device=self.device)

        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        # Should recognize that the first 4 positions are a sequence and the rest is padding
        # Result should be [0, 4, 7] where 7 is the total length including padding
        self.assertEqual(cu_seqlens[0, 0].item(), 0)
        self.assertEqual(cu_seqlens[0, 1].item(), 4)
        self.assertEqual(cu_seqlens[0, 2].item(), 7)

    def test_get_cu_seqlens_from_pos_ids_1d_input(self):
        """Test cu_seqlens creation with 1D input (should be expanded to 2D)."""
        position_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=self.device)

        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        # Should handle 1D input by expanding to 2D
        self.assertEqual(cu_seqlens.ndim, 2)
        self.assertEqual(cu_seqlens[0, 0].item(), 0)
        self.assertEqual(cu_seqlens[0, 1].item(), 5)

    def test_reset_ring_attn_position_ids_2d(self):
        """Test position ID reset for 2D input."""
        position_ids = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long, device=self.device)

        reset_pos_ids = reset_ring_attn_position_ids(position_ids)

        expected = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long, device=self.device)
        torch.testing.assert_close(reset_pos_ids, expected)

    def test_reset_ring_attn_position_ids_1d(self):
        """Test position ID reset for 1D input (should be expanded to 2D)."""
        position_ids = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=self.device)

        reset_pos_ids = reset_ring_attn_position_ids(position_ids)

        expected = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=self.device)
        torch.testing.assert_close(reset_pos_ids, expected)

    def test_reset_ring_attn_position_ids_multiple_batches(self):
        """Test position ID reset for multiple batches."""
        position_ids = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long, device=self.device)

        reset_pos_ids = reset_ring_attn_position_ids(position_ids)

        expected = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long, device=self.device)
        torch.testing.assert_close(reset_pos_ids, expected)

    def test_get_set_ring_attn_group(self):
        """Test getting and setting ring attention group."""
        # Initially should be None
        self.assertIsNone(get_ring_attn_group())

        # Create a mock process group
        mock_group = MagicMock()

        # Set the group
        set_ring_attn_group(mock_group)

        # Should return the set group
        self.assertEqual(get_ring_attn_group(), mock_group)

        # Reset to None
        set_ring_attn_group(None)
        self.assertIsNone(get_ring_attn_group())

    @require_ring_attn
    @patch("trl.models.ring_attn.update_ring_flash_attn_params")
    @patch("trl.models.ring_attn.get_ring_attn_group")
    def test_update_ring_attn_params_single_sequence(self, mock_get_group, mock_update_params):
        """Test update_ring_attn_params with a single sequence."""
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=self.device),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=self.device),
        }

        update_ring_attn_params(batch)

        # Verify that update_ring_flash_attn_params was called
        mock_update_params.assert_called_once()

        # Get the cu_seqlens argument that was passed
        args, kwargs = mock_update_params.call_args
        cu_seqlens = args[0]

        # Should be [0, 8] for a single sequence of length 8
        expected_cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
        torch.testing.assert_close(cu_seqlens.cpu(), expected_cu_seqlens)

        # Group should be passed as second argument
        self.assertEqual(args[1], mock_group)

    @require_ring_attn
    @patch("trl.models.ring_attn.update_ring_flash_attn_params")
    @patch("trl.models.ring_attn.get_ring_attn_group")
    def test_update_ring_attn_params_multiple_sequences(self, mock_get_group, mock_update_params):
        """Test update_ring_attn_params with multiple packed sequences."""
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        # Three packed sequences: lengths 120, 1128, 2984 (total 4232)
        position_ids = (
            torch.cat([torch.arange(120), torch.arange(1128), torch.arange(2984)]).unsqueeze(0).to(device=self.device)
        )

        batch = {
            "input_ids": torch.randint(1, 100, (1, 4232), dtype=torch.long, device=self.device),
            "position_ids": position_ids,
        }

        update_ring_attn_params(batch)

        # Verify that update_ring_flash_attn_params was called
        mock_update_params.assert_called_once()

        # Get the cu_seqlens argument that was passed
        args, kwargs = mock_update_params.call_args
        cu_seqlens = args[0]

        # Should be [0, 120, 1248, 4232] for three sequences
        expected_cu_seqlens = torch.tensor([0, 120, 1248, 4232], dtype=torch.int32)
        torch.testing.assert_close(cu_seqlens.cpu(), expected_cu_seqlens)

    @require_ring_attn
    @patch("trl.models.ring_attn.update_ring_flash_attn_params")
    @patch("trl.models.ring_attn.get_ring_attn_group")
    def test_update_ring_attn_params_no_position_ids(self, mock_get_group, mock_update_params):
        """Test update_ring_attn_params when position_ids are not provided."""
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.device)
            # No position_ids provided
        }

        update_ring_attn_params(batch)

        # Should create position_ids automatically
        self.assertIn("position_ids", batch)
        expected_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        torch.testing.assert_close(batch["position_ids"].cpu(), expected_position_ids)

        # Verify that update_ring_flash_attn_params was called
        mock_update_params.assert_called_once()

    def test_update_ring_attn_params_no_ring_attn_available(self):
        """Test update_ring_attn_params when ring-flash-attn is not available."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.device),
            "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=self.device),
        }

        with patch("trl.models.ring_attn.is_ring_attn_available", return_value=False):
            with self.assertRaises(ImportError) as context:
                update_ring_attn_params(batch)

            self.assertIn("ring-flash-attn is required", str(context.exception))

    def test_cu_seqlens_edge_cases(self):
        """Test edge cases for cu_seqlens creation."""
        # Empty sequence (should not happen in practice, but function handles it gracefully)
        position_ids = torch.tensor([[]], dtype=torch.long, device=self.device)
        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        # Empty sequence should return [0, 0] indicating no tokens
        expected = torch.tensor([[0, 0]], dtype=torch.int32, device=self.device)
        torch.testing.assert_close(cu_seqlens, expected)

        # Single token sequence
        position_ids = torch.tensor([[0]], dtype=torch.long, device=self.device)
        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        # Single token should return [0, 0, 1] due to padding behavior
        expected_single = torch.tensor([[0, 0, 1]], dtype=torch.int32, device=self.device)
        torch.testing.assert_close(cu_seqlens, expected_single)

    def test_cu_seqlens_complex_packed_scenario(self):
        """Test cu_seqlens creation for a complex packed scenario."""
        # Simulate a realistic packing scenario:
        # Seq 1: length 5, Seq 2: length 3, Seq 3: length 7, Seq 4: length 2
        position_ids = torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,  # Seq 1: length 5
                    0,
                    1,
                    2,  # Seq 2: length 3
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,  # Seq 3: length 7
                    0,
                    1,  # Seq 4: length 2
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )

        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)

        expected_boundaries = [0, 5, 8, 15, 17]
        for i, expected_boundary in enumerate(expected_boundaries):
            self.assertEqual(cu_seqlens[0, i].item(), expected_boundary)

    @require_torch
    def test_device_consistency(self):
        """Test that functions handle device placement correctly."""
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")

            # Test get_cu_seqlens_from_pos_ids device consistency
            position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=cuda_device)
            cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)
            self.assertEqual(cu_seqlens.device, cuda_device)

            # Test reset_ring_attn_position_ids device consistency
            reset_pos_ids = reset_ring_attn_position_ids(position_ids)
            self.assertEqual(reset_pos_ids.device, cuda_device)

    def test_dtype_consistency(self):
        """Test that functions handle dtypes correctly."""
        # Test that cu_seqlens always returns int32
        position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=self.device)
        cu_seqlens = get_cu_seqlens_from_pos_ids(position_ids)
        self.assertEqual(cu_seqlens.dtype, torch.int32)

        # Test that reset_ring_attn_position_ids preserves input dtype
        for dtype in [torch.long, torch.int32, torch.int64]:
            position_ids = torch.tensor([[0, 1, 2, 3]], dtype=dtype, device=self.device)
            reset_pos_ids = reset_ring_attn_position_ids(position_ids)
            self.assertEqual(reset_pos_ids.dtype, dtype)
