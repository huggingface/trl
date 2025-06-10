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
    get_ring_attn_group,
    reset_ring_attn_position_ids,
    set_ring_attn_group,
    update_ring_attn_params,
)

from .testing_utils import require_ring_flash_attn


class TestRingAttnHelpers(unittest.TestCase):
    """Test suite for ring attention helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch_device

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

    @require_ring_flash_attn
    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("trl.models.ring_attn.update_ring_flash_attn_params")
    @patch("trl.models.ring_attn.get_ring_attn_group")
    def test_update_ring_attn_params_single_sequence(
        self, mock_get_group, mock_update_params, mock_get_world_size, mock_get_rank, mock_all_gather
    ):
        """Test update_ring_attn_params with a single sequence."""
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=self.device),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=self.device),
        }

        def mock_all_gather_side_effect(tensor_list, tensor, group=None):
            if tensor.dtype == torch.int32 and tensor.numel() == 1:
                # Sequence length gathering (each rank has 8 tokens)
                tensor_list[0].copy_(torch.tensor([8], dtype=torch.int32, device=self.device))
                tensor_list[1].copy_(torch.tensor([8], dtype=torch.int32, device=self.device))
            else:
                # Position IDs gathering (rank 0: 0-7, rank 1: 8-15)
                tensor_list[0].copy_(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=tensor.dtype, device=self.device))
                tensor_list[1].copy_(
                    torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=tensor.dtype, device=self.device)
                )

        mock_all_gather.side_effect = mock_all_gather_side_effect

        update_ring_attn_params(batch)

        # Verify that update_ring_flash_attn_params was called
        mock_update_params.assert_called_once()

        # Get the cu_seqlens argument that was passed
        args, kwargs = mock_update_params.call_args
        cu_seqlens = args[0]

        # Should be [0, 16] for one global sequence of 16 tokens (8 + 8)
        expected_cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device=self.device)
        torch.testing.assert_close(cu_seqlens, expected_cu_seqlens)

        # Group should be passed as second argument
        self.assertEqual(args[1], mock_group)

    @require_ring_flash_attn
    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("trl.models.ring_attn.update_ring_flash_attn_params")
    @patch("trl.models.ring_attn.get_ring_attn_group")
    def test_update_ring_attn_params_multiple_sequences(
        self, mock_get_group, mock_update_params, mock_get_world_size, mock_get_rank, mock_all_gather
    ):
        """Test update_ring_attn_params with multiple packed sequences."""
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        # Two packed sequences that will be split across ranks
        # First rank: seq1 (0-2) + part of seq2 (0-1)
        # Second rank: rest of seq2 (2-4) + seq3 (0-2)
        rank0_position_ids = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long, device=self.device)
        rank1_position_ids = torch.tensor([2, 3, 4, 0, 1], dtype=torch.long, device=self.device)

        batch = {
            "input_ids": torch.randint(1, 100, (1, 5), dtype=torch.long, device=self.device),
            "position_ids": rank0_position_ids.unsqueeze(0),
        }

        def mock_all_gather_side_effect(tensor_list, tensor, group=None):
            if tensor.dtype == torch.int32 and tensor.numel() == 1:
                # Sequence length gathering (each rank has 5 tokens)
                tensor_list[0].copy_(torch.tensor([5], dtype=torch.int32, device=self.device))
                tensor_list[1].copy_(torch.tensor([5], dtype=torch.int32, device=self.device))
            else:
                # Position IDs gathering
                tensor_list[0].copy_(rank0_position_ids)
                tensor_list[1].copy_(rank1_position_ids)

        mock_all_gather.side_effect = mock_all_gather_side_effect

        update_ring_attn_params(batch)

        # Verify that update_ring_flash_attn_params was called
        mock_update_params.assert_called_once()

        # Get the cu_seqlens argument that was passed
        args, kwargs = mock_update_params.call_args
        cu_seqlens = args[0]

        # Global sequence: [0,1,2,0,1,2,3,4,0,1]
        # Three sequences: [0,1,2], [0,1,2,3,4], [0,1]
        # Lengths: 3, 5, 2 -> cu_seqlens: [0, 3, 8, 10]
        expected_cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.int32, device=self.device)
        torch.testing.assert_close(cu_seqlens, expected_cu_seqlens)

    @require_ring_flash_attn
    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.get_rank")
    @patch("torch.distributed.get_world_size")
    @patch("trl.models.ring_attn.update_ring_flash_attn_params")
    @patch("trl.models.ring_attn.get_ring_attn_group")
    def test_update_ring_attn_params_no_position_ids(
        self, mock_get_group, mock_update_params, mock_get_world_size, mock_get_rank, mock_all_gather
    ):
        """Test update_ring_attn_params when position_ids are not provided."""
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group
        mock_get_world_size.return_value = 2
        mock_get_rank.return_value = 0

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.device)
            # No position_ids provided
        }

        def mock_all_gather_side_effect(tensor_list, tensor, group=None):
            if tensor.dtype == torch.int32 and tensor.numel() == 1:
                # Sequence length gathering (each rank has 4 tokens)
                tensor_list[0].copy_(torch.tensor([4], dtype=torch.int32, device=self.device))
                tensor_list[1].copy_(torch.tensor([4], dtype=torch.int32, device=self.device))
            else:
                # Position IDs gathering (rank 0: 0-3, rank 1: 4-7)
                tensor_list[0].copy_(torch.tensor([0, 1, 2, 3], dtype=tensor.dtype, device=self.device))
                tensor_list[1].copy_(torch.tensor([4, 5, 6, 7], dtype=tensor.dtype, device=self.device))

        mock_all_gather.side_effect = mock_all_gather_side_effect

        update_ring_attn_params(batch)

        # Should create position_ids automatically
        self.assertIn("position_ids", batch)
        expected_position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=self.device)
        torch.testing.assert_close(batch["position_ids"], expected_position_ids)

        # Verify that update_ring_flash_attn_params was called
        mock_update_params.assert_called_once()

    def test_update_ring_attn_params_no_ring_attn_available(self):
        """Test update_ring_attn_params when ring-flash-attn is not available."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=self.device),
            "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=self.device),
        }

        with patch("trl.models.ring_attn.is_ring_flash_attn_available", return_value=False):
            with self.assertRaises(ImportError) as context:
                update_ring_attn_params(batch)

            self.assertIn("ring-flash-attn is required", str(context.exception))

    @require_torch
    def test_device_consistency(self):
        """Test that functions handle device placement correctly."""
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")

            # Test reset_ring_attn_position_ids device consistency
            position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=cuda_device)
            reset_pos_ids = reset_ring_attn_position_ids(position_ids)
            self.assertEqual(reset_pos_ids.device, cuda_device)

    def test_dtype_consistency(self):
        """Test that functions handle dtypes correctly."""
        # Test that reset_ring_attn_position_ids preserves input dtype
        for dtype in [torch.long, torch.int32, torch.int64]:
            position_ids = torch.tensor([[0, 1, 2, 3]], dtype=dtype, device=self.device)
            reset_pos_ids = reset_ring_attn_position_ids(position_ids)
            self.assertEqual(reset_pos_ids.dtype, dtype)
