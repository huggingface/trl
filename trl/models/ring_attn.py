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

import logging

import torch
import torch.distributed as dist

from ..import_utils import is_ring_attn_available


if is_ring_attn_available():
    from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params

logger = logging.getLogger(__name__)

RING_ATTN_GROUP = None


def get_ring_attn_group() -> dist.ProcessGroup | None:
    """
    Getter for ring attention group on this rank.

    Returns:
        The process group for ring attention for this rank, or None if not initialized.
    """
    return RING_ATTN_GROUP


def set_ring_attn_group(ring_attn_group: dist.ProcessGroup | None):
    """
    Setter for ring attention group on this rank.

    Args:
        ring_attn_group: Process group for ring attention.
    """
    global RING_ATTN_GROUP  # pylint: disable=global-statement
    RING_ATTN_GROUP = ring_attn_group


def register_ring_attn(sequence_parallel_degree: int, heads_k_stride: int = 1):
    """
    Create ring attention group and substitute flash attn with ring flash attn.

    Args:
        sequence_parallel_size (`int`):
            Sequence parallelism factor.
        heads_k_stride (`int`, *optional*, defaults to `1`):
            Sequence parallelism K head stride size. Passed through to `ring_flash_attn.substitute_hf_flash_attn`.
    """
    if not is_ring_attn_available():
        raise ImportError(
            "ring-flash-attn is required for sequence parallelism with ring attention. Please install it using: "
            "`pip install ring-flash-attn`."
        )

    if get_ring_attn_group() is not None:
        logger.info("Ring attention already registered, exiting early...")
        return

    if not dist.is_initialized():
        logger.error("Distributed process group is not initialized. Cannot register ring attention.")
        return

    logger.info(
        "Enabling ring attention sequence parallelism: each sequence will be processed across "
        f"{sequence_parallel_degree} GPUs"
    )

    world_size = dist.get_world_size()
    if sequence_parallel_degree > world_size:
        raise ValueError(
            f"sequence_parallel_degree ({sequence_parallel_degree}) must be less than or equal to world_size "
            f"({world_size})"
        )
    if world_size % sequence_parallel_degree != 0:
        raise ValueError(
            f"sequence_parallel_degree ({sequence_parallel_degree}) must evenly divide world_size ({world_size})"
        )

    rank = dist.get_rank()
    num_groups = world_size // sequence_parallel_degree
    group_assignments = {}
    local_group = None

    # Create sequence parallel groups
    for group_idx in range(num_groups):
        ring_attn_ranks = list(range(group_idx * sequence_parallel_degree, (group_idx + 1) * sequence_parallel_degree))
        # NCCL backend is assumed for GPU communication
        group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")

        # Track which GPUs are in which groups for logging
        for rank in ring_attn_ranks:
            group_assignments[rank] = group_idx

        # Assign the group to the current rank if it belongs to this group
        if rank in ring_attn_ranks:
            local_group = group

    if local_group is None:
        # This should theoretically not happen if ranks cover 0 to world_size-1 and checks above pass.
        raise RuntimeError(f"Rank {rank} was not assigned to any ring attention group.")

    set_ring_attn_group(local_group)

    # Log the GPU group assignments from rank 0 for clarity
    if rank == 0:
        logger.info(f"Sequence parallel group assignments (GPU Rank -> Group Index): {group_assignments}")

    substitute_hf_flash_attn(process_group=get_ring_attn_group(), heads_k_stride=heads_k_stride)


def update_ring_attn_params(batch: dict[str, torch.Tensor]):
    """
    Compute global cumulative sequence lengths (cu_seqlens) for ring attention.

    This function gathers position_ids from all ranks, reconstructs the global sequence
    structure, and computes cu_seqlens that satisfy ring-flash-attention requirements.

    Args:
        batch: Batch dictionary containing 'input_ids' and optionally 'position_ids'.
               If 'position_ids' is missing, it will be created automatically.
    """
    if not is_ring_attn_available():
        raise ImportError(
            "ring-flash-attn is required for sequence parallelism with ring attention. Please install it using: "
            "`pip install ring-flash-attn`."
        )

    input_ids = batch["input_ids"]
    position_ids = batch.get("position_ids")

    if position_ids is None:
        # This should rarely happen if data collator is good
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        batch["position_ids"] = position_ids  # Add back to batch for model use

    batch_size, seq_len = input_ids.shape
    ring_group = get_ring_attn_group()

    if ring_group is None:
        raise RuntimeError("Ring attention group is not initialized. Call register_ring_attn() first.")

    # For ring attention, we need to compute cu_seqlens based on global sequence structure
    # Each rank has a slice of the global sequence, so we need to gather information across ranks

    world_size = dist.get_world_size(group=ring_group)

    # Gather sequence lengths from all ranks to compute global sequence length
    local_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=input_ids.device)
    all_seq_lens = [torch.zeros_like(local_seq_len) for _ in range(world_size)]
    dist.all_gather(all_seq_lens, local_seq_len, group=ring_group)

    # Compute global sequence length and verify it's the same across all ranks
    global_seq_len = sum(tensor.item() for tensor in all_seq_lens)

    # Verify that sequence length is the same across all ranks (as expected for ring attention)
    first_rank_seq_len = all_seq_lens[0].item()
    if not all(tensor.item() == first_rank_seq_len for tensor in all_seq_lens):
        logger.warning(
            f"Sequence lengths are not uniform across ranks: {[t.item() for t in all_seq_lens]}. "
            "This may cause issues with ring attention."
        )

    # For ring attention, we typically have one global sequence split across ranks
    # However, we need to detect sequence boundaries within the global sequence

    # Gather position_ids from all ranks to reconstruct global sequence structure
    pos_ids = position_ids[0]  # Take first batch (should be batch_size=1 for padding-free)
    all_pos_ids = [torch.zeros(seq_len, dtype=pos_ids.dtype, device=pos_ids.device) for _ in range(world_size)]
    dist.all_gather(all_pos_ids, pos_ids, group=ring_group)

    # Concatenate to get global position_ids
    global_pos_ids = torch.cat(all_pos_ids, dim=0)

    # Find where position resets to 0 (indicates start of new sequence) in global sequence
    resets = torch.cat(
        [
            torch.tensor([True], device=global_pos_ids.device),  # First position is always a start
            global_pos_ids[1:] == 0,  # Position resets to 0 indicate new sequence starts
        ]
    )

    # Get sequence start positions in global sequence
    start_positions = torch.nonzero(resets, as_tuple=False).squeeze(-1)
    if start_positions.dim() == 0:
        start_positions = start_positions.unsqueeze(0)

    # Add the final position (total global sequence length)
    end_position = torch.tensor([global_seq_len], device=global_pos_ids.device)
    cu_seqlens = torch.cat([start_positions, end_position]).to(dtype=torch.int32)

    cu_seqlens = cu_seqlens.to(device=torch.cuda.current_device())
    update_ring_flash_attn_params(cu_seqlens, ring_group)


def reset_ring_attn_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """
    Reset position IDs for ring attention to be local to each rank while preserving sequence boundaries.

    For ring attention, each rank should have position IDs that are local to its slice
    while maintaining the sequence structure (resets to 0 at sequence boundaries).

    Args:
        position_ids: Original position IDs tensor with shape [batch_size, seq_len]

    Returns:
        Reset position IDs tensor with shape [batch_size, seq_len] where
        position IDs are local to this rank but preserve sequence boundaries
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)

    batch_size, seq_len = position_ids.shape
    reset_position_ids = torch.zeros_like(position_ids)

    for batch_idx in range(batch_size):
        seq_pos_ids = position_ids[batch_idx]

        # Find where sequences start (position_ids reset to 0 or are at the beginning)
        seq_starts = torch.cat(
            [
                torch.tensor([True], device=seq_pos_ids.device),  # First position is always a start
                seq_pos_ids[1:] == 0,  # Position resets to 0 indicate new sequence starts
            ]
        )

        # Generate local position IDs that reset at sequence boundaries
        local_pos_ids = torch.zeros_like(seq_pos_ids)
        current_pos = 0

        for i in range(seq_len):
            if seq_starts[i]:
                current_pos = 0  # Reset at sequence start
            local_pos_ids[i] = current_pos
            current_pos += 1

        reset_position_ids[batch_idx] = local_pos_ids

    return reset_position_ids
