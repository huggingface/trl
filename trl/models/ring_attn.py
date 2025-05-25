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
import torch.nn.functional as F

from ..import_utils import is_ring_attn_available


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


def register_ring_attn(sequence_parallel_degree: int, heads_k_stride: int | None = None):
    """
    Create ring attention group and substitute flash attn with ring flash attn.

    Args:
        sequence_parallel_size: Sequence parallelism factor.
        heads_k_stride: Sequence parallelism K head stride size. Passed
            through to `ring_flash_attn.substitute_hf_flash_attn`. Defaults to 1.
    """
    if get_ring_attn_group() is not None:
        logger.info("Ring attention already registered, exiting early...")
        return

    if not dist.is_initialized():
        logger.error("Distributed process group is not initialized. Cannot register ring attention.")
        return

    logger.info(
        "Enabling ring attention sequence parallelism: "
        f"each sequence will be processed across {sequence_parallel_degree} GPUs"
    )

    world_size = dist.get_world_size()
    if sequence_parallel_degree > world_size:
        raise ValueError(
            f"sequence_parallel_degree ({sequence_parallel_degree}) "
            f"must be less than or equal to world_size ({world_size})"
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
    for i in range(num_groups):
        ring_attn_ranks = list(
            range(
                i * sequence_parallel_degree,
                (i + 1) * sequence_parallel_degree,
            )
        )
        # NCCL backend is assumed for GPU communication
        group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")

        # Track which GPUs are in which groups for logging
        for r in ring_attn_ranks:
            group_assignments[r] = i

        # Assign the group to the current rank if it belongs to this group
        if rank in ring_attn_ranks:
            local_group = group

    if local_group is None:
        # This should theoretically not happen if ranks cover 0 to world_size-1
        # and checks above pass.
        raise RuntimeError(f"Rank {rank} was not assigned to any ring attention group.")

    set_ring_attn_group(local_group)

    # Log the GPU group assignments from rank 0 for clarity
    if rank == 0:
        logger.info(f"Sequence parallel group assignments (GPU Rank -> Group Index): {group_assignments}")

    if heads_k_stride is None:
        heads_k_stride = 1

    if is_ring_attn_available():
        from ring_flash_attn import substitute_hf_flash_attn

        substitute_hf_flash_attn(process_group=get_ring_attn_group(), heads_k_stride=heads_k_stride)
        logger.info("Successfully substituted HF flash attention with ring flash attention.")
    else:
        logger.error(
            "Could not import `substitute_hf_flash_attn` from `ring_flash_attn`. "
            "Please ensure the 'ring-flash-attn' package is installed."
        )
        # Reset the group if substitution fails to avoid inconsistent state
        set_ring_attn_group(None)
        raise ImportError("Could not import `substitute_hf_flash_attn` from `ring_flash_attn`.")


def get_cu_seqlens_from_pos_ids(
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """generate a cumulative sequence length mask for flash attention using pos ids"""
    if len(position_ids.shape) == 1:
        position_ids = position_ids.unsqueeze(0)

    device = position_ids.device
    results = []
    max_seq_lens = []

    for row in position_ids:
        # Count the number of consecutive zeros from the right side
        padding_length = (row == 0).int().flip(dims=[0]).cumprod(dim=0).sum().item()

        # Adjust the row to exclude padding
        adjusted_row = row[:-padding_length] if padding_length else row.clone()

        # Find where the position resets to 0 (indicating a new sequence)
        seq_starts = torch.cat(
            [
                torch.tensor([True], dtype=torch.bool, device=device),
                adjusted_row[1:] == 0,
            ]
        )
        # Get the indices where the sequence starts
        start_indices = torch.cat(
            [
                torch.nonzero(seq_starts).unbind(dim=1)[0],
                torch.tensor([len(adjusted_row)], dtype=torch.int32, device=device),
            ]
        )
        # Calculate the sequence lengths
        seq_lengths = start_indices[1:] - start_indices[:-1]
        # Calculate the cumulative sequence lengths
        cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seq_lengths.cumsum(0)])
        # Append the padding length to the cumulative sequence lengths
        if padding_length:
            cu_seqlens = torch.cat([cu_seqlens, torch.tensor([len(row)], dtype=torch.int32, device=device)])
        max_seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        results.append(cu_seqlens)
        max_seq_lens.append(max_seq_len)

    # Find the maximum value across all tensors
    max_value = max(t.max() for t in results)

    # Find the length of the longest tensor
    max_length = max(t.size(0) for t in results)

    # Pad each tensor to the same length and collect them in a list
    padded_results = [F.pad(t, (0, max_length - t.size(0)), "constant", max_value) for t in results]

    return torch.stack(padded_results).to(dtype=torch.int32), torch.stack(max_seq_lens)


def update_ring_attn_params(batch: dict[str, torch.Tensor]):
    """
    Calculate the cumulative sequence lengths for the current forward pass and pass the
    value to the substituted `ring_flash_attn`.

    Args:
        batch: A dictionary with a batch of data. May or may not contain `position_ids`
            data; if not, we compute it.
    """
    from ring_flash_attn import update_ring_flash_attn_params

    input_ids = batch["input_ids"]
    position_ids = batch.get("position_ids")

    if position_ids is None:
        # This should rarely happen if data collator is good
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        batch["position_ids"] = position_ids  # Add back to batch for model use

    cu_seqlens, _ = get_cu_seqlens_from_pos_ids(position_ids)
    cu_seqlens = cu_seqlens.squeeze().to(device=torch.cuda.current_device())
    update_ring_flash_attn_params(cu_seqlens, get_ring_attn_group())
