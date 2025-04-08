import logging

import torch.distributed as dist

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
        sequence_parallel_degree: Sequence parallelism factor.
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
            f"sequence_parallel_degree ({sequence_parallel_degree}) "
            f"must evenly divide world_size ({world_size})"
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

        substitute_hf_flash_attn(
            process_group=get_ring_attn_group(), heads_k_stride=heads_k_stride
        )
        logger.info("Successfully substituted HF flash attention with ring flash attention.")
    else:
        logger.error(
            "Could not import `substitute_hf_flash_attn` from `ring_flash_attn`. "
            "Please ensure the 'ring-flash-attn' package is installed."
        )
        # Reset the group if substitution fails to avoid inconsistent state
        set_ring_attn_group(None)
        raise ImportError("Could not import `substitute_hf_flash_attn` from `ring_flash_attn`.")
