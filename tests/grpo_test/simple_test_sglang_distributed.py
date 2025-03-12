import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sglang as sgl
from typing import Optional


def setup_process(rank: int, world_size: int) -> None:
    """Setup the distributed environment for a process.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Set environment variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Additional NCCL environment variables for better stability
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # Initialize the process group
    torch.cuda.set_device(rank)  # Set GPU device before initialization

    try:
        # Initialize the process group without explicit timeout
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:12355",
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e


def cleanup_process() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def initialize_sglang(rank: int, model_path: str) -> Optional[sgl.Engine]:
    """Initialize the SGLang engine.

    Args:
        rank: Process rank
        model_path: Path to the model

    Returns:
        Optional[sgl.Engine]: Initialized engine or None for non-rank-0 processes
    """
    if rank != 0:
        return None

    try:
        # Initialize the engine
        engine = sgl.Engine(
            model_path=model_path,
            base_gpu_id=rank,
            random_seed=42,
            mem_fraction_static=0.9,
        )
        return engine
    except Exception as e:
        return None


def run_process(rank: int, world_size: int) -> None:
    """Main function to run in each process.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    try:
        # Set the device for this process
        torch.cuda.set_device(rank)

        # Initialize distributed environment
        setup_process(rank, world_size)

        # Verify initialization
        if not dist.is_initialized():
            return

        # Synchronize processes
        dist.barrier()

        # Initialize SGLang (rank 0 only)
        engine = initialize_sglang(rank, "Qwen/Qwen2.5-0.5B-Instruct")

        # Wait for SGLang initialization to complete
        dist.barrier()

        # Run a test query on rank 0
        if rank == 0 and engine is not None:
            try:
                result = engine.generate(
                    sgl.Prompt("What is the capital of France?"),
                    sampling_params={"max_tokens": 50},
                )
            except Exception:
                pass

        # Final synchronization
        dist.barrier()

    except Exception:
        pass
    finally:
        # Clean up distributed environment
        cleanup_process()


def main() -> None:
    """Script entry point."""
    # Get world size
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")

    if world_size < 1:
        print("No GPUs available. Exiting.")
        return

    # Use at most 3 GPUs (adjust as needed)
    world_size = min(world_size, 3)
    print(f"Using {world_size} GPUs for distributed processing")

    # Launch processes
    mp.spawn(run_process, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    # For better error reporting
    mp.set_start_method("spawn")
    main()
