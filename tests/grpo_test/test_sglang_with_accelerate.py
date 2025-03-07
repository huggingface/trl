import os
import torch
import sglang as sgl
from typing import Optional
from accelerate import Accelerator


def initialize_sglang_engine(
    model_path: str,
    base_gpu_id: int = 0,
    random_seed: int = 42,
    mem_fraction: float = 0.9,
) -> Optional[sgl.Engine]:
    """
    Initialize the SGLang engine.

    Args:
        model_path: Path to the model to be loaded
        base_gpu_id: Base GPU ID to use for the engine
        random_seed: Random seed for reproducibility
        mem_fraction: Memory fraction to use for static allocation

    Returns:
        Optional[sgl.Engine]: Initialized SGLang engine or None if initialization fails
    """
    try:
        engine = sgl.Engine(
            model_path=model_path,
            base_gpu_id=base_gpu_id,
            random_seed=random_seed,
            mem_fraction_static=mem_fraction,
        )
        print(f"[SUCCESS] SGLang engine initialized successfully on GPU {base_gpu_id}!")
        return engine
    except Exception as e:
        print(f"[ERROR] Error initializing SGLang engine: {e}")
        return None


def main():
    """
    Main function that initializes the Accelerator and SGLang engine.
    """
    # Initialize the Accelerator - this handles distributed setup automatically
    accelerator = Accelerator()

    # Now the distributed environment should be properly initialized
    if torch.distributed.is_initialized():
        rank = accelerator.process_index
        world_size = accelerator.num_processes
        current_device = torch.cuda.current_device()
        print(f"[INFO] Process {rank} of {world_size} on device cuda:{current_device}")
    else:
        print("[WARNING] Distributed environment not properly initialized.")
        # Try to initialize manually as a fallback
        try:
            torch.distributed.init_process_group(backend="nccl")
            print("[INFO] Manually initialized distributed environment")
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        except Exception as e:
            print(f"[ERROR] Failed to initialize distributed environment: {e}")
            rank = 0  # Assume rank 0 for non-distributed setup
            world_size = 1

    # Only the main process (rank 0) initializes the SGLang engine
    if rank == 0:
        engine = initialize_sglang_engine(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            base_gpu_id=accelerator.local_process_index,  # Use local process index for GPU ID
            random_seed=42,
            mem_fraction=0.9,
        )
    else:
        print(f"[INFO] Process {rank} skipping SGLang engine initialization.")

    # Use torch.distributed.barrier to synchronize processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        print(f"[INFO] Process {rank} reached the barrier and is synchronized")


if __name__ == "__main__":
    main()
