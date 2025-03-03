import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sglang as sgl
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class GRPOConfig:
    """Configuration for GRPO with SGLang.

    Attributes:
        model_path: Path to the model to load in SGLang
        master_addr: Address of the master node
        master_port: Port of the master node
        num_gpus: Number of GPUs to use
        random_seed: Random seed for reproducibility
        mem_fraction: Memory fraction for SGLang
    """

    model_path: str
    master_addr: str = "localhost"
    master_port: str = "12355"
    num_gpus: Optional[int] = None
    random_seed: int = 42
    mem_fraction: float = 0.9


def setup_distributed(rank: int, config: GRPOConfig) -> None:
    """Initialize the distributed environment.

    Args:
        rank: Current process rank
        config: Configuration parameters
    """
    # Determine world size
    world_size = config.num_gpus or torch.cuda.device_count()

    # Set environment variables
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # Set device for this process
    torch.cuda.set_device(rank)

    # Initialize process group
    print(f"[Rank {rank}] Initializing process group")
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{config.master_addr}:{config.master_port}",
            world_size=world_size,
            rank=rank,
        )
        print(f"[Rank {rank}] Process group initialized successfully")
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize process group: {e}")
        raise e


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("[Cleanup] Destroyed process group")


def initialize_sglang(rank: int, config: GRPOConfig) -> Optional[sgl.Engine]:
    """Initialize the SGLang engine (rank 0 only).

    Args:
        rank: Current process rank
        config: Configuration parameters

    Returns:
        Optional[sgl.Engine]: Initialized engine or None
    """
    if rank != 0:
        print(f"[Rank {rank}] Skipping SGLang initialization")
        return None

    print(f"[Rank {rank}] Initializing SGLang engine")
    try:
        engine = sgl.Engine(
            model_path=config.model_path,
            base_gpu_id=rank,
            random_seed=config.random_seed,
            mem_fraction_static=config.mem_fraction,
        )
        print(f"[Rank {rank}] SGLang engine initialized successfully")
        return engine
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize SGLang engine: {e}")
        return None


def run_grpo_process(rank: int, config: GRPOConfig) -> None:
    """Main function to run in each process.

    Args:
        rank: Current process rank
        config: Configuration parameters
    """
    try:
        # Set up distributed environment
        setup_distributed(rank, config)

        # Synchronize processes after initialization
        print(f"[Rank {rank}] Waiting at first barrier")
        dist.barrier()
        print(f"[Rank {rank}] Passed first barrier")

        # Initialize SGLang (rank 0 only)
        engine = initialize_sglang(rank, config)

        # Synchronize after SGLang initialization
        print(f"[Rank {rank}] Waiting at second barrier")
        dist.barrier()
        print(f"[Rank {rank}] Passed second barrier")

        # Simulate GRPO training (in a real implementation, this would use TRL)
        if rank == 0:
            print("[GRPO] Simulating GRPO training...")
            if engine is not None:
                # Example: Run a test query with SGLang
                result = engine.generate(
                    sgl.Prompt("What is machine learning?"),
                    sampling_params={"max_tokens": 50},
                )
                print(f"[GRPO] Test response: {result[0].outputs[0].text}")

        # Final synchronization
        print(f"[Rank {rank}] Waiting at final barrier")
        dist.barrier()
        print(f"[Rank {rank}] All processes completed successfully")

    except Exception as e:
        print(f"[Rank {rank}] Failed with error: {e}")
    finally:
        # Clean up
        cleanup_distributed()


def main() -> None:
    """Script entry point."""
    # Configuration
    config = GRPOConfig(
        model_path="Qwen/Qwen2.5-0.5B-Instruct", num_gpus=3  # Use at most 3 GPUs
    )

    # Adjust num_gpus if needed
    available_gpus = torch.cuda.device_count()
    if config.num_gpus is None or config.num_gpus > available_gpus:
        config.num_gpus = min(available_gpus, 3)

    print(f"Using {config.num_gpus} GPUs for distributed GRPO")

    # Spawn processes
    mp.spawn(run_grpo_process, args=(config,), nprocs=config.num_gpus, join=True)


if __name__ == "__main__":
    # Set spawn method for better error reporting
    mp.set_start_method("spawn")
    main()
