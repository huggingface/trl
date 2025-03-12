import os
import sys
import yaml
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual distributed launcher for GRPO with SGLang"
    )
    parser.add_argument(
        "--config_file", type=str, required=True, help="YAML config file path"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to train"
    )
    parser.add_argument(
        "--reward_model",
        type=str,
        default="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        help="Reward model to use",
    )
    parser.add_argument(
        "--dataset", type=str, default="trl-internal-testing/zen", help="Dataset name"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="standard_prompt_only",
        help="Dataset config",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--mem_fraction", type=float, default=0.5, help="Memory fraction for SGLang"
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_environment(config):
    """Configure environment variables from YAML config."""
    # Set GPU visibility
    if "gpu_ids" in config:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_ids"]

    # Configure debugging
    if config.get("debug", False):
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # Extract num_processes without modifying GRPOConfig
    world_size = config.get("num_processes", torch.cuda.device_count())
    return min(world_size, torch.cuda.device_count())


def worker_process(
    rank, world_size, model, reward_model, dataset, config, mem_fraction
):
    """Worker function to run in each process."""
    try:
        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Set NCCL variables for stability
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

        # Set device for this process
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        logger.info(
            f"Process {rank}: Using device {device} ({torch.cuda.get_device_name(rank)})"
        )

        # Initialize distributed process group
        logger.info(f"Process {rank}: Initializing process group")
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:12355",
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"Process {rank}: Process group initialized")

        # Initialize SGLang engine on rank 0
        sglang_engine = None
        if rank == 0:
            logger.info(f"Process {rank}: Initializing SGLang engine")
            try:
                import sglang as sgl

                # Use base_gpu_id = world_size to avoid conflicts with training GPUs
                base_gpu_id = world_size
                logger.info(
                    f"Process {rank}: Using GPU {base_gpu_id} for SGLang engine"
                )

                # Clean memory on the target GPU
                with torch.cuda.device(base_gpu_id):
                    torch.cuda.empty_cache()

                # Initialize with conservative memory settings
                sglang_engine = sgl.Engine(
                    model_path=model,
                    base_gpu_id=base_gpu_id,
                    random_seed=42,
                    mem_fraction_static=mem_fraction,
                )
                logger.info(f"Process {rank}: SGLang engine initialized successfully")
            except Exception as e:
                logger.error(f"Process {rank}: Failed to initialize SGLang engine: {e}")
                import traceback

                traceback.print_exc()
                raise

        # Synchronize processes
        logger.info(f"Process {rank}: Waiting at barrier")
        dist.barrier()
        logger.info(f"Process {rank}: Passed barrier")

        # Create training arguments.
        # Notice checkpoint_path is now set to None to signal no external checkpoint.
        training_args = GRPOConfig(
            output_dir=config["output_dir"],
            learning_rate=1.0e-03,
            per_device_train_batch_size=3,
            num_generations=9,
            max_completion_length=32,
            report_to="none",
            use_sglang=True,
            sglang_base_gpu_id=world_size,
            sglang_mem_fraction_static=mem_fraction,
            checkpoint_path=None,  # No external checkpoint; use current model's weights.
        )

        # IMPORTANT: Set these attributes for manual distributed mode
        setattr(training_args, "_using_manual_distributed", True)
        setattr(training_args, "_sglang_engine", sglang_engine)

        # Create trainer
        logger.info(f"Process {rank}: Creating trainer")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_model,
            args=training_args,
            train_dataset=dataset,
        )

        # Run training
        logger.info(f"Process {rank}: Starting training")
        trainer.train()
        logger.info(f"Process {rank}: Training completed")

        # Final synchronization
        dist.barrier()

    except Exception as e:
        logger.error(f"Process {rank}: Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up resources
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"Process {rank}: Process group destroyed")


def main():
    """Main entry point for the script."""
    # Parse arguments
    args = parse_args()

    # Load YAML config
    yaml_config = load_config(args.config_file)

    # Prepare environment and get world size
    world_size = setup_environment(yaml_config)

    # Load dataset
    dataset = load_dataset(args.dataset, args.dataset_config, split="train")
    logger.info(f"Loaded dataset with {len(dataset)} examples")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    yaml_config["output_dir"] = output_dir

    # Configuration for worker processes
    config = {"output_dir": output_dir}

    # Set the multiprocessing start method
    mp.set_start_method("spawn", force=True)

    # Launch worker processes
    logger.info(f"Launching {world_size} worker processes")
    processes = []
    try:
        for rank in range(world_size):
            p = mp.Process(
                target=worker_process,
                args=(
                    rank,
                    world_size,
                    args.model,
                    args.reward_model,
                    dataset,
                    config,
                    args.mem_fraction,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for processes to complete
        for p in processes:
            p.join()

        # Check for failures
        failed = [rank for rank, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            logger.error(f"Training failed on processes: {failed}")
            sys.exit(1)
        else:
            logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.warning("Received keyboard interrupt, terminating processes")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
