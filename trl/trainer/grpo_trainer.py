# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import contextlib
import functools
import logging
import os
import sys
import textwrap
import time
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Sized, Union
from unittest.mock import patch

import sglang as sgl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import transformers
from accelerate import PartialState
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from ..extras.profiling import profiling_context, profiling_decorator
from ..import_utils import is_sglang_available, is_rich_available, is_vllm_available
from ..models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    (
    generate_model_card,
   
    get_comet_experiment_url,
   
    pad,
    print_prompt_completions_sample,
   
    selective_log_softmax,
),
)


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# print("NCCL_DEBUG =", os.environ.get("NCCL_DEBUG"))
# print("TORCH_DISTRIBUTED_DEBUG =", os.environ.get("TORCH_DISTRIBUTED_DEBUG"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SGLangDistributedManager:
    """
    Manages distributed training with SGLang integration to avoid conflicts between
    distributed communication and model serving.

    This class provides a robust interface for initializing SGLang in a separate process
    while ensuring proper synchronization with PyTorch distributed training.
    """

    def __init__(
        self,
        model_path: str,
        world_size: Optional[int] = None,
        master_addr: str = "localhost",
        master_port: str = "12355",
        backend: str = "nccl",
        timeout: int = 1800,
        max_restarts: int = 3,
    ):
        """
        Initialize the distributed manager.

        Args:
            model_path: Path to the model to load
            world_size: Number of processes to spawn (default: auto-detect GPUs)
            master_addr: Address for distributed coordination
            master_port: Port for distributed coordination
            backend: Distributed backend (nccl, gloo)
            timeout: Timeout in seconds for distributed operations
            max_restarts: Maximum number of restart attempts if processes fail
        """
        self.model_path = model_path
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.timeout = timeout
        self.max_restarts = max_restarts

        # Determine world size (number of processes)
        if world_size is None:
            self.world_size = torch.cuda.device_count()
            logger.info(f"Auto-detected {self.world_size} GPUs")
        else:
            self.world_size = min(world_size, torch.cuda.device_count())
            logger.info(f"Using {self.world_size} of {torch.cuda.device_count()} available GPUs")

        # Track processes and initialization status
        self.processes = []
        self.initialized = False

    def get_resource_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about available GPU resources.

        Returns:
            Dict mapping GPU IDs to resource information dictionaries
        """
        resources = {}
        for i in range(torch.cuda.device_count()):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(i)

                # Get memory usage
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                mem_allocated = torch.cuda.memory_allocated(i)
                mem_total = props.total_memory
                mem_free = mem_total - mem_allocated

                resources[i] = {
                    "name": props.name,
                    "total_memory": mem_total / (1024**3),  # Convert to GB
                    "free_memory": mem_free / (1024**3),  # Convert to GB
                    "allocated_memory": mem_allocated / (1024**3),  # Convert to GB
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            except Exception as e:
                resources[i] = {"error": str(e)}

        return resources

    def select_best_gpu_for_sglang(self) -> int:
        """
        Select the best GPU for SGLang initialization based on available memory.

        Returns:
            int: GPU ID with the most available memory
        """
        resources = self.get_resource_info()

        # First try to use a GPU not used for training (self.world_size and beyond)
        candidate_gpus = list(range(self.world_size, torch.cuda.device_count()))

        # If no additional GPUs, select from training GPUs based on memory
        if not candidate_gpus:
            candidate_gpus = list(range(torch.cuda.device_count()))

        # Find GPU with most free memory
        best_gpu = -1
        max_free_memory = 0

        for gpu_id in candidate_gpus:
            if gpu_id in resources and "free_memory" in resources[gpu_id]:
                free_memory = resources[gpu_id]["free_memory"]
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = gpu_id

        # Default to the last GPU if we couldn't determine the best one
        if best_gpu == -1:
            best_gpu = torch.cuda.device_count() - 1

        return best_gpu

    def init_process_group(self, rank: int) -> None:
        """
        Initialize the distributed process group.

        Args:
            rank: Process rank
        """
        # Set environment variables
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = self.master_port
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        # Set NCCL environment variables for better stability
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

        # Set device for this process
        torch.cuda.set_device(rank)

        # Initialize process group
        logger.info(f"[Rank {rank}] Initializing process group (backend={self.backend})")
        try:
            dist.init_process_group(
                backend=self.backend,
                init_method=f"tcp://{self.master_addr}:{self.master_port}",
                world_size=self.world_size,
                rank=rank,
                timeout=torch.timedelta(seconds=self.timeout),
            )
            logger.info(f"[Rank {rank}] Process group initialized successfully")
        except Exception as e:
            logger.error(f"[Rank {rank}] Failed to initialize process group: {e}")
            raise

    def init_sglang_engine(self, rank: int, base_gpu_id: int, mem_fraction: float, random_seed: int) -> Optional[Any]:
        """
        Initialize SGLang engine on rank 0.

        Args:
            rank: Process rank
            base_gpu_id: GPU ID for SGLang engine
            mem_fraction: Memory fraction for static allocation
            random_seed: Random seed for reproducibility

        Returns:
            Optional[Any]: SGLang engine if initialization succeeds, None otherwise
        """
        if rank != 0:
            logger.info(f"[Rank {rank}] Skipping SGLang engine initialization (only rank 0 initializes)")
            return None

        logger.info(
            f"[Rank {rank}] Initializing SGLang engine on GPU {base_gpu_id} with {mem_fraction} memory fraction"
        )

        try:
            # Create engine with specified parameters
            engine = sgl.Engine(
                model_path=self.model_path,
                base_gpu_id=base_gpu_id,
                random_seed=random_seed,
                mem_fraction_static=mem_fraction,
            )

            logger.info(f"[Rank {rank}] SGLang engine initialized successfully")
            return engine

        except Exception as e:
            logger.error(f"[Rank {rank}] Failed to initialize SGLang engine: {e}")
            import traceback

            traceback.print_exc()
            return None

    def cleanup(self, rank: int) -> None:
        """
        Clean up resources for process.

        Args:
            rank: Process rank
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"[Rank {rank}] Process group destroyed")

    def worker_process(self, rank: int, train_fn: callable, sglang_config: Dict[str, Any], args: Any) -> None:
        try:
            # Initialize process group
            self.init_process_group(rank)

            # Initialize SGLang engine on rank 0
            sglang_engine = None
            if rank == 0:
                sglang_engine = self.init_sglang_engine(
                    rank=rank,
                    base_gpu_id=sglang_config["base_gpu_id"],
                    mem_fraction=sglang_config["mem_fraction"],
                    random_seed=sglang_config["random_seed"],
                )

                # If SGLang initialization failed, we should abort
                if sglang_engine is None:
                    raise RuntimeError("SGLang engine initialization failed on rank 0")

                # **** Initialize the model update process group ****
                # Create a dedicated process group for online weight updates.
                update_group = torch.distributed.new_group(backend=self.backend)
                sglang_engine._model_update_group = update_group
                if hasattr(sglang_engine, "model_runner"):
                    sglang_engine.model_runner._model_update_group = update_group
                logger.info(
                    f"[Rank {rank}] Model update group initialized and assigned to SGLang engine and its ModelRunner."
                )
                # *************************************************************

            # Set custom attributes on args
            args._using_manual_distributed = True
            args._sglang_engine = sglang_engine

            # Synchronize processes after initialization
            logger.info(f"[Rank {rank}] Waiting at synchronization barrier")
            dist.barrier()
            logger.info(f"[Rank {rank}] Passed synchronization barrier")

            # Run training function
            train_fn(args)

            # Final synchronization
            dist.barrier()
            logger.info(f"[Rank {rank}] Training completed successfully")

        except Exception as e:
            logger.error(f"[Rank {rank}] Process failed with error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        finally:
            # Clean up resources
            self.cleanup(rank)

    def run(
        self,
        train_fn: callable,
        args: Any,
        sglang_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Launch distributed training with SGLang integration.

        Args:
            train_fn: Training function to run in each process
            args: Arguments to pass to the training function
            sglang_config: Configuration for SGLang engine (optional)
        """
        if self.initialized:
            logger.warning("Distributed manager already initialized, skipping")
            return

        self.initialized = True

        # Determine SGLang configuration
        if sglang_config is None:
            # Select best GPU for SGLang (different from training GPUs if possible)
            base_gpu_id = self.select_best_gpu_for_sglang()

            sglang_config = {
                "base_gpu_id": base_gpu_id,
                "mem_fraction": 0.9,
                "random_seed": 42,
            }

        logger.info(f"Launching distributed training with {self.world_size} processes")
        logger.info(f"SGLang engine will be initialized on GPU {sglang_config['base_gpu_id']}")

        # Set multiprocessing start method
        mp.set_start_method("spawn", force=True)

        # Launch processes with retry logic
        for attempt in range(self.max_restarts + 1):
            try:
                processes = []

                for rank in range(self.world_size):
                    p = mp.Process(
                        target=self.worker_process,
                        args=(rank, train_fn, sglang_config, args),
                    )
                    p.start()
                    processes.append(p)

                # Wait for all processes to complete
                for p in processes:
                    p.join()

                # Check for failures
                exit_codes = [p.exitcode for p in processes]
                if any(code != 0 for code in exit_codes):
                    failed_ranks = [rank for rank, code in enumerate(exit_codes) if code != 0]
                    logger.error(f"Training failed on ranks: {failed_ranks}")
                    raise RuntimeError(f"Training failed on ranks: {failed_ranks}")

                # Success
                logger.info("All processes completed successfully")
                break

            except Exception as e:
                if attempt < self.max_restarts:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    # Clean up before retry
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    # Wait before retrying
                    time.sleep(5)
                else:
                    logger.error(f"All {self.max_restarts} retry attempts failed. Giving up.")
                    raise


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    @classmethod
    def train_with_manual_distributed(
        cls,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        """
        Run GRPO training with manual distributed process management for SGLang.

        This method uses a robust distributed manager to handle process initialization,
        SGLang engine setup, and synchronization between processes, avoiding conflicts
        between torch.distributed and SGLang.

        Args:
            model: Model ID or instance to train
            reward_funcs: Reward functions for GRPO
            args: Training configuration
            train_dataset: Dataset for training
            eval_dataset: Dataset for evaluation (optional)
            processing_class: Tokenizer for processing inputs (optional)
            reward_processing_classes: Tokenizers for reward functions (optional)
            callbacks: Training callbacks (optional)
            optimizers: Optimizer and scheduler (optional)
            peft_config: PEFT configuration (optional)
        """
        # Process args if needed
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Store trainer kwargs as attribute on args
        trainer_kwargs = {
            "model": model,
            "reward_funcs": reward_funcs,
            "args": args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "processing_class": processing_class,
            "reward_processing_classes": reward_processing_classes,
            "callbacks": callbacks,
            "optimizers": optimizers,
            "peft_config": peft_config,
        }
        args._trainer_kwargs = trainer_kwargs

        # Get model path
        model_path = model if isinstance(model, str) else model.config._name_or_path

        # Configure SGLang
        sglang_config = {
            "base_gpu_id": getattr(args, "sglang_base_gpu_id", None),
            "mem_fraction": getattr(args, "sglang_mem_fraction_static", 0.9),
            "random_seed": getattr(args, "seed", 42),
        }

        # Define training function
        def train_function(args):
            # Retrieve trainer arguments
            kwargs = args._trainer_kwargs

            # Initialize trainer
            trainer = cls(**kwargs)

            # Run training
            trainer.train()

        # Create and run distributed manager
        world_size = min(
            torch.cuda.device_count(),
            getattr(args, "num_processes", torch.cuda.device_count()),
        )

        manager = SGLangDistributedManager(
            model_path=model_path,
            world_size=world_size,
            timeout=1800,  # 30 minutes timeout for operations
            max_restarts=2,  # Allow 2 retry attempts
        )

        # Run training with distributed manager
        manager.run(train_function, args, sglang_config)

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # First, check if we're using manual distributed mode
        self._using_manual_distributed = getattr(args, "_using_manual_distributed", False)
        self._sglang_engine = getattr(args, "_sglang_engine", None)

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.use_vllm = args.use_vllm
        self.use_sglang = getattr(args, "use_sglang", False)  # Add backend selection flag

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        # print("[DEBUG] Seed set to", args.seed)

        # Modified SGLang initialization
        if self.use_sglang:
            if not is_sglang_available():
                raise ImportError(
                    "SGLang is not available and `use_sglang` is set to True. "
                    "please install sglang according to the docs."
                )

            # Log distributed environment variables for debugging
            # print(f"[DEBUG] LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
            # print(f"[DEBUG] RANK: {os.environ.get('RANK')}")
            # print(f"[DEBUG] WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
            # print(
            #     f"[DEBUG] torch.distributed.is_initialized(): {torch.distributed.is_initialized()}"
            # )

            # If using manual distributed mode, the engine is already initialized
            if self._using_manual_distributed:
                if self.accelerator.is_main_process:
                    # print(
                    #     "[DEBUG] Using pre-initialized SGLang engine from manual distributed mode"
                    # )
                    self.sglang_engine = self._sglang_engine
                    self.sglang_sampling_params = {
                        "temperature": args.temperature,
                        "max_new_tokens": self.max_completion_length,
                    }
            else:
                # Original SGLang initialization code
                local_device = torch.device(f"cuda:{self.accelerator.process_index}")
                torch.cuda.set_device(local_device)
                # print(
                #     f"[DEBUG] Process rank {self.accelerator.process_index} set to device {local_device}"
                # )

                if torch.distributed.is_initialized():
                    current_device = torch.cuda.current_device()
                    # print(
                    #     f"[DEBUG] Process rank {self.accelerator.process_index} calling torch.distributed.barrier() on device {current_device}"
                    # )
                    torch.distributed.barrier(device_ids=[current_device])
                    # print(
                    #     f"[DEBUG] Process rank {self.accelerator.process_index} passed torch.distributed.barrier()"
                    # )

                # Initialize SGLang engine in main process
                if self.accelerator.is_main_process:
                    base_gpu_id = args.sglang_base_gpu_id
                    if base_gpu_id is None:
                        base_gpu_id = self.accelerator.num_processes
                    # print(
                    #     f"[DEBUG] [Main Process] Initializing SGLang Engine with base_gpu_id = {base_gpu_id}"
                    # )
                    try:
                        self.sglang_engine = sgl.Engine(
                            model_path=model_id,
                            base_gpu_id=base_gpu_id,
                            random_seed=args.seed,
                            mem_fraction_static=getattr(args, "sglang_mem_fraction_static", 0.9),
                        )
                        self.sglang_sampling_params = {
                            "temperature": args.temperature,
                            "max_new_tokens": self.max_completion_length,
                        }
                        # print(
                        #     "[DEBUG] [Main Process] SGLang engine initialized successfully."
                        # )
                    except Exception as e:
                        logger.error(f"[ERROR] [Main Process] Error initializing SGLang engine: {e}")
                        import traceback

                        traceback.print_exc()
                        raise e

                # print(
                #     f"[DEBUG] Process rank {self.accelerator.process_index} calling accelerator.wait_for_everyone()"
                # )
                self.accelerator.wait_for_everyone()
                # print(
                #     f"[DEBUG] Process rank {self.accelerator.process_index} passed accelerator.wait_for_everyone()"
                # )
        elif self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                device_type = PartialState().default_device.type
                device_module = getattr(torch, device_type)
                if vllm_device == "auto":
                    if device_module.device_count() == 1:
                        vllm_device = f"{device_type}:0"  # particular case when training with onyl 1 device: share it
                    else:
                        vllm_device = f"{device_type}:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == f"{device_type}"
                    and int(vllm_device.split(":")[1]) >= device_module.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {device_module.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"{device_type}:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )

                # For Ascend NPU (torch-npu), collective communication requires the establishment of a communication
                # group, and different processes must hold the same group number. However, multiple process groups will
                # be created internally within vLLM. This will cause the group id of the communication group on rank 0
                # to be different from that of other ranks, causing backward to hang on because the communication
                # domain cannot be established. So we need to patch it to make sure the group id of different ranks in
                # the training phase are the same.
                @contextlib.contextmanager
                def new_group_context():
                    new_group = torch.distributed.new_group
                    try:
                        torch.distributed.new_group = functools.partial(new_group, use_local_synchronization=True)
                        torch.npu.mem_get_info = functools.partial(torch.npu.mem_get_info, device=vllm_device)
                        yield
                    finally:
                        torch.distributed.new_group = new_group

                new_group_patch = new_group_context() if device_type == "npu" else contextlib.nullcontext()
                with world_size_patch, profiling_patch, new_group_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None

                # Sampling parameters
                self.sampling_params = SamplingParams(
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=-1 if args.top_k is None else args.top_k,
                    min_p=0.0 if args.min_p is None else args.min_p,
                    repetition_penalty=args.repetition_penalty,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def _update_sglang_engine_weights(self):
        """Update the SGLang engine weights from the current model."""
        is_main = (
            hasattr(self, "_using_manual_distributed")
            and self._using_manual_distributed
            and dist.get_rank() == 0
            or not getattr(self, "_using_manual_distributed", False)
            and self.accelerator.is_main_process
        )

        if not is_main:
            return

        try:
            checkpoint = self.args.checkpoint_path

            if checkpoint and os.path.exists(checkpoint):
                # Case 1: Valid checkpoint path exists - update from disk
                success, message = self.sglang_engine.update_weights_from_disk(checkpoint)
                if not success:
                    raise RuntimeError(f"Failed to update weights from {checkpoint}: {message}")
                # print(
                #     f"SGLang engine weights updated successfully from checkpoint: {checkpoint}"
                # )
            else:
                # Case 2: No valid checkpoint path - use current model weights
                # print(
                #     "No valid checkpoint path provided. Using current model weights..."
                # )

                # Extract state dictionary from the model
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        unwrapped_model = unwrapped_model._orig_mod

                    # Handle PEFT models if applicable
                    if is_peft_model(unwrapped_model):
                        unwrapped_model.merge_adapter()
                        state_dict = unwrapped_model.state_dict()
                        state_dict = {
                            k.removeprefix("base_model.model.").replace(".base_layer", ""): v
                            for k, v in state_dict.items()
                        }
                        state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                        state_dict = {
                            k.replace("modules_to_save.default.", ""): v
                            for k, v in state_dict.items()
                            if "original_module" not in k
                        }
                    else:
                        state_dict = unwrapped_model.state_dict()

                # Use update_weights_from_tensor instead of creating a temporary file
                named_tensors = list(state_dict.items())
                try:
                    success = self.sglang_engine.update_weights_from_tensor(named_tensors)
                except Exception as err:
                    logger.error(
                        f"Warning: update_weights_from_tensor failed with error: {err}\n"
                        "Falling back to current model state without update."
                    )
                    success = True  # Assume fallback is acceptable

                if not success:
                    raise RuntimeError("Failed to update weights from tensors")

                logger.info("SGLang engine weights updated successfully from current model state")

        except Exception as e:
            logger.error(f"Error updating SGLang engine weights: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to update SGLang engine weights: {e}") from e

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # Determine process rank/index based on initialization mode
        if hasattr(self, "_using_manual_distributed") and self._using_manual_distributed:
            process_rank = dist.get_rank()
            world_size = dist.get_world_size()
            is_main_process = process_rank == 0
        else:
            process_rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            is_main_process = self.accelerator.is_main_process

        # Get appropriate device
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        # Process prompts
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.use_sglang:
            # If the global step has advanced, update the SGLang engine weights
            if self.state.global_step != getattr(self, "_last_loaded_step", -1) and is_main_process:
                logger.info(
                    f"[Rank {process_rank}] Updating SGLang engine weights at global step {self.state.global_step}"
                )
                self._update_sglang_engine_weights()
                self._last_loaded_step = self.state.global_step

            # Gather prompt texts from all processes
            if self._using_manual_distributed:
                # Use PyTorch distributed for manual mode
                all_prompts = [None] * world_size
                dist.all_gather_object(all_prompts, prompts_text)
                all_prompts_text = []
                for prompts in all_prompts:
                    all_prompts_text.extend(prompts)
            else:
                # Use Accelerate for standard mode
                all_prompts_text = gather_object(prompts_text)

            logger.info(f"[Rank {process_rank}] Collected {len(all_prompts_text)} prompts for generation")

            # Generate completions on the main process
            if is_main_process:
                logger.info(f"[Rank {process_rank}] Sending generation request to SGLang engine")
                try:
                    outputs = self.sglang_engine.generate(all_prompts_text, self.sglang_sampling_params)
                    logger.info(f"[Rank {process_rank}] Received generation responses")
                except Exception as e:
                    logger.error(f"[Rank {process_rank}] Error during generation: {e}")
                    import traceback

                    traceback.print_exc()
                    raise e

                generated_texts = [output["text"] for output in outputs]
                completion_ids = [self.processing_class.encode(text) for text in generated_texts]
            else:
                completion_ids = [None] * len(all_prompts_text)

            # Broadcast completions to all processes
            if self._using_manual_distributed:
                # Use PyTorch distributed for manual mode
                if is_main_process:
                    # Convert tensors to lists for serialization
                    completion_ids_list = [
                        ids.tolist() if isinstance(ids, torch.Tensor) else ids for ids in completion_ids
                    ]
                    # Broadcast from rank 0
                    dist.broadcast_object_list([completion_ids_list], src=0)
                else:
                    # Receive broadcast
                    tmp_list = [None]
                    dist.broadcast_object_list(tmp_list, src=0)
                    completion_ids_list = tmp_list[0]
                    completion_ids = completion_ids_list
            else:
                # Use Accelerate for standard mode
                completion_ids = broadcast_object_list(completion_ids, from_process=0)

            # Get the slice for this process
            process_slice = slice(
                process_rank * len(prompts),
                (process_rank + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Prepare completion tensors
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        elif self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=self.sampling_params, use_tqdm=False
                    )
                completion_ids = []
                for outputs in all_outputs:
                    for output in outputs.outputs:
                        completion_ids.append(output.token_ids)
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=(wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None),
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def __del__(self):
        """Ensure proper cleanup when the trainer is destroyed."""
        if hasattr(self, "sglang_engine") and self.accelerator.is_main_process:
            try:
                self.sglang_engine.shutdown()
                logger.info("SGLang engine shut down.")
            except Exception as e:
                logger.warning(f"Warning: Error shutting down SGLang engine: {e}")

    def __getstate__(self):
        """Custom state management for trainer serialization.

        Ensures proper checkpoint saving by removing unpicklable SGLang objects.

        Returns:
            dict: Serializable state dictionary
        """
        state = self.__dict__.copy()

        # Remove unpicklable SGLang engine and related ZMQ objects
        if "_sglang_engine" in state:
            del state["_sglang_engine"]

        # Remove the sglang_engine from the args if present
        # (this is a backup in case the config's __getstate__ isn't called)
        if "args" in state and hasattr(state["args"], "_sglang_engine"):
            args_copy = state["args"].__dict__.copy()
            if "_sglang_engine" in args_copy:
                del args_copy["_sglang_engine"]

            # Create a new args object without the engine
            from copy import deepcopy

            state["args"] = deepcopy(state["args"])
            state["args"].__dict__ = args_copy

        return state

    def __setstate__(self, state):
        """Restore trainer state after deserialization.

        Args:
            state (dict): State dictionary to restore
        """
        self.__dict__.update(state)

        # SGLang engine will need to be reinitialized if needed
        # This happens automatically in training_step when using manual_distributed
        if not hasattr(self, "_sglang_engine"):
            self._sglang_engine = None
