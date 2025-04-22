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
import os
import random
import textwrap
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import numpy as np
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational
from ..environment.env_protocol import Environment
from ..import_utils import is_vllm_available
from ..models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        shuffle (`bool`, defaults to True):
            If True, randomizes the order of indices. If False, maintains sequential order.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler when shuffle=True).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2, shuffle=True)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2, shuffle=False)
    >>> list(sampler)
    [0, 0, 1, 1, 2, 2, 3, 3]
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        repeat_count: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            indexes = [
                idx
                for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
                for _ in range(self.repeat_count)
            ]
        else:
            indexes = [idx for idx in range(self.num_samples) for _ in range(self.repeat_count)]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class SSRBuffer:
    '''
    Selective Sample Replay manager. Maintains a buffer of high entropy samples for training.
    '''

    def __init__(self, alpha: float = 2.0, total_buffer_size: int = 1000, persist_steps: int = 1000):
        '''
        Args:
            alpha: float, handles prioritization intensity>=0.
                alpha = 0 means no prioritization,
                alpha = 1 means prioritization linearly proportional to advantage,
                alpha > 1 means more prioritization for high entropy samples.
            total_buffer_size: int, maximum size of the buffer. After the buffer is full, the oldest samples will be discarded.
            persist_steps: int, number of steps an example lives in the buffer. After this many steps, the example will be discarded.
        '''

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        self.alpha = alpha

        if total_buffer_size <= 0:
            raise ValueError("total_buffer_size must be greater than 0")
        self.total_buffer_size = total_buffer_size

        if persist_steps <= 0:
            raise ValueError("persist_steps must be greater than 0")
        self.persist_steps = persist_steps

        self.buffer = []

        # element of buffer format:
        # {
        #      "example": dict, - the example to be replayed
        #      "advantage": float, - the advantage observed last training step
        #      "ttl": int, - time to live, number of steps before the example will be discarded from the buffer
        # }

    def add_example(self, example: dict, advantage: float) -> None:
        '''
        Add an example to the buffer.
        '''
        # NOTE: We don't check if the buffer is full here. We'll do it at the end of each training step.
        buffer_element = {
            "example": example,
            "advantage": advantage,
            "ttl": self.persist_steps
        }
        self.buffer.append(buffer_element)

    @property
    def buffer_size(self) -> int:
        '''
        Number of examples in the buffer.
        '''
        return len(self.buffer)

    def draw_example(self) -> dict:
        '''
        Returns an example from the buffer. The probabilty of drawing an example j is:
        abs(advantage_j)**(self.alpha) / sum(abs(advantage_i)**(self.alpha) for i in range(len(self.buffer)))
        
        Raises a ValueError if the buffer is empty, otherwise, pops an example from the buffer and returns it.
        '''

        if self.buffer_size == 0:
            raise ValueError("Buffer is empty")

        values = []
        for buffer_element in self.buffer:
            values.append(abs(buffer_element["advantage"])**self.alpha)

        total = sum(values)
        probabilities = [value / total for value in values]

        # check that the probabilities sum to 1, with some tolerance
        if not np.isclose(sum(probabilities), 1.0, atol=1e-6):
            raise ValueError(f"Probabilities do not sum to 1, but instead sum to {sum(probabilities)}")

        # choose the index of the example to draw
        index = np.random.choice(range(len(self.buffer)), p=probabilities)

        # pop the example from the buffer
        buffer_element = self.buffer.pop(index)

        return buffer_element['example']

    def step(self) -> None:
        '''
        Handles reducing ttl's on objects in the buffer and removes objects that have expired.
       
        It is to be called once at the end of training step.
        '''
        # decrement the ttl of each buffer element
        for buffer_element in self.buffer:
            buffer_element['ttl'] -= 1

        # remove buffer elements that have expired
        self.buffer = [b for b in self.buffer if b['ttl'] > 0]

        # if the buffer is too big, discard the oldest examples
        if len(self.buffer) > self.buffer_size:
            # Sort by absolute advantage (priority), ascending
            self.buffer.sort(key=lambda x: abs(x['advantage']))
            # Keep only the top 'buffer_size' elements (highest priority)
            self.buffer = self.buffer[-self.buffer_size:]


class QwenGRPOTrainer(Trainer):
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
        model (`[PreTrainedModel]`):
            Model to be trained. A [`~transformers.PreTrainedModel`] object. Only Qwen2VL and Qwen2.5VL models are supported.
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
        processing_class ([`~transformers.PreTrainedTokenizerBase`]:
            Processing class used to process the data. The padding side must be set to "left".
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
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset when creating the sampler. If False, the dataset will be trained in order. Useful for curriculm learning.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        processing_class: PreTrainedTokenizerBase,
        env: Environment,
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        shuffle_dataset: bool = True,
        image_pad_id: int = 151655,
        inputs_to_log: list[str] = [],
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_id = model.config._name_or_path

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            # NOTE: unpacking the args is super hacky.

            # Extract model path from config if needed
            model_init_kwargs_dict = model_init_kwargs.__dict__

            ref_model_path = model_init_kwargs_dict["model_name_or_path"]
            ref_model_torch_dtype = model_init_kwargs_dict["torch_dtype"]
            use_peft = model_init_kwargs_dict["use_peft"]
            if use_peft:
                raise ValueError("PEFT is not supported in DeepSpeed Zero3 yet.")

            attn_implementation = model_init_kwargs_dict["attn_implementation"]

            if "Qwen2-VL" in ref_model_path:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    ref_model_path,
                    torch_dtype=ref_model_torch_dtype,
                    attn_implementation=attn_implementation,
                )
            elif "Qwen2.5-VL" in ref_model_path:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    ref_model_path,
                    torch_dtype=ref_model_torch_dtype,
                    attn_implementation=attn_implementation,
                )
            else:
                raise ValueError("The base model you provided was unexpected. Expected a Qwen2-VL or Qwen2.5-VL.")

            self.ref_model.use_cache = False

        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

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
            # Validate types (float or callable)
            for weight in args.reward_weights:
                if not isinstance(weight, float) and not callable(weight):
                    raise TypeError(f"Reward weights must be floats or callables, but found {type(weight)}")
            self.reward_weights_config = args.reward_weights # Store the original list/config
        else:
            # Default to list of 1.0 floats
            self.reward_weights_config = torch.ones(len(reward_funcs), dtype=torch.float32)

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
        self.use_vllm = args.use_vllm
        self.shuffle_dataset = shuffle_dataset

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        # intialize epsilon
        self.epsilon_low = args.epsilon_low
        self.epsilon_high = args.epsilon_high


        # TODO: make these configurable args
        self.use_ssr_buffer = True
        self.ssr_alpha = 2.0
        self.ssr_total_buffer_size = 10000
        self.ssr_persist_steps = 100000
        # if the buffer is smaller than this, we don't use it. Instead, draw from the dataset. This helps ensure we only select the best quality examples from the buffer on average.
        self.min_ssr_buffer_size = 50
        # the probability of using the SSR buffer on each step
        self.max_ssr_use_prob = 0.9

        if not 0 <= self.max_ssr_use_prob <= 1:
            raise ValueError("max_ssr_use_prob must be between 0 and 1")

        if self.use_ssr_buffer:
            self.ssr_buffer = SSRBuffer(alpha=self.ssr_alpha, total_buffer_size=self.ssr_total_buffer_size, persist_steps=self.ssr_persist_steps)
        else:
            self.ssr_buffer = None

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

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        print("Only one GPU available, sharing it between vLLM and training.")
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                        print(f"Using GPU {vllm_device} for vLLM.")

                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
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
                with world_size_patch, profiling_patch:
                    self.vlm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        max_model_len=self.args.vllm_max_model_len,
                        # Setting this to 1 as we only have one image per prompt for now. Setting it longer requires more resources, which is wasteful until we need it.
                        limit_mm_per_prompt={"image": self.args.limit_image_per_prompt, "video": self.args.limit_video_per_prompt},
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError("use_vllm must be True")

        self.env = env

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

        self.image_pad_id = image_pad_id
        self.inputs_to_log = inputs_to_log

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(
            self.train_dataset,
            self.num_generations,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(
            eval_dataset,
            self.num_generations,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        logits_to_keep,
    ):
        # NOTE: Flash attention is not supported here yet as our sequences are right padded.
        # Get logits for full sequence
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits

        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        # Only keep the logits for the completion portion
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]

        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            elif isinstance(unwrapped_model, PeftModel):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                unwrapped_model.unmerge_adapter()
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v
                    for k, v in state_dict.items()
                    if self.model.prefix not in k
                }
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            vlm_model = self.vlm.llm_engine.model_executor.driver_worker.model_runner.model
            vlm_model.load_weights(state_dict.items())

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:


        device = self.accelerator.device

        if not self.env:
            raise ValueError("No environment provided. Only supporting envs now. ")

        if self.use_ssr_buffer:
            # each process needs to know if we are using the buffer. Process 0 decides and then broadcasts the decision to all processes
            if self.accelerator.process_index == 0:
                print(f"buffer size: {self.ssr_buffer.buffer_size}")
                # step the buffer, needs to happen at each training step
                self.ssr_buffer.step()

                buffer_size = self.ssr_buffer.buffer_size
                if buffer_size >= self.min_ssr_buffer_size:
                    # Calculate dynamic probability based on buffer size
                    size_range = self.ssr_total_buffer_size - self.min_ssr_buffer_size
                    if size_range > 0: # Avoid division by zero if min and total size are the same
                        prob_ramp = (buffer_size - self.min_ssr_buffer_size) / size_range
                        current_ssr_use_prob = min(self.max_ssr_use_prob, max(0.0, prob_ramp))
                    else: # If min and total size are the same, use max probability if buffer is at least min size
                        current_ssr_use_prob = self.max_ssr_use_prob
                else:
                     # If buffer is smaller than min size, probability is 0
                    current_ssr_use_prob = 0.0

                print(f"Current SSR use probability: {current_ssr_use_prob:.4f}")
                should_use_buffer = random.random() < current_ssr_use_prob and buffer_size >= self.min_ssr_buffer_size

                should_use_buffer_list = [should_use_buffer for _ in range(self.accelerator.num_processes)]
            else:
                should_use_buffer_list = [None for _ in range(self.accelerator.num_processes)]
                # Non-zero processes also need the flag, although they don't decide it
                buffer_size = None # Placeholder for non-zero processes

            broadcast_object_list(should_use_buffer_list, from_process=0)
            should_use_buffer = should_use_buffer_list[0] # All processes now know if buffer is used

            if should_use_buffer:
                # process 0 will draw from the buffer, the other processes will hang out
                if self.accelerator.process_index == 0:

                    # put the current example in the buffer with a small advantage so we avoid "throwing it away"
                    self.ssr_buffer.add_example(inputs[0], 0.01)

                    print("Drawing from buffer")
                    example_from_buffer = self.ssr_buffer.draw_example()
                    local_inputs = [deepcopy(example_from_buffer) for _ in range(len(inputs))]
                    print(f"local_inputs after drawing from buffer length: {len(local_inputs)=}")
                else:
                    local_inputs = [None for _ in range(len(inputs))]

                # broadcast the inputs (from the buffer) to all processes
                broadcast_object_list(local_inputs, from_process=0)
                inputs = local_inputs
        else:
            # if we are not using the SSR buffer, we just use the inputs passed into the function and signal that this example is not from the buffer
            should_use_buffer = False
            buffer_size = 0 # Ensure buffer_size is defined even if not using SSR buffer

        print(f"should_use_buffer: {should_use_buffer}")



        # TODO: This is a hack that we should probably fix.
        # without this, each gpu receives different inputs, screwing up the advantage computation.
        # Simple synchronization of inputs across processes
        if self.accelerator.num_processes > 1:
            # Make sure all processes have a non-None value to gather
            # Use an empty list for non-main processes
            local_inputs = inputs if self.accelerator.process_index == 0 else []

            # Gather from all processes using torch.distributed.gather_object
            all_inputs = gather_object(local_inputs)

            # each process takes the inputs from process 0 as its inputs
            inputs = deepcopy(all_inputs)

        self.accelerator.wait_for_everyone()

        # conversations: list of conversations
        # prompts_text: list of prompts as strings
        # prompt_inputs: tokenized data (with image tokens injected) that we will use to compute log probs on the base model.
        # env_inputs: data in the format our env/vllm expects
        conversations, prompts_text, prompt_inputs, env_inputs = self.env.prepare_data(
            inputs=inputs, processing_class=self.processing_class
        )

        # unpack prompt_inputs
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask, pixel_values, image_grid_thw = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
            prompt_inputs["pixel_values"],
            prompt_inputs["image_grid_thw"],
        )

        if self.max_prompt_length is not None:
            raise ValueError("max_prompt_length is not supported.")

        # Generate completions using vLLM
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_env_inputs = gather_object(env_inputs)
            all_conversations = gather_object(conversations)

            if self.accelerator.is_main_process:
                if self.env is None:
                    raise ValueError("No environment provided. Only supporting envs now.")
                else:
                    generated_output = self.env.generate(
                        conversations=all_conversations,
                        vlm_inputs=all_env_inputs,
                        vlm=self.vlm,
                        sampling_params=self.sampling_params,
                    )

                    completion_ids = generated_output['ids']
                    completion_messages = generated_output.get('messages', None)
                    completion_mask = generated_output.get('mask', None)


            else:
                completion_ids = [None] * len(all_env_inputs)
                completion_messages = [None] * len(all_env_inputs)
                completion_mask = [None] * len(all_env_inputs)

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            completion_ids = completion_ids[process_slice]

            # Pad completion_ids to uniform length, mask from last output token (EOS)
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.tokenizer.pad_token_id)

            # broadcast and slice completion messages too.
            completion_messages = broadcast_object_list(completion_messages, from_process=0)
            completion_messages = completion_messages[process_slice]

            # Handle completion mask: broadcast from main process to all processes if available
            if completion_mask is not None:
                # Broadcast the completion_mask from the main process to all processes
                completion_mask = broadcast_object_list(completion_mask, from_process=0)

                # Each process takes its corresponding slice based on process index
                completion_mask = completion_mask[process_slice]

                # Convert mask elements to tensors and move to correct device
                completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
                # Pad masks to uniform length
                completion_mask = pad(completion_mask, padding_value=0)
            else:
                print("No completion mask provided. Computing mask based on EOS positions.")
                # Fallback: compute mask based on EOS positions if not provided
                eos_idx = torch.tensor([len(ids) - 1 for ids in completion_ids], device=device)
                sequence_indices = torch.arange(completion_ids.size(1), device=device).expand(completion_ids.size(0), -1)
                completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            # Handle the potential new images generated from the environment (tool) in completion_messages
            new_images = []
            for i, completion_message in enumerate(completion_messages):
                if completion_message is not None:
                    for message in completion_message:
                        for content in message["content"]:
                            if content.get("type", None) == "image":
                                new_images.append(content["image"])

            if len(new_images) > 0:
                # use the processor to get pixel_values and image_grid_thw for the new images
                new_images_info = self.processing_class(
                    text='',
                    images=new_images,
                    return_tensors='pt',
                    padding=True,
                )
                new_pixel_values = new_images_info["pixel_values"]
                new_image_grid_thw = new_images_info["image_grid_thw"]

                # Concatenate the new pixel_values and image_grid_thw with the existing ones
                # make sure pixel_values and new_pixel_values are on the same device. same for image_grid_thw and new_image_grid_thw
                new_pixel_values = new_pixel_values.to(device)
                new_image_grid_thw = new_image_grid_thw.to(device)
                pixel_values = torch.cat([pixel_values, new_pixel_values], dim=0)
                image_grid_thw = torch.cat([image_grid_thw, new_image_grid_thw], dim=0)
        else:
            raise ValueError("Attempted to generate with HF. Only supporting vllm now.")

        print("Finished with generation")

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep,
                    )

        print("Finished with ref logits")

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(conversations, completions_text):
                bootstrap = prompt[-1]["content"] if prompt[-1]["role"] == "assistant" else ""
                if isinstance(bootstrap, list):
                    if len(bootstrap) > 1:
                        raise ValueError("Only one bootstrap is supported for now.")
                    bootstrap = bootstrap[0]["text"]

                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(conversations), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                raise NotImplementedError("Models as reward functions are not supported yet.")
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(conversations, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(conversations, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                reward_kwargs["prompts_text"] = prompts_text
                reward_kwargs["completions_messages"] = completion_messages
                output_reward_func = reward_func(prompts=conversations, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        print("Finished with rewards per function")
        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        print(f"Finished with gathering rewards per function, {rewards_per_func=}, {self.accelerator.process_index=}")

        self.accelerator.wait_for_everyone()
        print(f"Finished with waiting for everyone, {self.accelerator.process_index=}")

        # # DEBUG: Verify prompt consistency across completions in each group
        # TODO: remove this probably?
        # if self.accelerator.is_main_process:
        #     all_prompts = gather_object(prompts_text)

        #     if not len(all_prompts) == self.num_generations:
        #         raise ValueError(
        #             f"We should have one prompt per generation, but we have {len(all_prompts)} prompts and {self.num_generations} generations"
        #         )
        #     if not len(set(all_prompts)) == 1:
        #         raise ValueError(f"All prompts should be the same. {all_prompts=}")
        #     print("PASSED PROMPT CONSISTENCY CHECK")

        # # Add synchronization point to prevent processes from getting out of sync
        # self.accelerator.wait_for_everyone()


        # Calculate current weights based on schedule/config
        current_step_weights = []
        current_global_step = self.state.global_step
        for weight_config in self.reward_weights_config:
            if callable(weight_config):
                # Call the schedule function with the current step
                current_weight = weight_config(current_global_step)
                if not 0.0 <= current_weight <= 1.0:
                     warnings.warn(f"Reward weight schedule returned {current_weight} at step {current_global_step}. Clamping to [0, 1].")
                     current_weight = max(0.0, min(1.0, current_weight))
                current_step_weights.append(current_weight)
            else:
                # Use the fixed float weight
                current_step_weights.append(weight_config)

        current_step_weights_tensor = torch.tensor(current_step_weights, dtype=torch.float32, device=device)

        # Log the calculated weights for this step
        for i, weight in enumerate(current_step_weights):
            reward_func = self.reward_funcs[i]
            reward_func_name = reward_func.__name__
            self._metrics[f"reward_weights/{reward_func_name}"].append(weight)

        # Apply calculated weights to each reward function's output and sum
        rewards = (rewards_per_func * current_step_weights_tensor.to(device).unsqueeze(0)).sum(dim=1)
        print(f"Finished with reward weighting, {rewards=} {self.accelerator.process_index=}")
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        print(f"Finished with grouped rewards, {mean_grouped_rewards=} {self.accelerator.process_index=}")
        print(f"Finished with grouped rewards std, {std_grouped_rewards=} {self.accelerator.process_index=}")

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)


        print("Finished with advantages")

        # if we are using the SSR buffer, we need to populate it with the current batch of examples
        # we DO allow an example from the buffer to be re-added after it is popped
        if self.use_ssr_buffer and self.accelerator.process_index == 0:
            # if the average absolute advantage is greater than 0, we add that example to the buffer with the average advantage
            average_abs_advantage = torch.abs(advantages).mean().item()
            if average_abs_advantage > 0:
                print(f"Adding {inputs[0]} to the SSR buffer with advantage {average_abs_advantage}")

                # add the example to the buffer with the average advantage
                self.ssr_buffer.add_example(inputs[0], average_abs_advantage)

        print("Finished with repopulating SSR buffer")

        self.accelerator.wait_for_everyone()


        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(conversations),
            (self.accelerator.process_index + 1) * len(conversations),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            inputs_data_to_log = {
                key: gather_object(
                    [i[key] for i in inputs if key in i]
                ) for key in self.inputs_to_log
            }
            # if the value is torch.Tensor, convert it to a list
            for key, value in inputs_data_to_log.items():
                if isinstance(value, torch.Tensor):
                    inputs_data_to_log[key] = value.tolist()

            # gather completion_ids and get num_image_pad_ids
            # completion_ids shape: (B*G, C) B is batch size, G is number of generations, C is completion length
            gathered_completion_ids = gather_object(completion_ids)
            # after gathering, there will be B*G items and each item is a tensor of shape their own(C,)
            # handle each item one by one
            num_image_pad_ids = [(ids == self.image_pad_id).sum().item() for ids in gathered_completion_ids]
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
                "reward_per_func": rewards_per_func.tolist(),
                "num_image_pad_ids": num_image_pad_ids,
                **inputs_data_to_log,
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        self._metrics["buffer_size"].append(self.ssr_buffer.buffer_size if self.use_ssr_buffer else 0)
        self._metrics["buffer_usage"].append(float(should_use_buffer))

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        pixel_values, image_grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = torch.min(per_token_loss1, per_token_loss2)

        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

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
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

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
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
