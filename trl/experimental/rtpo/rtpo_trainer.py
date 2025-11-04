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

from ...extras.profiling import profiling_decorator
from typing import List, Dict, Any, Union, Optional
import torch
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc
from .rtpo_config import RTPOConfig
from ...data_utils import is_conversational
from ...trainer.utils import shuffle_sequence_dict, split_pixel_values_by_grid, split_tensor_dict, unsplit_pixel_values_by_grid


# This class should be added to trl.data_utils when the RTPO method is stable enough to join trl.trainer
import math
from typing import Any, Union


class AnnealingScheduler:
    """
    General annealing scheduler, can be used like a learning rate scheduler.
    Supports: linear, cosine, exponential, constant, piecewise
    Can be 'up' (0→1) or 'down' (1→0).
    """

    def __init__(
        self,
        total_steps: Union[int, float],
        schedule_type: str = "linear",
        direction: str = "down",  # "down" means 1→0 annealing
        **kwargs: Any,
    ) -> None:
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.direction = direction
        self.kwargs = kwargs  # extra params like decay_rate, milestones, etc.

    def __call__(self, step: Union[int, float]) -> float:
        """Return annealing factor based on current step."""
        ratio = min(step / max(self.total_steps, 1), 1.0)

        if self.schedule_type == "linear":
            value = ratio

        elif self.schedule_type == "cosine":
            # Cosine annealing: starts fast, slows down later
            value = 0.5 * (1 - math.cos(math.pi * ratio))

        elif self.schedule_type == "exponential":
            # Exponential annealing: f(t)=1 - exp(-k*t)
            k = self.kwargs.get("decay_rate", 5.0)
            value = 1 - math.exp(-k * ratio)

        elif self.schedule_type == "piecewise":
            milestones = self.kwargs.get("milestones", [0.3, 0.6, 0.9])
            values = self.kwargs.get("values", [0.2, 0.5, 0.8, 1.0])
            for i, m in enumerate(milestones):
                if ratio < m:
                    value = values[i]
                    break
            else:
                value = values[-1]

        elif self.schedule_type == "constant":
            value = self.kwargs.get("value", 1.0)

        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        # Apply direction: up (0→1) or down (1→0)
        if self.direction == "down":
            return 1.0 - value
        elif self.direction == "up":
            return value
        else:
            raise ValueError(f"Invalid direction: {self.direction}")


def think_guigence_anneal(
    generation_batch: List[Dict[str, Any]],
    anneal_factor: float,
    tokenizer_or_processor: Union[PreTrainedTokenizerBase, ProcessorMixin],
) -> List[Dict[str, Any]]:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    if is_conversational(generation_batch[0]):
        for i in range(len(generation_batch)):
            if generation_batch[i]["prompt"][-1]["role"] == "assistant":
                think = generation_batch[i]["prompt"][-1]["content"]
                tokens = tokenizer.encode(think)
                anneal = tokenizer.decode(tokens[: int(anneal_factor * len(tokens))], skip_special_tokens=True)
                generation_batch[i]["prompt"][-1]["content"] = anneal
    return generation_batch


def drop_assistant_content(generation_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if is_conversational(generation_batch[0]):
        for i in range(len(generation_batch)):
            if generation_batch[i]["prompt"][-1]["role"] == "assistant":
                generation_batch[i]["prompt"].pop()
    return generation_batch


class RTPOTrainer(GRPOTrainer):
    """
    Trainer for Reverse Thinking Policy Optimization (RTPO).
    Example:

    ```python
    from datasets import load_dataset
    from trl import RTPOTrainer, RTPOConfig

    dataset = load_dataset("your-vlm-dataset", split="train")


    def reward_func(completions, **kwargs):
        # Your reward function for multimodal reasoning
        return [compute_reward(c) for c in completions]


    config = RTPOConfig(
        loss_type="grpo",  # Use GRPO as base
        perception_loss_weight=0.1,
        mask_ratio=0.3,
    )

    trainer = RTPOTrainer(
        model="Qwen/Qwen2-VL-2B-Instruct",
        reward_funcs=reward_func,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained (must be a vision-language model).
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards (same as GRPO).
        args ([`PAPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. Must include "prompt" and "image" columns.
        eval_dataset: Same requirements as train_dataset.
        processing_class: Processing class (tokenizer/processor) for the model.
        reward_processing_classes: Processing classes for reward models.
        callbacks: Training callbacks.
        optimizers: Optimizer and scheduler tuple.
        peft_config: PEFT configuration if using parameter-efficient fine-tuning.
    """

    _tag_names = ["trl", "rtpo"]
    _name = "RTPO"
    # _paper = { # TODO paper
    #     "title": "Perception-Aware Policy Optimization for Multimodal Reasoning",
    #     "id": "2507.06448",
    #     # docstyle-ignore
    #     "citation": textwrap.dedent(
    #         """\
    #         @misc{wang2025perceptionawarepolicyoptimizationmultimodal,
    #             title        = {{Perception-Aware Policy Optimization for Multimodal Reasoning}},
    #             author       = {Zhenhailong Wang and Xuehang Guo and Sofia Stoica and Haiyang Xu and Hongru Wang and Hyeonjeong Ha and Xiusi Chen and Yangyi Chen and Ming Yan and Fei Huang and Heng Ji},
    #             year         = 2025,
    #             url          = {https://arxiv.org/abs/2507.06448},
    #             archivePrefix= {arXiv},
    #             eprint       = {2507.06448},
    #             primaryClass = {cs.CL}
    #         }"""
    #     ),
    # }

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[RTPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        # Initialize with default RTPO config if not provided
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = RTPOConfig(f"{model_name}-RTPO")

        total_steps = args.max_steps or (len(self.get_train_dataloader()) * args.num_train_epochs)
        self.anneal_scheduler = AnnealingScheduler(
            total_steps=total_steps,
            schedule_type=args.schedule_type,  # linear / cosine / exponential / piecewise / constant
            direction=args.direction,  # up 0 -> 1 / down 1-> 0
            decay_rate=args.decay_rate,  # corresponding to the exponential parameter
            milestones=args.milestones,  # corresponding to the piecewise parameter
            values=args.values,  # corresponding to the piecewise parameter
            value=args.value,  # corresponding to the constant parameter
        )

        # Initialize parent GRPO trainer
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generation_batch = think_guigence_anneal(generation_batch, self.anneal_scheduler(self.state.global_step), self.processing_class)
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            generation_batch = drop_assistant_content(generation_batch)
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)  # no thinking guigence on eval.
        return inputs
