# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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


import math
import queue
import textwrap
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin

from trl.trainer.base_trainer import _BaseTrainer
from trl.trainer.utils import pad, patch_chunked_lm_head

from .async_grpo_config import AsyncGRPOConfig
from .async_rollout_worker import AsyncRolloutWorker


logger = get_logger(__name__)

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional arguments from the trainer (refer to the trainer's source for details). To ensure forward
# compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]


class RolloutWorkerProtocol(Protocol):
    rollout_buffer: queue.Queue

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def send_weights(self, iterator: Iterator[tuple[str, torch.Tensor]]) -> None: ...
    def update_model_version(self, version: int) -> None: ...


class StepIntervalCallback(TrainerCallback):
    """
    A callback that calls a function every N optimization steps.
    """

    def __init__(self, fn, every_n_steps: int):
        self.fn = fn
        self.every_n_steps = every_n_steps

    def on_step_end(self, _args, state, _control, **_kwargs):
        if state.global_step % self.every_n_steps == 0:
            self.fn()


class RolloutQueueDataset(torch.utils.data.IterableDataset):
    def __init__(self, rollout_queue, model_version_fn, max_staleness=3, timeout=120.0):
        self.queue = rollout_queue
        self.model_version_fn = model_version_fn
        self.max_staleness = max_staleness
        self.timeout = timeout

    def __iter__(self):
        while True:
            t0 = time.time()
            qsize = self.queue.qsize()
            if qsize == 0:
                logger.info("queue empty, waiting for rollout samples...")
            try:
                sample = self.queue.get(timeout=self.timeout)
            except queue.Empty:
                logger.warning(f"Rollout queue empty for {self.timeout}s, stopping epoch")
                return  # StopIteration ends epoch
            queue_wait_time_s = time.time() - t0
            if queue_wait_time_s > 1.0:
                logger.info(f"waited {queue_wait_time_s:.1f}s for sample (qsize={self.queue.qsize()})")

            staleness = self.model_version_fn() - sample.model_version
            if staleness > self.max_staleness:
                logger.info(f"dropping stale sample (staleness={staleness}, max={self.max_staleness})")
                continue  # drop stale, pull next

            yield {
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "advantage": sample.advantage,
                "metrics": {**sample.metrics, "queue_wait_time_s": queue_wait_time_s},
            }


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        attention_mask = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]
        completion_mask = [torch.tensor(example["completion_mask"], dtype=torch.float32) for example in examples]
        old_log_probs = [torch.tensor(example["old_log_probs"], dtype=torch.float32) for example in examples]
        advantages = torch.tensor([example["advantage"] for example in examples], dtype=torch.float32)

        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad(attention_mask, padding_value=0)
        completion_mask = pad(completion_mask, padding_value=0)
        old_log_probs = pad(old_log_probs, padding_value=0)

        # Total valid completion tokens across all samples in the full batch.
        # Repeated per sample so that DataLoaderDispatcher (dispatch_batches=True) slices correctly on dim=0
        global_n_tokens = completion_mask.sum()
        global_n_tokens_repeated = torch.full((len(examples),), global_n_tokens.item(), dtype=torch.float32)

        # Convert per-sample metrics dicts to a dict of 1D tensors so that Accelerate's
        # recursive broadcast (dispatch_batches=True) can handle them — it traverses nested
        # dicts of tensors but chokes on plain Python floats.
        metrics_list = [example["metrics"] for example in examples]
        metrics = (
            {
                key: torch.tensor([m.get(key, 0.0) for m in metrics_list], dtype=torch.float32)
                for key in metrics_list[0]
            }
            if metrics_list and metrics_list[0]
            else {}
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "global_n_tokens": global_n_tokens_repeated,
            "metrics": metrics,
        }


class AsyncGRPOTrainer(_BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
    Models](https://huggingface.co/papers/2402.03300). This trainer is the asynchronous version of GRPO, where
    generation is offloaded to an external vLLM server that runs asynchronously alongside training, decoupling rollout
    from the gradient update loop.

    Example:

    ```python
    from trl.experimental.async_grpo import AsyncGRPOTrainer
    from trl.rewards import accuracy_reward
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str`):
            Model to be trained. Must be a string, being the *model id* of a pretrained model hosted inside a model
            repo on huggingface.co, or a path to a *directory* containing model weights saved using
            [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
            using [`~transformers.AutoModelForCausalLM.from_pretrained`]. The model name is also used to identify the
            model on the vLLM server used for generation.
        reward_funcs (`RewardFunc | list[RewardFunc]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function: The function is provided with the prompts and the generated completions, plus
              any additional columns in the dataset. It should return a list of rewards. Reward functions can be either
              synchronous or asynchronous and can also return `None` when the reward is not applicable to those
              samples. This is useful for multi-task training where different reward functions apply to different types
              of samples. When a reward function returns `None` for a sample, that reward function is excluded from the
              reward calculation for that sample. For more details, see [Using a custom reward
              function](#using-a-custom-reward-function).
            - A list of reward functions, where each item is a reward function as described above. Rewards from all
              functions are summed.
        args ([`AsyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset are
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processing class used to process the data. The padding side must be set to `"left"`. If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        tools (list of `Callable`, *optional*):
            A list of callable tool functions (sync or async) that the model can invoke during generation. Each tool
            should be a standard Python function with properly type-hinted arguments and return values, and a
            Google-style docstring describing its purpose, arguments, and return value. For more details, see:
            https://huggingface.co/docs/transformers/en/chat_extras#passing-tools. The model uses the function's name,
            type hints, and docstring to determine how to call it. Ensure that the model's chat template supports tool
            use and that it has been fine-tuned for tool calling.
        environment_factory (`EnvironmentFactory`, *optional*):
            A callable that creates and returns an environment instance. The environment class should define methods
            that can be invoked as tools during generation. Each method should comply with the same requirements as the
            `tools` described above. If `environment_factory` is provided, an instance of the environment is created
            for each generation in the batch, allowing for parallel and independent interactions. The environment must
            also implement a callable `reset` method that can be used to reset state between generations. The `reset`
            method should return either `None` or a string: when it returns a string, that string is appended to the
            last user message before generation. This feature is experimental and may change or be removed at any time
            without prior notice.
    """

    _tag_names = ["trl", "async-grpo"]
    _name = "AsyncGRPO"
    _paper = {
        "title": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
        "id": "2402.03300",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @article{shao2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }"""),
    }

    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: AsyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | None = None,
        rollout_worker: RolloutWorkerProtocol | None = None,
    ):
        self.args = args or AsyncGRPOConfig()

        # Training arguments
        self.epsilon_low = self.args.epsilon
        self.epsilon_high = self.args.epsilon_high
        self.temperature = self.args.temperature

        # Model
        model_name = model
        model = AutoModelForCausalLM.from_pretrained(model, device_map=None, dtype=torch.float32)

        if self.args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        patch_chunked_lm_head(model, chunk_size=8192, temperature=self.temperature)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_name)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        # Initialize the Trainer
        super().__init__(
            model=model,
            args=self.args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_loss_func="non-None value to disable scaling",
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Infer max_steps from dataset size when not explicitly set. This must happen after super().__init__()
        # so that self.accelerator.num_processes is available for the correct calculation.
        samples_per_step = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.accelerator.num_processes
        )
        if self.args.max_steps <= 0 and train_dataset is not None and hasattr(train_dataset, "__len__"):
            samples_per_epoch = len(train_dataset) * self.args.num_generations
            self.args.max_steps = int(self.args.num_train_epochs * samples_per_epoch / samples_per_step)

        # Infer max_inflight_tasks when not explicitly set. Generating more samples than the trainer can consume
        # before they become stale is wasteful. The useful upper bound is max_staleness * samples_per_step.
        if self.args.max_inflight_tasks < 0:
            self.args.max_inflight_tasks = self.args.max_staleness * samples_per_step
            logger.info(
                f"max_inflight_tasks set to {self.args.max_inflight_tasks} "
                f"(max_staleness={self.args.max_staleness} × samples_per_step={samples_per_step})"
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._train_tokens_start_time = None
        self.model_version = 0
        # Create worker and queue on rank 0
        if self.accelerator.is_main_process:
            if self.train_dataset is None:
                raise ValueError("train_dataset is required for AsyncGRPOTrainer")

            if rollout_worker is not None:
                # Use the injected worker (e.g. a stub in tests). The queue is owned by the worker.
                self.rollout_worker = rollout_worker
            else:
                # Collect weight metadata once — names/dtypes/shapes are fixed for the lifetime of training.
                # DTensor.shape returns the global shape without triggering any all-gather.
                weight_names, weight_dtype_names, weight_shapes = [], [], []
                for name, param in model.named_parameters():
                    # DDP/FSDP1 wrapping, avoids vllm module not exist error
                    name = name.removeprefix("module.")
                    weight_names.append(name)
                    weight_dtype_names.append(str(param.dtype).split(".")[-1])
                    weight_shapes.append(list(param.shape))
                self.rollout_worker = AsyncRolloutWorker(
                    model_name=model_name,
                    dataset=train_dataset,
                    reward_funcs=reward_funcs,
                    tools=tools,
                    environment_factory=environment_factory,
                    num_generations=self.args.num_generations,
                    max_inflight_tasks=self.args.max_inflight_tasks,
                    queue_maxsize=self.args.queue_maxsize,
                    vllm_server_url=self.args.vllm_server_base_url,
                    max_tokens=self.args.max_completion_length,
                    temperature=self.args.temperature,
                    request_timeout=self.args.request_timeout,
                    server_timeout=self.args.vllm_server_timeout,
                    chat_template_kwargs=self.args.chat_template_kwargs,
                    max_tool_calling_iterations=self.args.max_tool_calling_iterations,
                    log_completions=self.args.log_completions,
                    num_completions_to_print=self.args.num_completions_to_print,
                    weight_names=weight_names,
                    weight_dtype_names=weight_dtype_names,
                    weight_shapes=weight_shapes,
                )
            self.rollout_queue = self.rollout_worker.rollout_buffer
        else:
            self.rollout_queue = None
            self.rollout_worker = None

        # Add callbacks
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))

    def get_train_dataloader(self) -> DataLoader:
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                max_staleness=self.args.max_staleness,
                timeout=self.args.vllm_server_timeout,
            )
        else:
            dataset = _EmptyIterableDataset()

        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=self.args.per_device_train_batch_size * self.accelerator.num_processes,
                collate_fn=DataCollatorForRollout(self.processing_class.pad_token_id),
                num_workers=0,
                # NOTE(@aminediro):
                # dispatch_batches = True for DataLoader whose underlying dataset is an IterableDataset
                # dataloader prepared by the Accelerator is only iterated through on the main process a
            )
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In AsyncGRPOTrainer, we need additional columns ("completion_mask", "old_log_probs",
        # "advantages", "global_n_tokens") to compute the loss, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "completion_mask",
                "old_log_probs",
                "advantages",
                "global_n_tokens",
                "metrics",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        old_log_probs = inputs["old_log_probs"]
        advantages = inputs["advantages"]

        # The collator pads to the global batch max length (across all ranks). After DataLoaderDispatcher slices and
        # sends rows to each rank, the local slice is still padded to that global max. Truncate to the longest real
        # sequence in this rank's slice so we don't run the forward pass over pure-padding columns.
        local_max_len = attention_mask.sum(dim=1).max()
        input_ids = input_ids[:, :local_max_len]
        attention_mask = attention_mask[:, :local_max_len]
        completion_mask = completion_mask[:, :local_max_len]
        old_log_probs = old_log_probs[:, :local_max_len]

        forward_start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )
        log_probs, entropy = outputs["log_probs"], outputs["entropy"]
        self._last_forward_time_s = time.time() - forward_start

        completion_mask = completion_mask[:, 1:]
        old_log_probs = old_log_probs[:, 1:]
        advantages = advantages.unsqueeze(1)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(ratio * advantages, clipped * advantages)

        # DDP/FSDP averages gradients across ranks (world_size).
        # To get correct per-token normalization we scale by 1/tokens_per_rank
        # = world_size / global_n_tokens, so after DDP averaging the effective
        loss = (per_token_loss * completion_mask).sum()
        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        loss = loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            valid_mask = completion_mask > 0
            local_count = valid_mask.sum().float()

            local_ratio_sum = (
                ratio[valid_mask].sum() if valid_mask.any() else torch.zeros((), device=completion_mask.device)
            )
            # Approx KL: http://joschu.net/blog/kl-approx.html
            local_kl_sum = (
                ((ratio[valid_mask] - 1) - log_ratio[valid_mask]).sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )

            local_entropy_sum = (
                entropy[valid_mask].sum() if valid_mask.any() else torch.zeros((), device=completion_mask.device)
            )

            clipped = (ratio < 1 - self.epsilon_low) | (ratio > 1 + self.epsilon_high)
            local_clip_sum = (
                clipped[valid_mask].float().sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )

            # Batch all-reduce: [ratio_sum, kl_sum, entropy_sum, clip_sum, count]
            stats = torch.stack([local_ratio_sum, local_kl_sum, local_entropy_sum, local_clip_sum, local_count])
            stats = self.accelerator.reduce(stats, reduction="sum")
            global_ratio_sum, global_kl_sum, global_entropy_sum, global_clip_sum, global_count = stats.unbind(0)
            self._metrics["train"]["ratio"].append((global_ratio_sum / global_count).item())
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["clip_ratio"].append((global_clip_sum / global_count).item())

            # Logging metrics from the rollout worker (reward, reward_std, etc.).
            # inputs["metrics"] is a dict of 1D tensors keyed by metric name.
            sample_metrics = inputs["metrics"]  # dict[str, Tensor(shape=[B_local])]
            keys = list(sample_metrics.keys())
            device = completion_mask.device
            n_samples = torch.tensor(completion_mask.shape[0], dtype=torch.float32, device=device)
            if keys:
                local_sums = torch.stack([sample_metrics[k].to(device).sum() for k in keys])
                stats = torch.cat([local_sums, n_samples.unsqueeze(0)])
                stats = self.accelerator.reduce(stats, reduction="sum")
                global_sums, global_n_samples = stats[:-1], stats[-1]
                for k, global_sum in zip(keys, global_sums, strict=True):
                    self._metrics["train"][k].append((global_sum / global_n_samples).item())

            completion_length = completion_mask.sum(dim=1).float()
            length_stats = torch.stack([completion_length.sum(), n_samples])
            length_stats = self.accelerator.reduce(length_stats, reduction="sum")
            self._metrics["train"]["completions/mean_length"].append((length_stats[0] / length_stats[1]).item())

            # Training throughput: completion tokens consumed by this training step per second.
            now = time.time()
            if self._train_tokens_start_time is not None:
                train_elapsed = now - self._train_tokens_start_time
                if train_elapsed > 0:
                    self._metrics["train"]["training_tok/s"].append(global_n_tokens.item() / train_elapsed)
            self._train_tokens_start_time = now

            self._metrics["train"]["forward_time_s"].append(self._last_forward_time_s)
            # NOTE: in dynamic mbs setup, we would need to agg across DP ranks.
            self._metrics["train"]["train_seq_len"].append(float(local_max_len))
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        # Average the metrics
        metrics = {}
        for key, val in self._metrics[mode].items():
            # Filter out NaN values before averaging. A reward function that returns None for all samples
            # in a batch produces NaN for that batch's metric. With logging_steps > 1, a naive sum()/len()
            # would let a single NaN contaminate valid data from other batches. Only return None when no
            # valid values remain (e.g. JSON loggers crash on float NaN).
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _streaming_iter(self):
        # Iterate parameters one at a time. For FSDP2 (DTensor), full_tensor() all-gathers just this parameter across
        # FSDP ranks, then frees it once the generator advances — avoiding materializing the full model in memory.
        device = self.accelerator.device
        for name, param in self.model.named_parameters():
            name = name.removeprefix("module.")  # DDP/FSDP1 wrapping
            full = param.full_tensor() if isinstance(param, DTensor) else param.detach()
            if full.device != device:
                full = full.to(device)
            yield name, full

    def _sync_weight(self):
        t0 = time.time()
        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process and self.rollout_worker:
            self.rollout_worker.pause()
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring weights... (barrier took {t_barrier - t_pause:.1f}s)")
        if self.accelerator.is_main_process and self.rollout_worker:
            self.rollout_worker.send_weights(self._streaming_iter())
        else:
            # Non-rank-0 processes must still participate in full_tensor() collectives for FSDP2.
            for _ in self._streaming_iter():
                pass
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process and self.rollout_worker:
            self.rollout_worker.resume()
            self.model_version += 1
            self.rollout_worker.update_model_version(self.model_version)
        weight_sync_time_s = time.time() - t0
        self._metrics["train"]["weight_sync_time_s"].append(weight_sync_time_s)
        logger.info(f"Weight sync: done. Total {weight_sync_time_s:.1f}s")

    def _inner_training_loop(self, *args, **kwargs):
        # Start the rollout worker here (not in __init__) so that checkpoint loading in Trainer.train()
        # has already restored the model weights. The sequence is: start worker thread → wait for NCCL
        # init → sync weights to vLLM → begin generation. This ensures vLLM always uses the current
        # policy before producing any samples (matters for resumed runs, harmless for fresh ones).
        self._sync_weight()
        if self.accelerator.is_main_process and self.rollout_worker:
            self.rollout_worker.start()
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process and self.rollout_worker:
                self.rollout_worker.stop()
