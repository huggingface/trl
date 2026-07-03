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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import requests
import torch
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin

from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import nanmax, nanmin, pad, patch_chunked_lm_head
from .async_grpo_config import AsyncGRPOConfig
from .async_rollout_worker import AsyncRolloutWorker
from .weight_transfer import WeightTransferClient


logger = get_logger(__name__)

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional arguments from the trainer (refer to the trainer's source for details). To ensure forward
# compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]


class RolloutWorkerProtocol(Protocol):
    """Interface a rollout worker must implement to be passed as `rollout_worker` to [`AsyncGRPOTrainer`].

    The default [`AsyncRolloutWorker`] spawns a CUDA-free child process and scores completions with the trainer's
    `reward_funcs`. Implement this protocol to plug in a custom rollout/scoring backend instead — for example, one that
    runs reward models on their own GPUs.

    Attributes:
        rollout_buffer (`queue.Queue`):
            Queue the trainer drains; the worker pushes scored `RolloutSample`s onto it.
    """

    rollout_buffer: queue.Queue

    def start(self) -> None:
        """Begin producing rollouts. Called once on train begin, after the initial weight sync."""
        ...

    def stop(self) -> None:
        """Stop the worker and release its resources. Called on train end."""
        ...

    def update_model_version(self, version: int) -> None:
        """Tell the worker which policy version is now live, so it can tag or discard stale samples."""
        ...

    def check_health(self, stale_after_s: float) -> None:
        """Raise if the worker has crashed or stopped producing within `stale_after_s` seconds."""
        ...


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


class _InitialWeightSyncCallback(TrainerCallback):
    """Idempotent: NCCL group setup + cold weight sync to vLLM on train begin."""

    def __init__(self, trainer: "AsyncGRPOTrainer"):
        self._trainer = trainer
        self._fired = False

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        if self._fired:
            return
        self._fired = True
        if self._trainer.accelerator.is_main_process and self._trainer.weight_transfer is not None:
            self._trainer.weight_transfer.init_weight_transfer()
        self._trainer._sync_weight()


class _StartRolloutWorkerCallback(TrainerCallback):
    """Idempotent: starts the rollout worker. Must be registered AFTER `_InitialWeightSyncCallback`."""

    def __init__(self, trainer: "AsyncGRPOTrainer"):
        self._trainer = trainer
        self._fired = False

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        if self._fired:
            return
        self._fired = True
        if self._trainer.accelerator.is_main_process and self._trainer.rollout_worker is not None:
            self._trainer.rollout_worker.start()


class RolloutQueueDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        rollout_queue,
        model_version_fn,
        check_health_fn,
        stale_after_s,
        max_staleness=3,
        poll_interval_s=5.0,
    ):
        self.queue = rollout_queue
        self.model_version_fn = model_version_fn
        self.check_health_fn = check_health_fn
        self.stale_after_s = stale_after_s
        self.max_staleness = max_staleness
        self.poll_interval_s = poll_interval_s

    def __iter__(self):
        while True:
            t0 = time.time()
            if self.queue.qsize() == 0:
                logger.info("queue empty, waiting for rollout samples...")
            try:
                sample = self.queue.get(timeout=self.poll_interval_s)
            except queue.Empty:
                # Returning here would broadcast None through accelerate's dispatch loop.
                self.check_health_fn(self.stale_after_s)
                continue
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


def _get_vllm_max_model_len(server_url: str, timeout: float) -> int:
    """Query the vLLM server for the served model's `max_model_len` (the cap on prompt + completion tokens)."""
    response = requests.get(f"{server_url.rstrip('/')}/v1/models", timeout=timeout)
    response.raise_for_status()
    return response.json()["data"][0]["max_model_len"]


def _balance_by_squared_length(examples: list[dict[str, Any]], num_groups: int) -> list[list[dict[str, Any]]]:
    """Greedily partition `examples` into `num_groups` rows (one per DP rank), balancing each row's Σ Lᵢ².

    Attention is O(L²) while the FFN is O(L), so equal token counts wouldn't equalize wall-time; balancing Σ Lᵢ² keeps
    the per-micro-batch all-reduce free of stragglers. Samples are placed longest-first into the row with the smallest
    running Σ Lᵢ² (LPT scheduling). With at least `num_groups` samples every row ends up non-empty.
    """
    groups = [[] for _ in range(num_groups)]
    squared_loads = [0] * num_groups
    for example in sorted(examples, key=lambda e: len(e["input_ids"]), reverse=True):
        n = len(example["input_ids"])
        i = min(range(num_groups), key=lambda j: squared_loads[j])
        groups[i].append(example)
        squared_loads[i] += n * n
    return groups


class FixedCountBatcher(torch.utils.data.IterableDataset):
    """Fixed-count batcher (the planner) wrapping [`RolloutQueueDataset`].

    Buffers `microbatch_size` (= `per_device_train_batch_size × num_processes`) samples, then partitions them across
    the `num_processes` rows (one per DP rank) balanced by Σ Lᵢ² (attention cost) so no rank straggles at the
    per-micro-batch all-reduce. The sample count is fixed, so this does not bound peak memory — use
    [`TokenBudgetBatcher`] for that. With `microbatch_size >= num_processes` every row is non-empty.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        microbatch_size (`int`):
            Number of samples buffered into each micro-batch before it is partitioned and emitted.
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, microbatch_size: int):
        self.dataset = dataset
        self.num_processes = num_processes
        self.microbatch_size = microbatch_size

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.microbatch_size:
                yield _balance_by_squared_length(batch, self.num_processes)
                batch = []


class TokenBudgetBatcher(torch.utils.data.IterableDataset):
    """Token-budgeted dynamic batcher (the planner) wrapping [`RolloutQueueDataset`].

    Keeps `num_processes` open rows (one per DP rank) and pulls single samples from the source one at a time, dropping
    each into the row with the smallest running Σ Lᵢ² (attention cost) that still fits within `token_budget` tokens.
    When the next sample fits in no row, the current micro-batch is emitted — a list of `num_processes` groups, already
    partitioned per rank — and a fresh one is started with that sample. The number of samples per row is therefore
    dynamic: short samples pack many per row, long ones pack few, while every row stays within `token_budget` tokens.
    This bounds peak memory independently of `per_device_train_batch_size` and keeps the rows Σ Lᵢ²-balanced so no rank
    straggles at the per-micro-batch all-reduce.

    Every emitted micro-batch has all `num_processes` rows non-empty (a rank forwarding zero tokens would desync
    FSDP/EP collectives): a micro-batch is only closed once every row holds at least one sample. A sample longer than
    `token_budget` fits in no row, so it is dropped with a warning; set `token_budget` ≥ the vLLM server's
    `max_model_len` (the cap on prompt + completion) to avoid dropping samples.

    Args:
        dataset ([`RolloutQueueDataset`]):
            Source yielding single rollout-sample dicts.
        num_processes (`int`):
            Number of DP ranks; the number of rows (one per rank) in each micro-batch.
        token_budget (`int`):
            Maximum real tokens packed into a single row (one rank's forward).
    """

    def __init__(self, dataset: "RolloutQueueDataset", num_processes: int, token_budget: int):
        self.dataset = dataset
        self.num_processes = num_processes
        self.token_budget = token_budget

    def __iter__(self):
        rows = [[] for _ in range(self.num_processes)]
        squared_loads = [0] * self.num_processes  # Σ Lᵢ² per row, drives the balancing
        token_counts = [0] * self.num_processes  # tokens per row, drives the budget
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                # Longer than the whole budget: fits in no row, so drop it (placing it would overshoot the budget
                # or force an empty row that desyncs FSDP/EP collectives).
                logger.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds token_budget={self.token_budget}. "
                    "Raise token_budget to avoid dropping samples."
                )
                continue
            fits = [i for i in range(self.num_processes) if token_counts[i] + n <= self.token_budget]
            if not fits:
                # No row has room (all are non-empty, since this sample fits an empty one): close and reset.
                yield rows
                rows = [[] for _ in range(self.num_processes)]
                squared_loads = [0] * self.num_processes
                token_counts = [0] * self.num_processes
                fits = list(range(self.num_processes))
            i = min(fits, key=lambda j: squared_loads[j])
            rows[i].append(sample)
            squared_loads[i] += n * n
            token_counts[i] += n


class _EmptyIterableDataset(torch.utils.data.IterableDataset):
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    """
    Padding-free collator (the packer) for rollout samples. Packs a micro-batch into `num_processes` rows (one per DP
    rank): each row concatenates its samples into a single sequence, with `position_ids` resetting per sequence and
    advantages expanded per-token. Rows are padded only to the longest row, so the batch stays rectangular for
    `DataLoaderDispatcher` to scatter row `i` -> rank `i`; this inter-rank padding is stripped per-rank in
    `compute_loss`.

    The micro-batch arrives already partitioned into `num_processes` rows by the upstream planner
    ([`FixedCountBatcher`] or [`TokenBudgetBatcher`]) — which balances each row's Σ Lᵢ² (attention cost) to avoid
    stragglers at the gradient all-reduce — so the collator only tensorizes the given rows.

    Args:
        pad_token_id (`int`):
            Token id used to pad `input_ids`.
        num_processes (`int`, *optional*, defaults to `1`):
            Number of DP ranks; the micro-batch is packed into this many rows.
    """

    pad_token_id: int
    num_processes: int = 1
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        # The dataloader uses batch_size=1 over a planner that pre-partitions each micro-batch into `num_processes`
        # rows, so `examples` is a length-1 list holding that single micro-batch (one group per rank).
        (groups,) = examples

        input_ids, attention_mask, completion_mask, old_log_probs, position_ids, advantages = [], [], [], [], [], []
        for group in groups:
            seq_lengths = [len(example["input_ids"]) for example in group]
            ids = [token for example in group for token in example["input_ids"]]
            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_mask.append(torch.ones(len(ids), dtype=torch.long))
            completion_mask.append(
                torch.tensor([m for example in group for m in example["completion_mask"]], dtype=torch.long)
            )
            old_log_probs.append(
                torch.tensor([lp for example in group for lp in example["old_log_probs"]], dtype=torch.float32)
            )
            position_ids.append(torch.cat([torch.arange(n) for n in seq_lengths]))
            advantages.append(
                torch.cat(
                    [torch.full((n,), example["advantage"]) for example, n in zip(group, seq_lengths, strict=False)]
                )
            )

        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad(attention_mask, padding_value=0)
        completion_mask = pad(completion_mask, padding_value=0)
        old_log_probs = pad(old_log_probs, padding_value=0.0)
        position_ids = pad(position_ids, padding_value=0)
        advantages = pad(advantages, padding_value=0.0)

        all_examples = [example for group in groups for example in group]

        # Total valid completion tokens across all samples in the full batch.
        # Repeated per rank so that DataLoaderDispatcher (dispatch_batches=True) slices correctly on dim=0
        global_n_tokens = sum(sum(example["completion_mask"]) for example in all_examples)
        global_n_tokens = torch.full((self.num_processes,), float(global_n_tokens), dtype=torch.float32)

        # Per-sample metrics grouped per rank, as a dict of 2D tensors (one row per rank) so that Accelerate's
        # recursive broadcast (dispatch_batches=True) can scatter them — it traverses nested dicts of tensors but
        # chokes on plain Python floats. Rows are padded with NaN so padded slots are ignored by the nan-aware
        # aggregation in `compute_loss`.
        metrics = (
            {
                key: pad(
                    [
                        torch.tensor([example["metrics"].get(key, 0.0) for example in group], dtype=torch.float32)
                        for group in groups
                    ],
                    padding_value=float("nan"),
                )
                for key in all_examples[0]["metrics"]
            }
            if all_examples[0]["metrics"]
            else {}
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_log_probs": old_log_probs,
            "position_ids": position_ids,
            "advantages": advantages,
            "global_n_tokens": global_n_tokens,
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
    >>> from trl.experimental.async_grpo import AsyncGRPOTrainer
    >>> from trl.rewards import accuracy_reward
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = AsyncGRPOTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     reward_funcs=accuracy_reward,
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str`):
            Model to be trained. Must be a string, being the *model id* of a pretrained model hosted inside a model
            repo on huggingface.co, or a path to a *directory* containing model weights saved using
            [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
            using [`~transformers.AutoModelForCausalLM.from_pretrained`]. The model name is also used to identify the
            model on the vLLM server used for generation.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. May be omitted when the reward is supplied
            by the environment through `environment_factory` (see below). Can be either:

            - A single reward function: The function is provided with the prompts and the generated completions, plus
              any additional columns in the dataset. It should return a list of rewards. Reward functions can be either
              synchronous or asynchronous and can also return `None` when the reward is not applicable to those
              samples. This is useful for multi-task training where different reward functions apply to different types
              of samples. When a reward function returns `None` for a sample, that reward function is excluded from the
              reward calculation for that sample. For more details, see [Using a custom reward
              function](#using-a-custom-reward-function).
            - A list of reward functions, where each item is a reward function as described above. Rewards from all
              functions are summed.

            Unlike [`GRPOTrainer`], rewards are computed in a spawned child process, so each reward function (along
            with `tools` and `environment_factory`) must be picklable: use a module-level function,
            `functools.partial`, or a callable class instance — lambdas and closures will fail at startup. The child
            process also runs with `CUDA_VISIBLE_DEVICES=""`, so a GPU-backed reward model runs on CPU (slow), not the
            trainer's GPU.
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
            last user message before generation. The environment may also define a `get_reward` method taking no
            argument and returning a `float`: when present, the environment owns the reward, and `get_reward` is called
            once per completed rollout to score it from the environment's internal state. It acts as an additional
            reward source (with weight 1, logged under the environment's class name) alongside `reward_funcs`, which
            then becomes optional. This feature is experimental and may change or be removed at any time without prior
            notice.
        rollout_worker (`RolloutWorkerProtocol`, *optional*):
            Custom rollout worker implementing [`RolloutWorkerProtocol`]. If `None`, a default [`AsyncRolloutWorker`]
            is created, which spawns a CUDA-free child process and scores completions with the trainer's
            `reward_funcs`. Pass a custom worker to plug in a different rollout/scoring backend instead — for example,
            one that runs reward models on their own GPUs.
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
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
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
        self.epsilon_high = self.args.epsilon_high if self.args.epsilon_high is not None else self.args.epsilon
        self.temperature = self.args.temperature

        # Model
        model_name = model
        model_init_kwargs = self.args.model_init_kwargs or {}
        model_init_kwargs.setdefault("trust_remote_code", self.args.trust_remote_code)
        # FlashAttention is required: training runs in padding-free mode, where sequences are concatenated into a
        # single row and `cu_seq_lens` are derived from `position_ids` resets. SDPA/eager can't handle this.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,
            dtype=torch.float32,
            attn_implementation="kernels-community/flash-attn3",
            **model_init_kwargs,
        )

        if self.args.use_liger_kernel:
            raise NotImplementedError("`use_liger_kernel` is not supported yet.")

        # MoE load-balancing auxiliary loss, applied to Mixture-of-Experts models (no effect otherwise)
        text_config = model.config.get_text_config()
        is_moe = getattr(text_config, "output_router_logits", None) is not None
        self.aux_loss_enabled = is_moe and self.args.router_aux_loss_coef != 0.0
        self.router_aux_loss_coef = self.args.router_aux_loss_coef

        patch_chunked_lm_head(
            model, chunk_size=8192, temperature=self.temperature, output_router_logits=self.aux_loss_enabled
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.args.trust_remote_code)
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
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
                # Weight transfer is also expected to be wired by the test fixture (or left as None
                # if the stub doesn't sync to a real vLLM).
                self.rollout_worker = rollout_worker
                self.weight_transfer = None
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
                self.weight_transfer = WeightTransferClient(
                    vllm_server_url=self.args.vllm_server_base_url,
                    server_timeout=self.args.vllm_server_timeout,
                    weight_update_info={
                        "names": weight_names,
                        "dtype_names": weight_dtype_names,
                        "shapes": weight_shapes,
                        "packed": True,
                    },
                )
                self.rollout_worker = AsyncRolloutWorker(
                    model_name=model_name,
                    dataset=train_dataset,
                    reward_funcs=reward_funcs,
                    processing_class=processing_class,
                    tools=tools,
                    environment_factory=environment_factory,
                    num_generations=self.args.num_generations,
                    max_inflight_tasks=self.args.max_inflight_tasks,
                    queue_maxsize=self.args.queue_maxsize,
                    vllm_server_url=self.args.vllm_server_base_url,
                    max_tokens=self.args.max_completion_length,
                    temperature=self.args.temperature,
                    request_timeout=self.args.request_timeout,
                    chat_template_kwargs=self.args.chat_template_kwargs,
                    max_tool_calling_iterations=self.args.max_tool_calling_iterations,
                    log_completions=self.args.log_completions,
                    num_completions_to_print=self.args.num_completions_to_print,
                )
            # TODO(@aminediro): decide if this is returned by the worker or common API that is passed to the worker later.
            self.rollout_queue = self.rollout_worker.rollout_buffer
        else:
            self.rollout_queue = None
            self.rollout_worker = None
            self.weight_transfer = None

        # Add callbacks. Registration order matters: weight sync first, then worker start.
        self.add_callback(_InitialWeightSyncCallback(self))
        self.add_callback(_StartRolloutWorkerCallback(self))
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))

    def get_train_dataloader(self) -> DataLoader:
        num_processes = self.accelerator.num_processes
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                check_health_fn=self.rollout_worker.check_health,
                stale_after_s=self.args.heartbeat_stale_after_s,
                max_staleness=self.args.max_staleness,
            )
            # Default the token budget to the vLLM server's max_model_len (the cap on prompt + completion), so no
            # rollout sample can exceed it. Only the built-in worker manages a vLLM server (weight_transfer is set);
            # with a custom rollout_worker there may be none to query, so require an explicit budget instead.
            if self.args.token_budget is None:
                if self.weight_transfer is None:
                    raise ValueError(
                        "Set `token_budget` explicitly when passing a custom `rollout_worker`: the default is the "
                        "vLLM server's max_model_len, which is only queried for the built-in rollout worker."
                    )
                self.args.token_budget = _get_vllm_max_model_len(
                    self.args.vllm_server_base_url, self.args.vllm_server_timeout
                )
                logger.info(f"token_budget unset; defaulting to vLLM max_model_len={self.args.token_budget}")
            # The planner partitions the rollout stream into Σ Lᵢ²-balanced micro-batches of `num_processes` rows.
            # TokenBudgetBatcher caps each row at `token_budget` tokens (dynamic count, bounds peak memory);
            # FixedCountBatcher uses a fixed `per_device_train_batch_size × num_processes` samples per micro-batch.
            if self.args.token_budget > 0:
                dataset = TokenBudgetBatcher(dataset, num_processes, self.args.token_budget)
            else:
                dataset = FixedCountBatcher(
                    dataset, num_processes, self.args.per_device_train_batch_size * num_processes
                )
        else:
            dataset = _EmptyIterableDataset()

        # Each planner item is one complete micro-batch (`num_processes` pre-packed rows), so the dataloader pulls them
        # one at a time (batch_size=1) and the collator tensorizes each into a rectangular `(num_processes, T_max)`
        # batch that DataLoaderDispatcher scatters row `i` -> rank `i`.
        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=DataCollatorForRollout(self.processing_class.pad_token_id, num_processes),
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
                "position_ids",
                "advantages",
                "global_n_tokens",
                "metrics",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Padding-free: the collator already packed this rank's samples into a single row (real tokens concatenated,
        # `position_ids` resetting per sequence, advantages expanded per-token), then padded the row to the longest
        # rank's length so DataLoaderDispatcher could scatter rectangular rows. Strip that trailing inter-rank padding
        # here.
        mask_bool = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"][mask_bool].unsqueeze(0)
        completion_mask = inputs["completion_mask"][mask_bool].unsqueeze(0)
        old_log_probs = inputs["old_log_probs"][mask_bool].unsqueeze(0)
        position_ids = inputs["position_ids"][mask_bool].unsqueeze(0)
        advantages = inputs["advantages"][mask_bool].unsqueeze(0)

        forward_start = time.time()
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )
        log_probs, entropy = outputs["log_probs"], outputs["entropy"]
        self._last_forward_time_s = time.time() - forward_start

        completion_mask = completion_mask[:, 1:]
        old_log_probs = old_log_probs[:, 1:]
        advantages = advantages[:, 1:]
        log_ratio = log_probs - old_log_probs
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

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

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            aux_loss = outputs["aux_loss"]
            loss = loss + self.router_aux_loss_coef * aux_loss / self.current_gradient_accumulation_steps

        with torch.no_grad():
            valid_mask = completion_mask > 0
            local_count = valid_mask.sum().float()

            local_ratio_sum = (
                coef_1[valid_mask].sum() if valid_mask.any() else torch.zeros((), device=completion_mask.device)
            )
            # Approx KL: http://joschu.net/blog/kl-approx.html
            local_kl_sum = (
                ((coef_1[valid_mask] - 1) - log_ratio[valid_mask]).sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )

            local_entropy_sum = (
                entropy[valid_mask].sum() if valid_mask.any() else torch.zeros((), device=completion_mask.device)
            )

            # Compute the clipped probability ratios. A token is counted as clipped only when clipping is binding in a
            # policy-relevant direction: low clip when the advantage is negative, high clip when it is positive.
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            local_low_clip_sum = (
                is_low_clipped[valid_mask].float().sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )
            local_high_clip_sum = (
                is_high_clipped[valid_mask].float().sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )
            local_region_clip_sum = (
                is_region_clipped[valid_mask].float().sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )

            # Per-rank clip fractions, gathered below to report the cross-rank saturation extrema.
            local_low_clip_mean = local_low_clip_sum / local_count.clamp(min=1.0)
            local_high_clip_mean = local_high_clip_sum / local_count.clamp(min=1.0)

            # Batch all-reduce: [ratio_sum, kl_sum, entropy_sum, low_clip_sum, high_clip_sum, region_clip_sum, count]
            stats = torch.stack(
                [
                    local_ratio_sum,
                    local_kl_sum,
                    local_entropy_sum,
                    local_low_clip_sum,
                    local_high_clip_sum,
                    local_region_clip_sum,
                    local_count,
                ]
            )
            stats = self.accelerator.reduce(stats, reduction="sum")
            (
                global_ratio_sum,
                global_kl_sum,
                global_entropy_sum,
                global_low_clip_sum,
                global_high_clip_sum,
                global_region_clip_sum,
                global_count,
            ) = stats.unbind(0)
            self._metrics["train"]["ratio"].append((global_ratio_sum / global_count).item())
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["clip_ratio/low_mean"].append((global_low_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/high_mean"].append((global_high_clip_sum / global_count).item())
            self._metrics["train"]["clip_ratio/region_mean"].append((global_region_clip_sum / global_count).item())

            # Cross-rank saturation extrema, mirroring GRPOTrainer's clip_ratio/low_min and clip_ratio/high_max:
            # the smallest per-rank low-clip and largest per-rank high-clip fractions across ranks.
            gathered_low_clip = self.accelerator.gather(local_low_clip_mean)
            gathered_high_clip = self.accelerator.gather(local_high_clip_mean)
            self._metrics["train"]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            self._metrics["train"]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())

            if self.aux_loss_enabled:
                gathered_aux = self.accelerator.reduce(aux_loss.detach().to(torch.float32), reduction="sum")
                self._metrics["train"]["aux_loss"].append((gathered_aux / world_size).item())

            # Logging metrics from the rollout worker (reward, reward_std, etc.).
            # inputs["metrics"] is a dict keyed by metric name; each value is this rank's row of per-sample values,
            # NaN-padded (the nan-aware aggregation below ignores both padding and unscorable samples).
            sample_metrics = inputs["metrics"]  # dict[str, Tensor(shape=[1, n_samples_local])]
            keys = list(sample_metrics.keys())
            device = completion_mask.device
            n_samples = (position_ids == 0).sum().to(torch.float32)
            if keys:
                # nan-aware per key: unscorable samples carry NaN, so a plain .sum() would poison the whole metric.
                local_sums = torch.stack([torch.nansum(sample_metrics[k].to(device)) for k in keys])
                local_counts = torch.stack(
                    [(~torch.isnan(sample_metrics[k].to(device))).sum().to(torch.float32) for k in keys]
                )
                stats = torch.cat([local_sums, local_counts])
                stats = self.accelerator.reduce(stats, reduction="sum")
                n = len(keys)
                global_sums, global_counts = stats[:n], stats[n:]
                for k, global_sum, global_count in zip(keys, global_sums, global_counts, strict=True):
                    metric = (global_sum / global_count).item() if global_count > 0 else float("nan")
                    self._metrics["train"][k].append(metric)

            length_stats = torch.stack([completion_mask.sum().float(), n_samples])
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
            self._metrics["train"]["train_seq_len"].append(float(position_ids.max() + 1))
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

        logs.update(metrics)
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
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.pause()
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring weights... (barrier took {t_barrier - t_pause:.1f}s)")
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.send_weights(self._streaming_iter())
        else:
            # Non-rank-0 processes must still participate in full_tensor() collectives for FSDP2.
            for _ in self._streaming_iter():
                pass
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            if self.weight_transfer:
                self.weight_transfer.resume()
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        weight_sync_time_s = time.time() - t0
        self._metrics["train"]["weight_sync_time_s"].append(weight_sync_time_s)
        logger.info(f"Weight sync: done. Total {weight_sync_time_s:.1f}s")

    def _inner_training_loop(self, *args, **kwargs):
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process:
                if self.rollout_worker:
                    self.rollout_worker.stop()
                if self.weight_transfer:
                    self.weight_transfer.destroy()
