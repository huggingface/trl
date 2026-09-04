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

import contextlib
import copy
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.parallelism_config import ParallelismConfig
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.tensor import DTensor, Replicate
from transformers import (
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.distributed.configuration_utils import DistributedConfig
from transformers.distributed.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    _get_parameter_tp_plan,
    _use_local_dtensor_params,
)
from transformers.generation import ContinuousBatchingConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available

from ...chat_template_utils import add_response_schema, parse_response, supports_tool_calling
from ...trainer.base_trainer import _BaseTrainer
from ...trainer.utils import create_model_from_path, nanstd, pad, patch_chunked_lm_head
from .zero_sync_grpo_config import ZeroSyncGRPOConfig


# Functions with this signature can be used as reward functions
RewardFunc = Callable[[list, list], list[float]]


@dataclass(frozen=True)
class TurnRecord:
    """One generation call: the whole prompt this turn, the tokens the model produced, their logprobs."""

    prompt_ids: list[int]
    output_ids: list[int]
    output_log_probs: list[float]


def _chain_to_sequences(turns: list[TurnRecord]) -> tuple[list[dict[str, Any]], int]:
    """Reconcile one conversation's turns (in order) into training rows; fork when the tokens drift.

    Every turn renders the full conversation through the chat template, so a template that rewrites
    history (dropped reasoning, summarized turns) re-tokenizes previously generated tokens differently.
    A turn continues the current row only if its prompt still starts with every token held so far; a
    single changed token forks a new row, so trained tokens are never silently rewritten as context.
    Returns the rows and the number of forks.
    """
    rows: list[dict[str, Any]] = []
    forks = 0
    for turn in turns:
        if rows and turn.prompt_ids[: len(rows[-1]["input_ids"])] == rows[-1]["input_ids"]:
            row = rows[-1]
            context = turn.prompt_ids[len(row["input_ids"]) :]
        else:
            forks += bool(rows)
            row = {"input_ids": [], "completion_mask": [], "logprobs": []}
            rows.append(row)
            context = turn.prompt_ids
        row["input_ids"].extend(context + turn.output_ids)
        row["completion_mask"].extend([0] * len(context) + [1] * len(turn.output_ids))
        row["logprobs"].extend([0.0] * len(context) + turn.output_log_probs)
    return [row for row in rows if any(row["completion_mask"])], forks


def _compute_on_local_view(module):
    """Run `module` on the local view of the replicated parameters in its subtree.

    A replicated parameter is still a DTensor, and an op mixing one with a plain tensor raises. Keeping it a DTensor
    matters because gradient clipping cannot mix the two kinds either, so the unwrapping happens here instead. It
    covers the descendants because a module does not always read its parameters through its own forward: the gated
    delta net convolves with `self.conv1d.weight` itself, so unwrapping the convolution alone would never fire. And
    it is a forward wrapper rather than one context around the step, because gradient checkpointing replays these
    forwards during the backward pass.
    """
    original = type(module).forward
    replicated = [
        submodule
        for submodule in module.modules()
        if any(True for _ in submodule.parameters(recurse=False))
        and all(
            isinstance(param, DTensor) and all(isinstance(p, Replicate) for p in param.placements)
            for param in submodule.parameters(recurse=False)
        )
    ]

    def forward(*args, **kwargs):
        with contextlib.ExitStack() as stack:
            for submodule in replicated:
                stack.enter_context(_use_local_dtensor_params(submodule))
            return original(module, *args, **kwargs)

    module.forward = forward


class _SharedPromptStream:
    """One pass over the prompts, shared by the training loop and by the generation queue.

    The loop pulls a batch per step; the queue pulls whatever else it needs to keep `rollouts_in_flight` rollouts
    generating. Both take from the same iterator, so a prompt is read once and the loop simply advances past what the
    queue already took.
    """

    def __init__(self, loader, trainer):
        self._loader = loader
        self._trainer = trainer
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self._loader)
        self._trainer._prompt_stream = self
        return self

    def __next__(self):
        return next(self._iterator)

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class ZeroSyncGRPOTrainer(_BaseTrainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with zero-sync generation. Generation and
    training share ONE copy of the weights: a transformers continuous batching manager generates from the same
    parameter tensors the optimizer updates in place, so there is no second engine, no weight synchronization and no
    generation/training memory duplication. The engine's per-token logprobs are the exact behavior-policy logprobs of
    the completions and are used as the old policy in the clipped loss.

    Generation never stops. Prompts are submitted to the engine continuously, each completion is collected as it
    finishes, a group's advantages are computed as soon as its last completion lands, and a training batch is formed
    from whichever scored samples are ready first. A slow group never blocks a batch of fast ones; it simply lands in
    a later batch. Completions therefore lag the policy by a bounded number of optimizer steps; the measured logprob
    gap this introduces is small and concentrated in each completion's earliest tokens, and the clipped loss against
    the engine's own logprobs accounts for it.

    Groups are formed per process: each process generates and scores its own prompts, so advantages require no
    cross-process communication.

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.zero_sync_grpo import ZeroSyncGRPOTrainer

    dataset = load_dataset("trl-lib/tldr-chat", split="train")


    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c[0]["content"])) for c in completions]


    trainer = ZeroSyncGRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=reward_num_unique_chars,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object.
        reward_funcs (`Callable` or `list[Callable]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. The functions are provided with one
            group's prompts, completions and completion ids, plus any additional columns in the dataset, and must
            return a list of floats (or `None` for samples the function does not apply to).
        args ([`~trl.experimental.zero_sync_grpo.ZeroSyncGRPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"` in the
            [conversational](dataset_formats#conversational) format (a list of messages). Any additional columns are
            passed to the reward functions.
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's
            name with [`~transformers.AutoProcessor.from_pretrained`]. For vision-language models the processor's
            tokenizer is used; only text-only data is supported.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None,
            None)`):
            A tuple containing the optimizer and the scheduler to use.
    """

    _tag_names = ["trl", "zero-sync-grpo"]
    _name = "Zero-Sync GRPO"

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: ZeroSyncGRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        tools: list[Callable] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = ZeroSyncGRPOConfig(f"{model_name}-ZeroSyncGRPO")

        # Model. The architecture (causal LM or vision-language model) is inferred from the checkpoint; VLMs are
        # supported with text-only data.
        if isinstance(model, str):
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
            # The same weights both generate and train, so default to the checkpoint dtype rather than fp32
            model_init_kwargs.setdefault("dtype", "auto")
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            if args.tp_size > 1:
                # The model is split across the processes rather than placed on one device, and the two ways of
                # deciding where parameters go are mutually exclusive.
                model_init_kwargs["distributed_config"] = DistributedConfig(tp_size=args.tp_size)
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
            if "attn_implementation" not in model_init_kwargs:
                if "linear_attention" in (getattr(model.config.get_text_config(), "layer_types", None) or []):
                    # Hybrid models read the sample boundaries from the flash varlen kwargs; flex attention is
                    # not supported there, and a dense implementation would attend across samples
                    if is_flash_attn_3_available(kernels_fallback_ok=True):
                        model.set_attn_implementation("flash_attention_3")
                    elif is_flash_attn_2_available(kernels_fallback_ok=True):
                        model.set_attn_implementation("flash_attention_2")
                    else:
                        raise ValueError("Packed training on a hybrid model requires flash attention")
                else:
                    # Packed rows attend through a block-diagonal mask; flex attention skips the masked-out
                    # cross-sample blocks, where a dense implementation still pays for them.
                    model.set_attn_implementation("flex_attention")
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `ZeroSyncGRPOConfig`, but your model is already "
                    "instantiated. This argument can only be used when the `model` argument is a string."
                )

        # Processing class. For VLMs this is a processor whose tokenizer handles the text-only conversations.
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        else:
            self._tokenizer = processing_class
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        self.reward_func_names = [func.__name__ for func in reward_funcs]

        # Training arguments
        self.num_generations = args.num_generations
        self.max_completion_length = args.max_completion_length
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.epsilon = args.epsilon
        # Linear attention layers (hybrid models like Qwen3.5) scan packed rows with varlen kernels that take one
        # flat row with explicit boundaries, so packing then goes in a single unpadded row instead of several rows
        self.packed_single_row = "linear_attention" in (
            getattr(model.config.get_text_config(), "layer_types", None) or []
        )
        self.rollouts_in_flight = args.rollouts_in_flight
        self.tp_size = args.tp_size
        # Giving the KV cache memory to the training step only works when generation is quiescent while it runs, which
        # is what the trainer owning the engine gives us.
        if args.release_kv_cache_during_step and args.tp_size == 1:
            raise ValueError(
                "`release_kv_cache_during_step` requires `tp_size > 1`, since generation otherwise runs "
                "in its own thread and is never quiescent."
            )
        self.release_kv_cache_during_step = args.release_kv_cache_during_step

        # The ranks a model is not split across hold replicas of it, which train on their own prompts and have to
        # agree on the update. Their gradients are summed over a group holding one rank per replica: the ranks that
        # carry the same tensor parallel shard. Parameters stay whole, which is what lets generation read them in
        # place, so this is ZeRO-0 over the replicas; the optimizer states are what a ZeRO-2 step would split next.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size % args.tp_size != 0:
            raise ValueError(
                f"The world ({world_size} processes) must divide into replicas of `tp_size` ({args.tp_size}) "
                "processes each."
            )
        self.dp_size = world_size // args.tp_size
        if self.dp_size > 1 and args.parallelism_config is None:
            # The split of the world into replicas follows from `tp_size`, so there is nothing for the caller to
            # decide. Accelerate needs to be told it, or it wraps the model for data parallelism itself, which it
            # cannot do to a tensor-parallel one.
            args.parallelism_config = ParallelismConfig(dp_replicate_size=self.dp_size, tp_size=args.tp_size)
        self._replica_group = None
        self._parameter_owner: dict[str, int] = {}
        if self.dp_size > 1:
            rank = int(os.environ.get("RANK", "0"))
            # Every rank builds every group, in the same order, and keeps the one it belongs to
            groups = [
                torch.distributed.new_group([r for r in range(world_size) if r % args.tp_size == shard])
                for shard in range(args.tp_size)
            ]
            self._replica_group = groups[rank % args.tp_size]
        # Under tensor parallelism the trainer advances generation itself, through a second view of the model, so
        # that every rank issues its collectives in the same order. Both are built in `_init_manager`.
        self._generation_view = None
        self._request_counter = 0
        self._decode_steps = 0
        self.max_tool_calling_iterations = args.max_tool_calling_iterations

        # Tools. `add_response_schema` teaches the tokenizer how to parse tool calls back out of a
        # generation, unless the chat template already carries a response template.
        self.tools = tools
        self.tool_dict = {tool.__name__: tool for tool in tools} if tools else {}
        if tools:
            if not supports_tool_calling(processing_class):
                raise ValueError(
                    "The provided chat template does not support tool calling. The template must be able to render a "
                    "full tool-calling conversation (user -> assistant with tool_calls -> tool)."
                )
            if getattr(self._tokenizer, "response_template", None) is None:
                processing_class = add_response_schema(processing_class)
                self._tokenizer = getattr(processing_class, "tokenizer", processing_class)

        if args.per_device_train_batch_size % self.num_generations != 0:
            raise ValueError(
                f"The per-device train batch size ({args.per_device_train_batch_size}) must be evenly divisible by "
                f"the number of generations per prompt ({self.num_generations}): each step consumes "
                "`per_device_train_batch_size` scored samples while the dataloader feeds "
                "`per_device_train_batch_size / num_generations` prompts."
            )

        # Under tensor parallelism the vocabulary projection is replicated rather than split. The chunked lm_head
        # below multiplies by the weight directly, which a sharded weight cannot serve, and the projection is the one
        # place where splitting buys least: it costs one copy of that matrix per process and removes a gather from
        # every forward. Everything else stays split.
        if args.tp_size > 1:
            output_embeddings = model.get_output_embeddings()
            input_embeddings = model.get_input_embeddings()
            tied = input_embeddings.weight is output_embeddings.weight
            # Kept as a DTensor, replicated rather than split: every parameter then has the same type, which
            # gradient clipping requires, and the chunked lm_head reads its local view.
            weight = output_embeddings.weight
            device_mesh = model._device_mesh
            # Some models keep the projection out of their tensor parallel plan, so it is already a full tensor
            local_weight = weight.full_tensor().contiguous() if isinstance(weight, DTensor) else weight.data
            replicated = nn.Parameter(DTensor.from_local(local_weight, device_mesh, [Replicate()], run_check=False))
            output_embeddings.weight = replicated
            # The transform installed for the split weight would now mix a plain weight with a DTensor input.
            output_embeddings.__dict__.pop("forward", None)
            _compute_on_local_view(output_embeddings)
            if tied:
                input_embeddings.weight = replicated
                # The transform installed for the split weight would now mix a plain weight with a DTensor input.
                input_embeddings.__dict__.pop("forward", None)
                _compute_on_local_view(input_embeddings)
            # An untied input embedding loads outside the plan, as a plain tensor, and the sweep below replicates it

            # Whatever the plan leaves out loads as a plain tensor: norms, and on hybrid models the gated delta
            # net's convolution and gates. Those parameters see the same inputs and produce the same gradients on
            # every rank, so they are replicated, but fused optimizers and gradient clipping still reject a group
            # mixing them with DTensors. Their modules then compute on the local view, like the embeddings above.
            converted = []
            for module in model.modules():
                plain_names = [
                    name for name, param in module.named_parameters(recurse=False) if not isinstance(param, DTensor)
                ]
                if not plain_names:
                    continue
                for name in plain_names:
                    param = getattr(module, name)
                    setattr(
                        module,
                        name,
                        nn.Parameter(
                            DTensor.from_local(param.data, device_mesh, [Replicate()], run_check=False),
                            requires_grad=param.requires_grad,
                        ),
                    )
                converted.append(module)
            # In a second pass, so that each wrapper sees the whole of its subtree already replicated: modules are
            # walked parents first, and a child converted later would not have been picked up.
            for module in converted:
                _compute_on_local_view(module)

        # Compute per-token logprobs without ever materializing the [batch, seq, vocab] logits: the lm_head runs in
        # chunks with an online logsumexp. Long completions make this the difference between training and an OOM.
        # The patch replaces `forward`, and the generation engine decodes through that same object, so the standard
        # forward is kept for calls without labels, which is what decoding does.
        generation_forward = model.forward
        # Under tensor parallelism the replicated lm_head would make every rank compute the whole vocabulary;
        # passing the group makes each rank take every tp_size-th chunk instead, combined with two collectives.
        tp_group = model._device_mesh.get_group() if args.tp_size > 1 else None
        patch_chunked_lm_head(model, chunk_size=8192, temperature=self.temperature, tp_group=tp_group)
        training_forward = model.forward

        def forward(*args, labels=None, **kwargs):
            if labels is None:
                return generation_forward(*args, **kwargs)
            return training_forward(*args, labels=labels, **kwargs)

        model.forward = forward

        def data_collator(features):
            # No data collation is needed in GRPO
            return features

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

        # Zero-sync generation state. The continuous batching manager is created lazily on the first generation (the
        # model must be on its final device first). `_inflight` maps every pending request id to its group; a group's
        # advantages are computed as soon as its last completion lands, and its samples join `_ready`, from which
        # training batches are drawn.
        self._manager = None
        self._pending: deque[dict[str, Any]] = deque()
        self._prompt_stream = None
        self._inflight = {}
        self._ready = deque()

        # Metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_train_dataloader(self):
        # The loop and the generation queue read prompts from one iterator. The queue has to run ahead of the loop to
        # keep the engine full, and reading ahead through a second iterator would hand the same prompts to both.
        return _SharedPromptStream(self._build_train_dataloader(), self)

    def _build_train_dataloader(self):
        # Each step consumes `per_device_train_batch_size` scored samples and every prompt produces
        # `num_generations` of them, so the dataloader feeds `per_device_train_batch_size / num_generations`
        # prompts per step.
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size // self.num_generations,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        finally:
            if self._manager is not None:
                # The generation thread is not a daemon, and Python joins those before it runs any atexit hook, so
                # the process would never exit. A flush would wait for every rollout still in flight, and there are
                # always some, since the trainer keeps `rollouts_in_flight` rollouts generating; none of them will be
                # trained on now.
                self._manager.stop(block=True, timeout=30, hard_stop=True)
                self._manager = None

    def _make_generation_view(self, model):
        """A second view of the model for the engine to decode through, over the same parameters.

        `init_continuous_batching` switches a model to a paged attention implementation, which is written for the
        packed inputs the engine prepares and raises on the training forward. The switch is a setting on the config,
        shared by every module and every thread, so it cannot be flipped around each forward: under tensor parallelism
        there is no moment to flip it, and in the data parallel case the engine decodes in its own thread throughout.
        Giving the engine its own view, with its own config, means the switch never has to happen. The view shares
        every parameter, so an optimizer step is what the engine decodes from, and it costs no extra memory: only the
        module objects and the config are copied.
        """

        # The deepcopy memo maps every original config object to its copy, sub-configs included, so each module
        # of the view keeps the same config it held on the model (composite models give their text and vision
        # submodels their own sub-configs).
        memo: dict[int, Any] = {}
        copy.deepcopy(model.config, memo)

        def clone(module):
            copied = copy.copy(module)
            copied._parameters = dict(module._parameters)
            copied._buffers = dict(module._buffers)
            copied._modules = {name: clone(child) for name, child in module._modules.items()}
            # Fresh hook containers: a shallow copy shares them, so the hooks that advance generation would fire
            # again inside the engine's own forward.
            for attribute, value in list(copied.__dict__.items()):
                if attribute.endswith(("_hooks", "_hooks_with_kwargs")):
                    copied.__dict__[attribute] = type(value)()
            copied.__dict__.pop("forward", None)  # the tensor parallel forward is reinstalled below
            if hasattr(copied, "config"):
                copied.config = memo.get(id(module.config), module.config)
            return copied

        view = clone(model)
        for name, module in view.named_modules():
            style = _get_parameter_tp_plan(parameter_name=name, tp_plan=model.tp_plan or {}, is_weight=False)
            # Replicated parameters are DTensors too, and a transform that splits their input would be wrong.
            sharded = any(
                isinstance(param, DTensor) and any(not isinstance(p, Replicate) for p in param.placements)
                for param in module.parameters(recurse=False)
            )
            if style is not None and style in ALL_PARALLEL_STYLES and sharded:
                ALL_PARALLEL_STYLES[style].install_forward(module, model._device_mesh)
            elif any(isinstance(param, DTensor) for param in module.parameters(recurse=False)):
                _compute_on_local_view(module)
        return view

    def _init_manager(self):
        # The manager is attached to the unwrapped training model: decoding reads the same parameter tensors the
        # optimizer updates in place. warmup() must run before start() so the cuda graphs are captured on the main
        # thread (transformers#48312).
        model = self.accelerator.unwrap_model(self.model)
        generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        cb_kwargs = dict(self.args.continuous_batching_config or {})
        cb_kwargs["return_logprobs"] = True  # the engine's logprobs are the behavior-policy logprobs
        # Generation and training share the device, so the KV cache must leave room for gradients, optimizer states
        # and activations; the engine's own default claims most of the free memory.
        cb_kwargs.setdefault("max_memory_percent", 0.2)
        if self.release_kv_cache_during_step:
            # Async batching keeps a batch in flight while the next is prepared, and the cache cannot be taken from
            # under it. It is worth nothing here anyway, measured at 1.52 against 1.53 s/step.
            cb_kwargs["use_async_batching"] = False
        if self.tp_size > 1:
            # The engine turns NCCL's graph-mixing support off, which fits a pure-decode server but slows the
            # training collectives here (measured 3.62 against 3.71 s/step at batch 128). The trainer strictly
            # alternates generation and training, so captured and eager collectives are never in flight together
            # and the NCCL default, which is also the safe setting, can stay.
            cb_kwargs.setdefault("disable_nccl_graph_mixing", False)
        # Under tensor parallelism the engine decodes through its own view of the model, and the trainer steps it
        # rather than letting it run in the background. The engine and the trainer then hold one NCCL communicator
        # each, and NCCL requires every rank to issue the operations on its communicators in the same host-side order,
        # recommending "a deterministic order issued from a single host thread per-device". Two racing threads cannot
        # promise that, and when the orders disagree the run deadlocks inside an ordinary kernel launch rather than
        # inside a collective. The trainer is that single thread: it runs the engine between its own steps, so the
        # forward and backward have the device to themselves and generation is quiescent while they run.
        self._generation_view = self._make_generation_view(model)
        self._manager = self._generation_view.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=ContinuousBatchingConfig(**cb_kwargs),
        )
        self._manager.warmup()
        if self.tp_size == 1:
            self._manager.start()

    def _tokenize_conversation(self, messages: list[dict[str, Any]]) -> list[int]:
        # Re-tokenize the WHOLE conversation each turn: the reconciler in `_chain_to_sequences` catches template
        # rewrites of earlier turns precisely because the prompt is rebuilt from the message list instead of glued
        # onto held tokens.
        return self.processing_class.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            tools=self.tools or None,
            **self.chat_template_kwargs,
        )["input_ids"][0]

    def _enqueue_group(self, example: dict[str, Any]) -> None:
        # Queue `num_generations` rollouts of the prompt. Each rollout is a chain of turns: a completed turn that
        # calls tools spawns the next turn's request, and the rollout is scored when its final turn lands. They are
        # queued rather than started: `_fill_slots` starts them as the engine has room.
        group = {"example": example, "scored": [], "size": self.num_generations}
        for _ in range(self.num_generations):
            self._pending.append(
                {
                    "group": group,
                    "messages": list(example["prompt"]),
                    "completion": [],
                    "turns": [],
                    "prompt_ids": self._tokenize_conversation(example["prompt"]),
                    "iterations": 0,
                    "tool_calls": 0,
                }
            )

    def _enqueue_group_batch(self, batch) -> None:
        for example in batch:
            self._enqueue_group(example)

    def _fill_slots(self) -> None:
        """Hand queued rollouts to the engine, and read another prompt when it has run out of work.

        A decode step costs about the same whatever number of sequences it carries: it reads the whole model and pays
        the per-layer all-reduces either way. Letting the batch drain between optimizer steps and refill in one burst
        therefore wastes decode steps on a batch that is half empty. The rollouts of a group start as slots free
        rather than all at once, which is no change in kind: zero-sync updates the weights while generation runs, so
        the members of a group already see more than one version of the policy.

        This runs after every completion, which is the moment worth asking at: a rollout just freed its blocks and the
        engine has just run, so an empty waiting queue really means it has room for more rather than meaning it has not
        looked yet. Asking once a step instead, straight after handing over that step's batch, always finds the batch
        still waiting and so never reads anything.
        """
        # Only while training: the stream is the training loop's own, and eval must not consume its prompts.
        idle = self._manager is not None and not self._manager.batch_processor.scheduler.waiting_requests
        if self.model.training and self._prompt_stream is not None and not self._pending and idle:
            try:
                # the stream yields a batch of prompts; taking one and dropping the rest would skip them
                self._enqueue_group_batch(next(self._prompt_stream))
            except StopIteration:
                pass
        while self._pending:
            rollout = self._pending.popleft()
            request_id = self._next_request_id()
            self._manager.add_request(
                rollout["prompt_ids"], request_id=request_id, max_new_tokens=self.max_completion_length
            )
            self._inflight[request_id] = rollout

    def _next_request_id(self) -> str:
        """Name a request identically on every rank.

        `add_request` only queues on the tensor parallel driver and returns nothing elsewhere, while the results come
        back on every rank. Every rank submits the same rollouts in the same order, so a counter names them the same.
        """
        self._request_counter += 1
        return f"zero-sync-{self._request_counter}"

    def _drain(self, timeout: float) -> None:
        # Collect and score every completion the engine has finished; a group's advantages are computed once its last
        # completion lands. A dead background thread never raises from `get_result` (it returns None forever,
        # transformers#48334), so poll `fatal_error` to fail fast instead of spinning.
        while True:
            if self.tp_size > 1:
                # Nothing else advances the engine: the trainer owns it, and this is the only place it runs.
                if self._manager.step():
                    self._decode_steps += 1
                timeout = 0.0
            result = self._manager.get_result(timeout=timeout)
            if result is None:
                fatal_error = self._manager.background_thread_status.fatal_error
                if fatal_error is not None:
                    raise RuntimeError("The continuous batching background thread died.") from fatal_error
                return
            # Capturing the cuda graphs submits the engine's own warmup requests, and their results come back through
            # this same queue. They are not rollouts, so drop them instead of looking them up.
            rollout = self._inflight.pop(result.request_id, None)
            if rollout is not None:
                self._advance_rollout(rollout, result)
            # A rollout that ended freed a slot; a rollout that called a tool took its own slot back
            self._fill_slots()

    def _advance_rollout(self, rollout: dict[str, Any], result) -> None:
        # One turn just finished: record it, and either spawn the next turn (the model called tools) or score the
        # rollout (it produced a final answer). Scoring happens the moment the rollout finishes, so reward
        # computation overlaps generation.
        turn_ids = list(result.generated_tokens)
        rollout["turns"].append(TurnRecord(rollout["prompt_ids"], turn_ids, list(result.logprobs[: len(turn_ids)])))
        if self.tools:
            assistant_message = parse_response(self._tokenizer, turn_ids, prefix=rollout["prompt_ids"])
        else:
            assistant_message = {
                "role": "assistant",
                "content": self._tokenizer.decode(turn_ids, skip_special_tokens=True),
            }
        rollout["messages"].append(assistant_message)
        rollout["completion"].append(assistant_message)

        tool_calls = assistant_message.get("tool_calls")
        may_iterate = (
            self.max_tool_calling_iterations is None or rollout["iterations"] < self.max_tool_calling_iterations
        )
        if tool_calls and may_iterate:
            tool_messages = self._execute_tool_calls(tool_calls)
            rollout["tool_calls"] += len(tool_calls)
            rollout["messages"].extend(tool_messages)  # tool result goes back as a MESSAGE, re-tokenized next turn
            rollout["completion"].extend(tool_messages)
            rollout["iterations"] += 1
            rollout["prompt_ids"] = self._tokenize_conversation(rollout["messages"])
            request_id = self._next_request_id()
            self._manager.add_request(
                rollout["prompt_ids"], request_id=request_id, max_new_tokens=self.max_completion_length
            )
            self._inflight[request_id] = rollout
            return

        group = rollout["group"]
        group["scored"].append(self._score_rollout(group["example"], rollout))
        if len(group["scored"]) == group["size"]:
            self._finalize_group(group)

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        tool_messages = []
        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            tool = self.tool_dict.get(name)
            if tool is None:
                # A hallucinated tool name is a policy error that should decay with training
                tool_messages.append({"role": "tool", "name": name, "content": str({"error": f"unknown tool {name}"})})
                continue
            try:
                result = tool(**function.get("arguments", {}))
            except Exception as error:
                result = {"error": str(error)}
            tool_messages.append({"role": "tool", "name": name, "content": str(result)})
        return tool_messages

    def _score_rollout(self, example: dict[str, Any], rollout: dict[str, Any]) -> dict[str, Any]:
        # The reward functions receive the whole conversation's completion (assistant and tool messages) as
        # one-element lists, with the same signature as everywhere else in TRL.
        completion_ids = [token for turn in rollout["turns"] for token in turn.output_ids]
        reward_kwargs = {key: [example[key]] for key in example if key not in ["prompt"]}

        rewards_per_func = torch.zeros(len(self.reward_funcs))
        for i, reward_func in enumerate(self.reward_funcs):
            output_reward_func = reward_func(
                prompts=[example["prompt"]],
                completions=[rollout["completion"]],
                completion_ids=[completion_ids],
                **reward_kwargs,
            )
            # Convert None values to NaN
            rewards_per_func[i] = torch.nan if output_reward_func[0] is None else output_reward_func[0]

        # Reconcile the turn chain into training rows: one row for a conversation the template only appended to,
        # several when a rewrite forked it.
        sequences, forks = _chain_to_sequences(rollout["turns"])
        return {
            "sequences": sequences,
            "num_completion_tokens": len(completion_ids),
            "tool_calls": rollout["tool_calls"],
            "forks": forks,
            "rewards_per_func": rewards_per_func,
        }

    def _finalize_group(self, group: dict[str, Any]) -> None:
        rewards_per_func = torch.stack([scored["rewards_per_func"] for scored in group["scored"]])

        # A completion for which every reward function returned None is unscorable. nansum would collapse it to 0,
        # which both biases the group baseline and hands the completion a spurious advantage. Mark these rows NaN so
        # they're excluded from the (nan-aware) baseline below; their advantage is forced to 0 afterwards.
        unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)
        rewards = rewards_per_func.nansum(dim=1)
        rewards[unscorable_mask] = torch.nan
        advantages = (rewards - torch.nanmean(rewards)) / (nanstd(rewards) + 1e-4)

        # Unscorable completions carry no learning signal: their reward is NaN here, so zero their advantage to keep
        # them from moving the policy.
        advantages = torch.nan_to_num(advantages, nan=0.0)

        # A rollout produces one row, or several when the chat template rewrote history. They all carry the rollout's
        # advantage, and its metrics ride along on the first one so a fork isn't counted twice.
        for scored, reward, reward_per_func, advantage in zip(
            group["scored"], rewards, rewards_per_func, advantages, strict=True
        ):
            metrics = {
                "reward": reward,
                "rewards_per_func": reward_per_func,
                "completion_length": scored["num_completion_tokens"],
                "tool_calls": scored["tool_calls"],
                "forks": scored["forks"],
            }
            for row in scored["sequences"]:
                self._ready.append({**row, "advantage": advantage, "metrics": metrics})
                metrics = None  # only the rollout's first row carries them

    def _take_from_pool(self, num_samples: int) -> list[dict[str, Any]]:
        """Hand this replica its share of the samples every replica has ready.

        A sample is data: nothing ties it to the replica that generated it. Pooling them lets the batches be built
        for balance instead of for locality, so the replicas reach their forward together and carry the same load.
        Every rank runs the same assignment over the same pool, so no one has to be told the result.
        """
        pool: list[Any] = [None] * self.dp_size
        torch.distributed.all_gather_object(pool, list(self._ready), group=self._replica_group)
        replica = self.accelerator.process_index // self.tp_size

        # Which samples to train on is decided by age, not by length. Preferring the long ones leaves the short
        # ones sitting in the ready lists, and since a rollout is either in flight or waiting there, a growing
        # backlog is fewer sequences decoding at once, which is exactly what makes the engine efficient. Each
        # replica gives up its oldest, and whoever has a surplus covers a replica that is short.
        taking = [min(num_samples, len(samples)) for samples in pool]
        while sum(taking) < num_samples * self.dp_size:
            spare = [r for r in range(self.dp_size) if taking[r] < len(pool[r])]
            taking[min(spare, key=lambda r: taking[r])] += 1
        chosen = [
            (owner, index, pool[owner][index]) for owner in range(self.dp_size) for index in range(taking[owner])
        ]

        # Longest first into the emptiest replica, by the sum of squared lengths: attention is quadratic, so that
        # is what predicts both the step time and the activation peak, and it is the quantity worth equalizing.
        loads = [0] * self.dp_size
        assigned: list[list[Any]] = [[] for _ in range(self.dp_size)]
        for owner, index, sample in sorted(chosen, key=lambda item: -len(item[2]["input_ids"])):
            open_replicas = [r for r in range(self.dp_size) if len(assigned[r]) < num_samples]
            target = min(open_replicas, key=lambda r: loads[r])
            assigned[target].append((owner, index, sample))
            loads[target] += len(sample["input_ids"]) ** 2
        taken = {index for group in assigned for owner, index, _ in group if owner == replica}
        self._ready = deque(sample for index, sample in enumerate(self._ready) if index not in taken)
        self._metrics["train"]["batch/row_imbalance"].append(max(loads) / (sum(loads) / len(loads)))
        self._metrics["train"]["batch/from_other_replicas"].append(
            sum(1 for owner, _, _ in assigned[replica] if owner != replica) / max(len(assigned[replica]), 1)
        )
        return [sample for _, _, sample in assigned[replica]]

    def _prepare_inputs(self, generation_batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        if self._manager is None:
            self._init_manager()

        # A step queues one batch of prompts and consumes one batch of samples, so the depth of the pipeline is
        # whatever it starts with. It is filled once, here, by reading prompts ahead of the training loop; from then
        # on each step queues its own batch and `_fill_slots` starts them as rollouts finish, which is what holds the
        # decode batch at `rollouts_in_flight` instead of letting it drain and refill once per step.
        #
        # Filling it used to mean submitting the first batch `generation_ahead` times over, so the opening steps
        # trained on copies of a single prompt.
        num_samples = len(generation_batch) * self.num_generations
        self._enqueue_group_batch(generation_batch)
        # Read further prompts from the same stream the loop is walking, enough to keep every slot filled and one
        # batch spare: prompts arrive a batch at a time while rollouts finish one at a time, and without that spare a
        # burst of completions would leave slots idle until the next step. The loop carries on from where this left
        # off, so no prompt is read twice and none is skipped.
        if mode == "train" and self._prompt_stream is not None:
            while len(self._pending) < self.rollouts_in_flight + num_samples:
                try:
                    self._enqueue_group_batch(next(self._prompt_stream))
                except StopIteration:
                    break
        self._fill_slots()

        # Time spent waiting for the engine. Near zero means generation is fully hidden behind the training step; a
        # large value means the engine is the bottleneck, so raise `rollouts_in_flight` (more requests generating at
        # once) or give it a bigger KV pool.
        if self.release_kv_cache_during_step:
            self._manager.restore_memory()  # generation needs its cache back before it can run
        wait_start = time.perf_counter()
        if self._replica_group is None:
            while len(self._ready) < num_samples:
                self._drain(timeout=1.0)
        else:
            # The replicas draw one pool between them. A replica whose completions come back short would otherwise
            # fill its batch early, train, and then sit at the gradient sum waiting for a replica still decoding,
            # and it would carry a lighter forward while that one carries the heavy tail alone. Waiting for the
            # count to cover every replica, then handing the samples out, is what keeps the steps aligned.
            # Asking every replica for its count costs a collective, and between two of them the engine takes one
            # decode step, on the same devices through a different communicator. Asking once per decode step was
            # enough to cost 10% of decoding. Asked once every eight instead, on the same schedule everywhere: a
            # replica that skipped a collective the others were waiting on would hang them all.
            drained = 0
            while True:
                if drained % 8 == 0:
                    counts = torch.tensor([len(self._ready)], device=self.accelerator.device)
                    torch.distributed.all_reduce(counts, group=self._replica_group)
                    if int(counts.item()) >= num_samples * self.dp_size:
                        break
                self._drain(timeout=1.0)
                drained += 1
        self._metrics[mode]["generation_wait_s"].append(time.perf_counter() - wait_start)
        if self.tp_size > 1:
            # How far generation got for this step. Under tensor parallelism the trainer advances the engine itself,
            # between its own steps, so this counts the decode steps taken while waiting for enough scored samples.
            self._metrics[mode]["generation/decode_steps"].append(self._decode_steps)
            self._decode_steps = 0
        if self.release_kv_cache_during_step:
            # Generation is done for this step, so hand its memory to the forward and backward that come next. The
            # rollouts in flight lose their cache and re-prefill when they are next scheduled, which is what the
            # option costs: the weights move during the step anyway, so the cache it throws away was already stale.
            released = self._manager.release_memory()
            self._metrics[mode]["generation/kv_released_gib"].append(released / 2**30)
        if self._replica_group is not None:
            samples = self._take_from_pool(num_samples)
        else:
            # Packing already removes the padding, and picking like-length samples would concentrate the long
            # ones into a single batch, whose activations then blow past memory (a batch of 256 samples all near
            # the length cap is over twice the tokens of a mixed one). Arrival order keeps the mix.
            samples = [self._ready.popleft() for _ in range(num_samples)]

        # Metrics of the rollouts this step trains on. They are per process: groups are formed and scored locally,
        # and `Trainer.log` reports process zero.
        rollout_metrics = [sample["metrics"] for sample in samples if sample["metrics"] is not None]
        rewards = torch.stack([metrics["reward"] for metrics in rollout_metrics])
        rewards_per_func = torch.stack([metrics["rewards_per_func"] for metrics in rollout_metrics])
        for i, reward_func_name in enumerate(self.reward_func_names):
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(
                torch.nanmean(rewards_per_func[:, i]).item()
            )
        self._metrics[mode]["reward"].append(torch.nanmean(rewards).item())
        self._metrics[mode]["reward_std"].append(nanstd(rewards).item())
        self._metrics[mode]["completions/mean_length"].append(
            sum(metrics["completion_length"] for metrics in rollout_metrics) / len(rollout_metrics)
        )
        if self.tools:
            self._metrics[mode]["tools/call_count"].append(
                sum(metrics["tool_calls"] for metrics in rollout_metrics) / len(rollout_metrics)
            )
            self._metrics[mode]["completions/forks"].append(
                sum(metrics["forks"] for metrics in rollout_metrics) / len(rollout_metrics)
            )

        # Build the training tensors. Each row already carries its full token sequence, the mask over the trained
        # (model-generated) tokens and the engine's behavior-policy logprobs aligned to them, 0.0 on context.
        pad_token_id = self._tokenizer.pad_token_id
        if self.packed_single_row:
            # One flat row, no padding: samples go back to back and the boundaries travel as cu_seq_lens_q, which
            # the varlen attention kernels and the linear attention layers consume directly. Position ids still
            # restart at every sample, and the loss shift stays sound at the seams for the same reason as below.
            lengths = torch.tensor([len(s["input_ids"]) for s in samples])
            cu_seq_lens = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)])
            return {
                "input_ids": torch.tensor([t for s in samples for t in s["input_ids"]]).unsqueeze(0).to(device),
                "position_ids": torch.cat([torch.arange(len(s["input_ids"])) for s in samples])
                .unsqueeze(0)
                .to(device),
                "completion_mask": torch.tensor([m for s in samples for m in s["completion_mask"]], dtype=torch.long)
                .unsqueeze(0)
                .to(device),
                "old_per_token_logps": torch.tensor([lp for s in samples for lp in s["logprobs"]], dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                "advantages": torch.cat([s["advantage"].repeat(len(s["input_ids"])) for s in samples])
                .to(torch.float32)
                .unsqueeze(0)
                .to(device),
                "cu_seq_lens_q": cu_seq_lens.to(device),
                "cu_seq_lens_k": cu_seq_lens.to(device),
                "max_length_q": int(lengths.max()),
                "max_length_k": int(lengths.max()),
            }
        # Pack the samples back to back, first-fit by decreasing length, instead of padding each to the
        # longest. Position ids restart at every sample and no attention mask is passed, so the model builds
        # the block-diagonal mask from them. The global shift in the loss stays sound at the seams: the first
        # token of a sample is always context (mask 0), so no term ever crosses two samples. Advantages and
        # behavior logprobs become per-token, since one row now holds several samples.
        capacity = max(4096, max(len(s["input_ids"]) for s in samples))
        rows = []  # each row is a list of samples whose total length fits the capacity
        row_space = []
        for sample in sorted(samples, key=lambda s: len(s["input_ids"]), reverse=True):
            length = len(sample["input_ids"])
            for i, space in enumerate(row_space):
                if length <= space:
                    rows[i].append(sample)
                    row_space[i] -= length
                    break
            else:
                rows.append([sample])
                row_space.append(capacity - length)
        input_ids, position_ids, completion_mask, old_per_token_logps, advantages = [], [], [], [], []
        for row in rows:
            input_ids.append(torch.tensor([t for s in row for t in s["input_ids"]]))
            position_ids.append(torch.cat([torch.arange(len(s["input_ids"])) for s in row]))
            completion_mask.append(torch.tensor([m for s in row for m in s["completion_mask"]], dtype=torch.long))
            old_per_token_logps.append(torch.tensor([lp for s in row for lp in s["logprobs"]], dtype=torch.float32))
            advantages.append(torch.cat([s["advantage"].repeat(len(s["input_ids"])) for s in row]).to(torch.float32))
        return {
            "input_ids": pad(input_ids, padding_value=pad_token_id, padding_side="right").to(device),
            # Padding restarts the position ids, so the mask treats it as one more (never-trained) sample
            "position_ids": pad(position_ids, padding_value=0, padding_side="right").to(device),
            "completion_mask": pad(completion_mask, padding_value=0, padding_side="right").to(device),
            "old_per_token_logps": pad(old_per_token_logps, padding_value=0.0, padding_side="right").to(device),
            "advantages": pad(advantages, padding_value=0.0, padding_side="right").to(device),
        }

    def create_optimizer(self, model=None):
        """Give each replica the optimizer state of only its share of the parameters.

        Adam carries two moments per parameter, so its state is what usually decides whether a model fits. The
        replicas split the parameters between them, each updates its own share, and passes the result back to the
        others. The parameters themselves stay whole everywhere, which is what generation reads.

        Measured on a 14B at tp=4 with two replicas: 9.2 GiB less per GPU at the peak, and the step itself takes
        half as long, since each replica now updates half the parameters.
        """
        if self._replica_group is None or self.optimizer is not None:
            return super().create_optimizer(model)

        opt_model = self.model if model is None else model
        trainable = [(name, param) for name, param in opt_model.named_parameters() if param.requires_grad]
        # Largest first into the emptiest replica, so the shares come out even whatever the parameter sizes are.
        self._parameter_owner = {}
        held = [0] * self.dp_size
        for name, param in sorted(trainable, key=lambda item: -item[1].numel()):
            owner = min(range(self.dp_size), key=lambda replica: held[replica])
            self._parameter_owner[name] = owner
            held[owner] += param.numel()

        replica = self.accelerator.process_index // self.tp_size
        decay = set(self.get_decay_parameter_names(opt_model))
        mine = [(name, param) for name, param in trainable if self._parameter_owner[name] == replica]
        groups = [
            {"params": [p for n, p in mine if n in decay], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in mine if n not in decay], "weight_decay": 0.0},
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

        step = self.optimizer.step

        def step_and_share(*args, **kwargs):
            out = step(*args, **kwargs)
            self._share_updated_parameters()
            # The optimizer only knows this replica's share, so its zero_grad would leave the rest holding
            # gradients that the next step would accumulate onto.
            opt_model.zero_grad(set_to_none=True)
            return out

        self.optimizer.step = step_and_share
        return self.optimizer

    def _share_updated_parameters(self) -> None:
        """Send each parameter from the replica that updated it to the others."""
        model = self.accelerator.unwrap_model(self.model)
        shard = self.accelerator.process_index % self.tp_size
        with torch.no_grad():
            for name, param in model.named_parameters():
                owner = self._parameter_owner.get(name)
                if owner is None:
                    continue
                # Every rank in the group holds the same slice of the model, so the local views line up
                local = param.data.to_local() if isinstance(param.data, DTensor) else param.data
                torch.distributed.broadcast(local, src=owner * self.tp_size + shard, group=self._replica_group)

    def training_step(self, *args, **kwargs):
        output = super().training_step(*args, **kwargs)
        # `sync_gradients` is true only on the last micro-step, so the replicas exchange one summed gradient per
        # optimizer step rather than one per accumulation micro-step.
        # Without tensor parallelism the model is a plain DDP one and accelerate already reduced the gradients;
        # DDP cannot wrap a tensor-parallel model, which is why the reduction is done here instead.
        if self.tp_size > 1 and self._replica_group is not None and self.accelerator.sync_gradients:
            for param in self.accelerator.unwrap_model(self.model).parameters():
                if param.grad is None:
                    continue
                # A sharded gradient is summed shard by shard: every rank in the group holds the same shard of the
                # model, so summing their local views is the same as summing the whole gradients.
                grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                torch.distributed.all_reduce(grad, group=self._replica_group)
                grad /= self.dp_size
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        completion_mask = inputs["completion_mask"]
        advantages = inputs["advantages"]

        # Per-token logprobs of the sampled tokens under the current policy, computed by the chunked lm_head, so
        # only completion positions are scored and the full-vocab logits never materialize.
        boundary_kwargs = {
            key: inputs[key]
            for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
            if key in inputs
        }
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            position_ids=inputs.get("position_ids"),
            labels=inputs["input_ids"],
            completion_mask=completion_mask,
            **boundary_kwargs,
        )
        per_token_logps = outputs["log_probs"]
        mask = completion_mask[:, 1:].to(per_token_logps.dtype)

        # Clipped surrogate loss. The old policy is the generation engine itself: its logprobs are exact for the
        # sampled tokens, so no extra forward pass is needed to obtain them.
        old_per_token_logps = inputs["old_per_token_logps"][:, 1:]
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        # Packed rows hold several samples, so the advantage is per token there; padded rows hold one each
        advantages = advantages[:, 1:] if advantages.dim() == 2 else advantages.unsqueeze(1)
        per_token_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        # Log the metrics
        mode = "train" if self.model.training else "eval"
        is_clipped = ((ratio < 1 - self.epsilon) | (ratio > 1 + self.epsilon)) & (mask > 0)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(is_clipped.sum() / mask.sum().clamp(min=1.0)).mean().item()
        )
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather((outputs["entropy"] * mask).sum() / mask.sum().clamp(min=1.0)).mean().item()
        )
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
