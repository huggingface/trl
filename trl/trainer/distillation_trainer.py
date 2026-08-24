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

import copy
import inspect
import math
import os
import textwrap
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging.version import Version
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.utils import is_liger_kernel_available, is_peft_available, is_rich_available

from ..data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from ..distributed import DistributedBackend
from ..extras.profiling import profiling_context, profiling_decorator
from ..generation.vllm_generation import VLLMGeneration
from ..import_utils import is_vllm_available
from ..models import prepare_deepspeed
from ..models.utils import _ForwardRedirection, unwrap_model_for_generation
from .base_trainer import _BaseTrainer
from .distillation_config import DistillationConfig
from .utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    maybe_gather_lm_head_ctx,
    pad,
    print_prompt_completions_sample,
    repeat_iterable_dataset,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, PromptLearningConfig, get_peft_model
    from peft.tuners.tuners_utils import BaseTunerLayer


if is_trackio_available():
    import trackio


if is_wandb_available():
    import wandb


logger = get_logger(__name__)


# Number of valid completion positions projected through the `lm_head` per chunk in the memory-efficient JSD loss
# (mirrors SFT's `_CHUNKED_LM_HEAD_CHUNK_SIZE`).
_CHUNKED_LM_HEAD_CHUNK_SIZE = 256


def _chunk(h_s, w_s, b_s, s_scale, s_softcap, h_t, w_t, b_t, t_scale, t_softcap, beta, temperature, valid):
    # Project both hidden states to vocab logits inside the checkpointed body so only `(chunk, H)` is retained across
    # the backward, never `(chunk, V)`. ZeRO-3 shards the `lm_head`, so gather it tightly around each projection.
    # `logit_scale` (Cohere) / `final_logit_softcapping` (Gemma) are applied per model to match its full forward.
    with maybe_gather_lm_head_ctx(w_s, b_s):
        student_logits = h_s.float() @ w_s.float().t()
        if b_s is not None:
            student_logits = student_logits + b_s.float()
    if s_scale != 1.0:
        student_logits = student_logits * s_scale
    if s_softcap is not None:
        student_logits = s_softcap * torch.tanh(student_logits / s_softcap)
    # The teacher is a fixed target: compute its logits under `no_grad` so the projection builds no autograd graph
    # and the teacher accumulates no gradients (the teacher params are not frozen by `prepare_model`). Everything
    # downstream inherits this since `teacher_logits` is already detached.
    with maybe_gather_lm_head_ctx(w_t, b_t), torch.no_grad():
        teacher_logits = h_t.float() @ w_t.float().t()
        if b_t is not None:
            teacher_logits = teacher_logits + b_t.float()
    if t_scale != 1.0:
        teacher_logits = teacher_logits * t_scale
    if t_softcap is not None:
        teacher_logits = t_softcap * torch.tanh(teacher_logits / t_softcap)
    # Distillation (softmax) temperature: soften both distributions before the divergence, applied after any
    # per-model scaling/softcapping (matching the model's full forward, then the loss's temperature).
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # beta: 0 = forward KL, 1 = reverse KL, else generalized JSD. `F.kl_div(input, target)` computes
    # `target * (log target - input)`, hence the swapped argument order relative to the KL written in the paper.
    if beta == 0.0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]), dim=0
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    # A chunk's tail may hold positions packed out of the valid prefix; zero those rows before summing.
    per_token_jsd = jsd.sum(dim=-1) * valid
    per_token_entropy = -(student_log_probs.exp() * student_log_probs).sum(dim=-1) * valid
    return per_token_jsd.sum(), per_token_entropy.sum()


def _chunked_divergence_loss(
    student_hidden_states: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float,
    chunk_size: int,
    num_items_in_batch: torch.Tensor | int | None = None,
    student_lm_head_bias: torch.Tensor | None = None,
    teacher_lm_head_bias: torch.Tensor | None = None,
    student_logit_scale: float = 1.0,
    teacher_logit_scale: float = 1.0,
    student_final_logit_softcapping: float | None = None,
    teacher_final_logit_softcapping: float | None = None,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient generalized JSD over student/teacher hidden states and their `lm_head` weights.

    The full `lm_head` projections are never materialized. Valid (unmasked) completion positions are packed to the
    front (via `argsort` on the completion mask, a static-shape op) and processed in chunks of `chunk_size`, rounding
    the count up to a whole chunk so masked positions land in a skippable tail. Each chunk's `[chunk_size, vocab_size]`
    logits (for both models) are kept alive only during its own forward/backward via gradient checkpointing, so peak
    logits memory is `2 * chunk_size * vocab_size` instead of `2 * batch_size * seq_len * vocab_size`.

    Args:
        student_hidden_states (`torch.Tensor`):
            Student backbone output of shape `(B, K, H)`, aligned to the completion tokens (before the `lm_head`).
        teacher_hidden_states (`torch.Tensor`):
            Teacher backbone output of shape `(B, K, H)`, aligned to the same completion tokens.
        student_lm_head_weight (`torch.Tensor`):
            Student `lm_head` weight of shape `(V, H)`.
        teacher_lm_head_weight (`torch.Tensor`):
            Teacher `lm_head` weight of shape `(V, H)`.
        completion_mask (`torch.Tensor`):
            Binary mask of shape `(B, K)`; `1` marks completion positions included in the loss.
        beta (`float`):
            Interpolation coefficient. `0.0` = forward KL, `1.0` = reverse KL, else generalized JSD.
        chunk_size (`int`):
            Number of valid positions processed per chunk. Peak memory scales linearly with this.
        num_items_in_batch (`torch.Tensor`, `int` or `None`, *optional*):
            Total number of valid tokens across the global batch. When provided, the loss is reduced as `sum /
            num_items_in_batch` (gradient-accumulation-correct); when `None`, reduction is `mean` over local valid
            positions.
        student_lm_head_bias (`torch.Tensor`, *optional*):
            Student `lm_head` bias of shape `(V,)`, added to each chunk's logits when provided.
        teacher_lm_head_bias (`torch.Tensor`, *optional*):
            Teacher `lm_head` bias of shape `(V,)`, added to each chunk's logits when provided.
        student_logit_scale (`float`, *optional*, defaults to `1.0`):
            Multiplier applied to the student's logits before the softmax (Cohere-style `logit_scale`).
        teacher_logit_scale (`float`, *optional*, defaults to `1.0`):
            Multiplier applied to the teacher's logits before the softmax.
        student_final_logit_softcapping (`float`, *optional*):
            If set, applies `softcap * tanh(logits / softcap)` to the student's logits (Gemma-style), after the scale.
        teacher_final_logit_softcapping (`float`, *optional*):
            If set, applies `softcap * tanh(logits / softcap)` to the teacher's logits, after the scale.
        temperature (`float`, *optional*, defaults to `1.0`):
            Softmax temperature applied to both distributions before the divergence, after any scale/softcapping.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: scalar loss, sum of per-token student entropy (in nats), and
        number of valid completion positions — all over the local batch. Raw sums are returned so callers can reduce
        correctly across ranks.
    """
    # Under FSDP2, lm_head.weight is a DTensor (Shard(0) or Replicate). Passing it directly into the
    # gradient-checkpointed chunk loop causes FSDP2 to re-gather it once per chunk during backward recomputation.
    # full_tensor() converts it to a plain tensor once; all chunks reference that tensor, so only one all-gather occurs
    # (in full_tensor()'s backward). Done per model since the student and teacher have their own heads.
    if isinstance(student_lm_head_weight, torch.distributed.tensor.DTensor):
        student_lm_head_weight = student_lm_head_weight.full_tensor()
        if student_lm_head_bias is not None:
            student_lm_head_bias = student_lm_head_bias.full_tensor()
    if isinstance(teacher_lm_head_weight, torch.distributed.tensor.DTensor):
        teacher_lm_head_weight = teacher_lm_head_weight.full_tensor()
        if teacher_lm_head_bias is not None:
            teacher_lm_head_bias = teacher_lm_head_bias.full_tensor()

    # Each model flattens with its own hidden width: the teacher may be wider/narrower than the student (only the
    # vocabulary must match), and each projects through its own `lm_head`.
    h_s = student_hidden_states.reshape(-1, student_hidden_states.size(-1))
    h_t = teacher_hidden_states.reshape(-1, teacher_hidden_states.size(-1))
    valid = completion_mask.reshape(-1) != 0
    n_valid_tensor = valid.sum()

    entropy_sum = h_s.new_zeros((), dtype=torch.float32)
    if n_valid_tensor == 0:
        # Whole micro-batch masked. Keep the loss connected to the autograd graph through every trainable parameter so
        # `.backward()` succeeds and DDP / FSDP gradient sync doesn't hang on a missing param. Only the student carries
        # gradients (the teacher is frozen).
        with maybe_gather_lm_head_ctx(student_lm_head_weight, student_lm_head_bias):
            loss = (h_s.float().sum() + student_lm_head_weight.float().sum()) * 0.0
            if student_lm_head_bias is not None:
                loss = loss + student_lm_head_bias.float().sum() * 0.0
        return loss, entropy_sum, n_valid_tensor

    # Pack valid positions to the front so masked ones form whole trailing chunks. `argsort` on the boolean mask is a
    # static-shape op (unlike `h_s[valid]`, whose output shape is data-dependent and poisons XLA compilation).
    order = valid.to(torch.int8).argsort(descending=True, stable=True)
    h_s = h_s[order]
    h_t = h_t[order]
    valid = valid[order]

    # Process only the whole chunks covering the valid prefix: bounds XLA recompiles and drops fully-masked chunks on GPU.
    n_padded = (n_valid_tensor / chunk_size).ceil().to(torch.int64) * chunk_size

    loss = h_s.new_zeros((), dtype=torch.float32)
    for start in range(0, n_padded, chunk_size):
        chunk_loss, chunk_entropy = torch.utils.checkpoint.checkpoint(
            _chunk,
            h_s[start : start + chunk_size],
            student_lm_head_weight,
            student_lm_head_bias,
            student_logit_scale,
            student_final_logit_softcapping,
            h_t[start : start + chunk_size],
            teacher_lm_head_weight,
            teacher_lm_head_bias,
            teacher_logit_scale,
            teacher_final_logit_softcapping,
            beta,
            temperature,
            valid[start : start + chunk_size].float(),
            use_reentrant=False,
        )
        loss = loss + chunk_loss
        entropy_sum = entropy_sum + chunk_entropy

    if num_items_in_batch is None:
        loss = loss / n_valid_tensor
    else:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss, entropy_sum, n_valid_tensor


class DistillationTrainer(_BaseTrainer):
    """
    Trainer for knowledge distillation. The student is trained on-policy — it generates the completions itself — to
    match the teacher's next-token distribution under a generalized Jensen-Shannon divergence (interpolating forward
    KL, reverse KL, and JSD via `beta`), as introduced in [On-Policy Distillation of Language
    Models](https://huggingface.co/papers/2306.13649).

    Example:

    ```python
    >>> from trl import DistillationTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("trl-lib/tldr", split="train")

    >>> trainer = DistillationTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     teacher_model="Qwen/Qwen2.5-1.5B-Instruct",
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using `<ModelArchitecture>.from_pretrained` (where `<ModelArchitecture>` is derived from the model
              config) with the keyword arguments in `args.model_init_kwargs`. If `dtype` is not specified in
              `args.model_init_kwargs`, it defaults to `float32`. This differs from
              [`~transformers.PreTrainedModel.from_pretrained`], where (since Transformers v5) the dtype is inferred
              from the model config.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
            - A [`~peft.PeftModel`] object. Only causal language models are supported.
        teacher_model (`str` or [`~transformers.PreTrainedModel`], *optional*):
            Teacher model whose next-token distribution the student is trained to match. Can be a *model id* / path
            (loaded like `model`, using `args.teacher_model_init_kwargs`) or an instantiated
            [`~transformers.PreTrainedModel`]. It must share the student's vocabulary. May be omitted by subclasses
            that supply the teacher another way (e.g. a remote server).
        args ([`DistillationConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`], *optional*):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            When `train_dataset` is an [`~datasets.IterableDataset`] (e.g. a streaming dataset), `max_steps` must be
            set in the training arguments, since its length cannot be inferred and the total number of training steps
            is required to bound the training loop and configure the learning rate scheduler.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`], [`~datasets.DatasetDict`], [`~datasets.IterableDatasetDict`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
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
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier. Combine with `peft_config`
            for QLoRA training. Ignored if the model is already instantiated.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "distillation"]
    _name = "Distillation"
    _paper = {
        "title": "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes",
        "id": "2306.13649",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{agarwal2024on-policy,
                title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
                author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
                year         = 2024,
                booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
                publisher    = {OpenReview.net},
                url          = {https://openreview.net/forum?id=3zKtaqxLhW},
            }"""),
    }

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        teacher_model: str | PreTrainedModel = None,
        args: DistillationConfig | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DistillationConfig(f"{model_name}-Distillation")

        # Student model loading
        # `_VALID_DICT_FIELDS` already parses any JSON-string form of these in `DistillationConfig.__post_init__`, so
        # they are dicts (or None) here; copy so the setdefaults below don't mutate the config.
        model_init_kwargs = dict(args.model_init_kwargs or {})
        teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs or {})
        if isinstance(model, str):
            model_name_or_path = model
            if quantization_config is not None:
                if "quantization_config" in model_init_kwargs:
                    raise ValueError(
                        "You set `quantization_config` both as a trainer argument and in `args.model_init_kwargs`. "
                        "Please set it in only one place, preferably as a trainer argument."
                    )
                model_init_kwargs["quantization_config"] = quantization_config
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            model_name_or_path = get_config_model_id(model.config)
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistillationConfig`, but your model is already "
                    "instantiated. The `model_init_kwargs` will be ignored."
                )
            if quantization_config is not None:
                logger.warning(
                    "You passed `quantization_config` to the trainer, but your model is already instantiated. The "
                    "`quantization_config` will be ignored."
                )
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                model_name_or_path,
                truncation_side="left",
                padding_side="left",
                trust_remote_code=args.trust_remote_code,
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # PEFT
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError(
                    "You passed `peft_config` but the `peft` library is not installed. "
                    "Install it with `pip install trl[peft]`."
                )
            if not isinstance(peft_config, PeftConfig):
                raise TypeError(
                    f"`peft_config` must be a `peft.PeftConfig` instance (e.g. `peft.LoraConfig`), "
                    f"got {type(peft_config).__name__}."
                )
            if is_peft_model(model):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                    "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                    "with the new `peft_config` to the trainer."
                )
            # Create PEFT model
            # ZeRO-3 + PEFT for non-quantized models:
            # - PEFT's default autocast_adapter_dtype=True upcasts LoRA adapter params to fp32 even when the base model is bf16.
            # - ZeRO-3's _allgather_params_coalesced allocates output buffers using the dtype of the first persistent parameter,
            #   so mixed-dtype persistent_parameters (bf16 base + fp32 LoRA) cause a TypeError on the first optimizer step.
            # - Passing autocast_adapter_dtype=False keeps adapter params in the base model dtype (bf16), fixing the mismatch.
            # - This is safe: the fp32 upcast is a QLoRA-specific concern (low-bit quantized base models), not needed for
            #   non-quantized bf16 training.
            # - See:
            #   - TRL issue: https://github.com/huggingface/trl/issues/6089
            #   - Upstream issue: https://github.com/deepspeedai/DeepSpeed/issues/8072
            # - autocast_adapter_dtype was introduced in PEFT 0.12.0; before, no upcast existed: no need to pass the kwarg
            get_peft_model_kwargs = {}
            if (
                args.deepspeed_plugin is not None
                and args.deepspeed_plugin.zero_stage == 3
                and not _is_quantized_model
                and Version(peft.__version__) >= Version("0.12.0")
            ):
                get_peft_model_kwargs["autocast_adapter_dtype"] = False
            model = get_peft_model(model, peft_config, **get_peft_model_kwargs)

        # PEFT + DeepSpeed ZeRO-3 requires reentrant checkpointing. For more details, see
        # https://github.com/huggingface/trl/issues/2514#issuecomment-2692152703.
        # Can be removed once https://github.com/deepspeedai/DeepSpeed/pull/8130 is merged and released.
        if (
            is_peft_model(model)
            and args.deepspeed_plugin is not None
            and args.deepspeed_plugin.zero_stage == 3
            and args.gradient_checkpointing
        ):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            use_reentrant = args.gradient_checkpointing_kwargs.get("use_reentrant")
            if use_reentrant is False:
                logger.warning(
                    "You are using PEFT with DeepSpeed ZeRO-3 and gradient checkpointing with `use_reentrant=False`. "
                    "`use_reentrant` is forced to `True` in this configuration to ensure correct training. To remove "
                    "this warning, unset `use_reentrant` in `gradient_checkpointing_kwargs` or set it to `True`."
                )
            args.gradient_checkpointing_kwargs["use_reentrant"] = True

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        if _is_quantized_model:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Both loss paths (chunked and Liger) read `lm_head.weight` directly and run the backbone via
        # `_get_last_hidden_state`, bypassing `PeftModel.forward()` — so PEFT setups that live outside the backbone
        # weights are silently ignored. Fail loudly rather than train on a silently-wrong objective.
        if is_peft_model(model):
            # When the LM head is targeted by a PEFT adapter (`"lm_head"` in `target_modules`), `lm_head.weight` is the
            # frozen base weight and the trainable adapter lives in separate submodules the loss never sees, so the head
            # adapter would receive no gradient.
            output_embeddings = model.get_output_embeddings()
            if isinstance(output_embeddings, BaseTunerLayer):
                raise ValueError(
                    "Applying a PEFT adapter to `lm_head` is not supported. The distillation loss reads "
                    "`lm_head.weight` directly, so the adapter on the head is ignored and never trained. Remove "
                    "`'lm_head'` from your `target_modules`."
                )
            # Prompt-learning methods (PromptTuning, PrefixTuning, P-Tuning) inject virtual tokens via
            # `PeftModel.forward()`, which the loss bypasses by calling the backbone directly, so virtual tokens are
            # never prepended and the loss is computed on the wrong sequence.
            if any(isinstance(cfg, PromptLearningConfig) for cfg in model.peft_config.values()):
                raise ValueError(
                    "Prompt-learning PEFT methods (PromptTuning, PrefixTuning, P-Tuning) are not supported. The "
                    "distillation loss bypasses `PeftModel.forward()` by calling the backbone directly, so virtual "
                    "tokens are never prepended and the loss is computed on the wrong sequence. Use a weight-based "
                    "adapter such as LoRA instead."
                )

        # The chunked JSD loss (and the Liger path) call the student backbone directly, bypassing the DDP/FSDP
        # wrapper's forward; route them through `_forward_redirection` so `prepare_for_backward()` still fires.
        self._forward_redirection = _ForwardRedirection()

        # Liger fused JSD loss
        self.use_liger_kernel = False
        if args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the distillation loss. Run "
                    "`pip install liger-kernel`."
                )
            self.liger_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            self.use_liger_kernel = True

        # Teacher model setup
        # `teacher_model` may be None: subclasses (e.g. ServerDistillationTrainer) supply the teacher another way.
        if teacher_model is not None:
            if isinstance(teacher_model, str):
                dtype = teacher_model_init_kwargs.get("dtype")
                teacher_model_init_kwargs["dtype"] = dtype if dtype in ["auto", None] else getattr(torch, dtype)
                if args.teacher_model_revision is not None:
                    teacher_model_init_kwargs.setdefault("revision", args.teacher_model_revision)
                # Distributed training requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    teacher_model_init_kwargs["device_map"] = None
                teacher_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
                teacher_model = create_model_from_path(teacher_model, **teacher_model_init_kwargs)
            elif args.teacher_model_init_kwargs is not None:
                raise ValueError(
                    "You passed teacher_model_init_kwargs to the config, but your teacher_model is already "
                    "instantiated."
                )

        # Iterable datasets can't be indexed, so the RepeatSampler can't be attached to them. Instead, the sampler's
        # ordering is reproduced by streaming (see `get_train_dataloader`/`get_eval_dataloader` and
        # `repeat_iterable_dataset`). This requires `dispatch_batches=False`: with the default dispatch path, batches
        # are collated on the main process and Accelerate tries to concatenate the string `prompt` column, which fails;
        # `dispatch_batches=False` also lets each process shard the stream into contiguous slices, as the sampler does.
        # See https://github.com/huggingface/trl/issues/3213
        uses_iterable_dataset = (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        )
        if uses_iterable_dataset:
            if args.accelerator_config.dispatch_batches:
                raise ValueError(
                    "Iterable datasets require `dispatch_batches=False`, but it is set to `True` in "
                    "`accelerator_config`. Please set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        # An iterable train set bakes the generation-batch repeats into the stream, so it must be read by a single
        # worker: multiple workers would shard and interleave it, breaking the generation-batch ordering that
        # `_prepare_inputs` relies on. Map-style train keeps its workers.
        if isinstance(train_dataset, IterableDataset) and args.dataloader_num_workers != 0:
            logger.warning(
                f"Iterable datasets require `dataloader_num_workers=0` to preserve prompt grouping; overriding the "
                f"provided value ({args.dataloader_num_workers})."
            )
            args.dataloader_num_workers = 0

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in distillation
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. Here, loss scaling instead depends on the total number of completion tokens across the global
            # accumulated batch. To control scaling ourselves, we must disable Trainer's built-in scaling. The simplest
            # (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses that behavior
            # without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        self._dist = DistributedBackend(self.accelerator)

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        # Prepare teacher model (after super().__init__ so accelerator is ready)
        if teacher_model is not None:
            # The divergence compares the full next-token distribution of the student against the teacher's, so both
            # must be defined over the same vocabulary.
            student_vocab_size = self.model.config.get_text_config().vocab_size
            teacher_vocab_size = teacher_model.config.get_text_config().vocab_size
            if student_vocab_size != teacher_vocab_size:
                raise ValueError(
                    f"The student model has vocab_size {student_vocab_size} but the teacher model has vocab_size "
                    f"{teacher_vocab_size}. Distillation compares the teacher's full next-token distribution, which "
                    f"requires a shared vocabulary. Use a teacher with the same vocab_size, or GOLD for "
                    f"cross-tokenizer distillation."
                )
            # The Liger fused JSD kernel projects `h @ Wᵀ` directly and has no `logit_scale` /
            # `final_logit_softcapping` parameters, so (unlike the chunked path) it cannot reproduce Cohere
            # `logit_scale` or Gemma `final_logit_softcapping`. Refuse rather than silently optimize a different
            # objective than the model's real forward.
            if self.use_liger_kernel:
                for name, model in [("student", self.model), ("teacher", teacher_model)]:
                    # On VLMs the logit post-processing lives on `text_config`, so read it through
                    # `get_text_config()`. Muse Glimmer names its pre-softcap multiplier `output_multiplier`.
                    config = model.config.get_text_config()
                    logit_scale = getattr(config, "logit_scale", None)
                    if logit_scale is None:
                        logit_scale = getattr(config, "output_multiplier", None)
                    scaled = logit_scale not in (None, 1.0)
                    softcapped = getattr(config, "final_logit_softcapping", None) is not None
                    if scaled or softcapped:
                        raise ValueError(
                            f"`use_liger_kernel=True` is incompatible with the {name} model's `logit_scale` / "
                            f"`final_logit_softcapping` (e.g. Cohere / Gemma models): the Liger fused JSD loss reads "
                            f"`lm_head.weight` directly and cannot apply them, so it would optimize a different "
                            f"objective than the model's real forward. Set `use_liger_kernel=False` to use the chunked "
                            f"loss, which applies both."
                        )
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
            else:
                self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        else:
            self.teacher_model = None

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        # Store config values
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.max_completion_length = args.max_completion_length
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.shuffle_dataset = args.shuffle_dataset

        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Ensure each process receives a unique seed so different processes generate different completions when
        # generating with transformers. We could skip it if we use vLLM, but it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Generation config
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "cache_implementation": args.cache_implementation,
        }
        if args.generation_kwargs is not None:
            generation_kwargs.update(args.generation_kwargs)
        self.generation_config = GenerationConfig(**generation_kwargs, disable_compile=True)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs

        # Metrics & Logging
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        generation_batch_size = (
            args.per_device_train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        )
        self._logs = {
            "prompt": deque(maxlen=generation_batch_size),
            "completion": deque(maxlen=generation_batch_size),
        }

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)

        # vLLM for student generation
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.gradient_accumulation_steps,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                trust_remote_code=args.trust_remote_code,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=None,  # distillation trains on the teacher distribution, not sampled logprobs
                generation_kwargs=args.generation_kwargs,
            )
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In DistillationTrainer, we preprocess data, so using the model's signature columns
        # doesn't work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate
    # completions once every gradient_accumulation_steps step—rather than once per accumulation step—which is
    # significantly more efficient. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: this method is a copy-paste of the original `Trainer.get_train_dataloader` with two changes: the
    # batch size is multiplied by `gradient_accumulation_steps`, and iterable datasets are wrapped (see
    # `repeat_iterable_dataset`).
    def get_train_dataloader(self):
        dataset = self.train_dataset
        if isinstance(dataset, IterableDataset):
            # Iterable datasets can't be indexed, so RepeatSampler can't be attached. Reproduce its ordering by
            # transforming the stream instead (see `repeat_iterable_dataset`). The full permutation done by
            # RepeatSampler becomes a buffered shuffle here.
            if self.shuffle_dataset:
                dataset = dataset.shuffle(seed=self.args.seed)
            # Effective training batch = the generation batch (the deferred `generation_batch_size` slot-in).
            generation_batch_size = (
                self.args.per_device_train_batch_size
                * self.accelerator.num_processes
                * self.args.gradient_accumulation_steps
            )
            dataset = repeat_iterable_dataset(
                dataset,
                mini_repeat_count=1,
                batch_size=generation_batch_size,
                repeat_count=self.args.gradient_accumulation_steps,
            )
        return self._get_dataloader(
            dataset=dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Repeat each generation batch `gradient_accumulation_steps` times so the completions generated once per
        # generation batch (see `_prepare_inputs`) are reused across the accumulation window. Distillation is n=1,
        # so there is no per-prompt repeat (`mini_repeat_count=1`).
        if dataset is None:
            dataset = self.train_dataset
        # The generation batch is the full effective training batch. GRPO names it `generation_batch_size` (paired with
        # the deferred `steps_per_generation`); both are deferred here, so derive it inline — slot-in point.
        generation_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=1,
            batch_size=generation_batch_size,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )

    # This method overrides `Trainer.get_eval_dataloader` to wrap iterable eval datasets, reproducing the
    # RepeatSampler ordering that can't be attached to them (see `get_train_dataloader`). Map-style datasets keep the
    # default path via `_get_eval_sampler`, which shuffles with `seed`, so the iterable wrap shuffles too (buffered)
    # to walk prompts in a matching order.
    # Maintenance note: this method is a copy-paste of the original `Trainer.get_eval_dataloader`, with the iterable
    # wrapping as the only addition.
    def get_eval_dataloader(self, eval_dataset: str | Dataset | IterableDataset | None = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        if isinstance(eval_dataset, IterableDataset):
            # Apply the `__init__` iterable config here too
            if self.args.accelerator_config.dispatch_batches:
                raise ValueError(
                    "Iterable datasets require `dispatch_batches=False`, but it is set to `True` in "
                    "`accelerator_config`. Please set it to `False`."
                )
            self.accelerator.dataloader_config.dispatch_batches = False
            eval_dataset = eval_dataset.shuffle(seed=self.args.seed)
            eval_dataset = repeat_iterable_dataset(eval_dataset, mini_repeat_count=1)
            # Force a single worker for this loader only, without persisting the change
            num_workers = self.args.dataloader_num_workers
            self.args.dataloader_num_workers = 0

        try:
            return self._get_dataloader(
                dataset=eval_dataset,
                description="Evaluation",
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler,
                dataloader_key=dataloader_key,
            )
        finally:
            if isinstance(eval_dataset, IterableDataset):
                self.args.dataloader_num_workers = num_workers

    def _tokenize_prompts(self, prompts: list):
        """Tokenize prompts and extract images/multimodal fields for generation."""
        if is_conversational({"prompt": prompts[0]}):
            # Normalize string content to content blocks for VLM processors that don't handle plain strings.
            if self._is_vlm:
                prompts = [prepare_multimodal_messages(prompt) for prompt in prompts]

            # Extract images from messages for VLM support
            images = []
            has_images = False
            for prompt in prompts:
                prompt_images = []
                for message in prompt:
                    if isinstance(message["content"], list):
                        for part in message["content"]:
                            if part["type"] == "image":
                                prompt_images.append(part["image"])
                                has_images = True
                images.append(prompt_images if prompt_images else None)
            images = images if has_images else None

            # Workaround for a bug in transformers 5.3.0 where some processors (e.g. Qwen2.5-VL) crash on
            # batched unpadded input (transformers#44514).
            # Fixed in transformers 5.4.0 (transformers#44563).
            needs_padding_workaround = Version("5.3.0") <= Version(transformers.__version__) < Version("5.4.0")
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **({"padding": True} if needs_padding_workaround else {}),
                **self.chat_template_kwargs,
            )
            if needs_padding_workaround:
                # Unpad input_ids: remove padding tokens using attention_mask to get per-sequence lists
                prompt_ids = [
                    [tok for tok, m in zip(ids, mask, strict=True) if m]
                    for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"], strict=True)
                ]
            else:
                prompt_ids = tokenized["input_ids"]
            # For VLMs, the processor returns extra multimodal fields (pixel_values, image_grid_thw, etc.)
            multimodal_fields = {k: v for k, v in tokenized.items() if k not in ("input_ids", "attention_mask")}
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
            images = None
            multimodal_fields = {}
        return prompt_ids, images, multimodal_fields

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        device = self.accelerator.device

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # Sync weights if training step changed
            if self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs. Distillation is n=1 and uses the teacher distribution rather than
            # sampled logprobs, so we request one completion per prompt and discard vLLM's logprobs.
            _, completion_ids, _, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=1,
                profiler=profiling_context(self, "vLLM.generate"),
            )

        else:
            # Regular generation path: left-pad token IDs into tensors
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left")
            attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left")
            generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
            # For VLMs, include multimodal fields as tensors (pixel_values, image_grid_thw, etc.)
            for k, v in multimodal_fields.items():
                if isinstance(v, torch.Tensor):
                    generate_inputs[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], list):
                    # Per-token field (e.g., token_type_ids): left-pad like input_ids
                    generate_inputs[k] = pad([torch.tensor(x) for x in v], padding_value=0, padding_side="left")
                else:
                    generate_inputs[k] = torch.tensor(np.array(v))
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self._tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
            ]

        return completion_ids

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
        completion_ids = self._generate_single_turn(prompt_ids, images, multimodal_fields)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        # Fail clearly if the generation backend returned no completions (avoids a cryptic min() error below).
        if agg_completion_lengths.numel() == 0:
            raise RuntimeError(
                "No completions were generated. This usually means the generation backend failed to return any "
                "results; see the generation logs above for the underlying error."
            )
        total_prompt_tokens = agg_prompt_lengths.sum()

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + agg_completion_lengths.sum()).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LFM2-VL
        if spatial_shapes is not None:
            model_inputs["spatial_shapes"] = spatial_shapes
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        if image_position_ids is not None:
            model_inputs["image_position_ids"] = image_position_ids

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        # `base_model` gives the backbone model (skipping `lm_head`) — text decoder for LMs, multimodal wrapper for
        # VLMs (so vision-token injection runs before the text decoder). `get_decoder()` won't do: on VLMs it
        # returns just the text stack and feeds image-placeholder IDs through it.
        # Pre-5.0 transformers VLMs set `base_model_prefix = ""` so `base_model is self` (re-runs `lm_head`).
        # Fall back to `.model` there.
        if self._is_vlm and Version(transformers.__version__) < Version("5.0.0"):
            backbone = unwrapped_model.model
        else:
            backbone = unwrapped_model.base_model
        last_hidden_state = backbone(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Name kept aligned with GRPO/RLOO for consistency; distillation has no rewards, so nothing is actually scored.
    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, Any]:
        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, images=image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        prompt_ids_list, completion_ids_list = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)

        num_items_in_batch = self.accelerator.gather(completion_mask.sum()).sum()

        num_images = [len(img_list) if img_list else 0 for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs.
        if images is not None:
            prompts_text = [
                apply_chat_template({"prompt": prompt}, self.processing_class, **self.chat_template_kwargs)["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Recover LFM2-VL tile counts; the full processor drops row/column metadata.
        num_tiles = None
        if images is not None and "spatial_shapes" in forward_kwargs:
            image_info = self.processing_class.image_processor(
                images=images, return_tensors="pt", return_row_col_info=True
            )
            tiles_per_image = image_info["image_rows"] * image_info["image_cols"]
            if self.processing_class.image_processor.use_thumbnail:
                tiles_per_image = tiles_per_image + (tiles_per_image > 1).to(tiles_per_image.dtype)
            num_tiles = [group.sum().item() for group in torch.split(tiles_per_image, num_images)]
        # Same for InternVL, whose pixel_values is tile-indexed ([total_tiles, channels, height, width]).
        elif (
            images is not None
            and forward_kwargs["pixel_values"].ndim == 4
            and forward_kwargs["pixel_values"].size(0) != sum(num_images)
        ):
            num_patches = self.processing_class.image_processor(
                images=images, crop_to_patches=True, return_tensors="pt"
            )["num_patches"]
            num_tiles = [group.sum().item() for group in torch.split(num_patches, num_images)]

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - token_type_ids.size(1)
                if padding_size > 0:
                    token_type_ids = torch.cat(
                        [token_type_ids.new_zeros((token_type_ids.size(0), padding_size)), token_type_ids], dim=1
                    )
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        # If mm_token_type_ids are used, extend them with zeros for the completion part
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and mm_token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - mm_token_type_ids.size(1)
                if padding_size > 0:
                    mm_token_type_ids = torch.cat(
                        [mm_token_type_ids.new_zeros((mm_token_type_ids.size(0), padding_size)), mm_token_type_ids],
                        dim=1,
                    )
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # Log the prompt and completion texts
        if self.log_completions:
            prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            self._logs["prompt"].extend(gather_object(prompts_text))
            self._logs["completion"].extend(gather_object(completions_text))

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "num_items_in_batch": num_items_in_batch,
        }
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "spatial_shapes" in forward_kwargs:
            output["spatial_shapes"] = forward_kwargs["spatial_shapes"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if "image_position_ids" in forward_kwargs:
            output["image_position_ids"] = forward_kwargs["image_position_ids"]
        if images is not None:
            output["num_images"] = num_images
            if num_tiles is not None:
                output["num_tiles"] = num_tiles
        return output

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × gradient accumulation steps)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every gradient_accumulation_steps)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.gradient_accumulation_steps
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.gradient_accumulation_steps)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # transformers computes `num_items_in_batch` from the raw dataloader labels, before on-policy generation
        # replaces the completions; use the count over the generated completions instead (computed in
        # `_generate_and_score_completions`). Divide by the process count so the per-process loss compensates for DDP
        # gradient averaging.
        if self.model.training and inputs.get("num_items_in_batch") is not None:
            num_items_in_batch = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes

        # Route the whole loss (backbone + `lm_head` projection + JSD) through the DDP/FSDP wrapper via
        # `_forward_redirection`, so DDP.forward() fires `prepare_for_backward()` and FSDP/DeepSpeed keep the student's
        # sharded parameters (including the `lm_head`) materialized for the projection.
        unwrapped_student = self.accelerator.unwrap_model(model)
        loss, entropy_sum, num_valid_tokens = self._forward_redirection(
            model, unwrapped_student, self._compute_loss, unwrapped_student, inputs, num_items_in_batch
        )

        # Log the mean per-token student entropy (in nats). The reduction runs here, after `_forward_redirection`
        # returns, so the `gather_for_metrics` collective does not run inside the DDP/FSDP-wrapped forward (a hang/
        # ordering risk). The Liger path produces no entropy, so it logs none. Mirrors `SFTTrainer.compute_loss`.
        if entropy_sum is not None:
            mode = "train" if self.model.training else "eval"
            num_valid_tokens = self.accelerator.gather_for_metrics(num_valid_tokens).sum()
            entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
            entropy = (entropy_sum / num_valid_tokens).item() if num_valid_tokens > 0 else 0.0
            self._metrics[mode]["entropy"].append(entropy)

        return (loss, None) if return_outputs else loss

    def _compute_loss(self, unwrapped_student, inputs, num_items_in_batch):
        # Chunked JSD path: project the teacher/student hidden states to vocab logits one chunk at a time (never
        # materializing the full `(B, C, V)` logits), so the teacher's dense distribution can be matched without
        # buffering it. Runs inside the student wrapper's forward (see `compute_loss`).
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        completion_mask = inputs["completion_mask"]
        logits_to_keep = inputs["completion_ids"].size(1)  # only the completion tokens are trained on

        # Multimodal (VLM) fields, extracted during generation and split to this micro-batch by `_prepare_inputs`.
        multimodal_keys = (
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "spatial_shapes",
            "image_sizes",
            "token_type_ids",
            "mm_token_type_ids",
            "image_position_ids",
        )
        multimodal_inputs = {k: inputs[k] for k in multimodal_keys if k in inputs}

        student_hidden_states = self._get_last_hidden_state(
            unwrapped_student, input_ids, attention_mask, logits_to_keep, **multimodal_inputs
        )

        # Route the teacher backbone through its own wrapper via `_forward_redirection` too, so FSDP/DeepSpeed
        # materialize its sharded parameters before the forward runs (the backbone call would otherwise see shards).
        self.teacher_model.eval()
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        with torch.no_grad():
            teacher_hidden_states = self._forward_redirection(
                self.teacher_model,
                unwrapped_teacher,
                self._get_last_hidden_state,
                unwrapped_teacher,
                input_ids,
                attention_mask,
                logits_to_keep,
                **multimodal_inputs,
            )

        student_lm_head = unwrapped_student.get_output_embeddings()
        teacher_lm_head = unwrapped_teacher.get_output_embeddings()

        if self.use_liger_kernel:
            # Fused JSD over the same hidden states as the chunked path. `true_labels` only masks positions (the
            # hard-loss weight is 0), so any non-ignore id marks a valid completion token; `_get_last_hidden_state`
            # already returns the completion-aligned positions, so no shift is needed.
            true_labels = torch.where(
                completion_mask.bool(), inputs["completion_ids"], torch.full_like(inputs["completion_ids"], -100)
            ).reshape(-1)
            # Under FSDP2 the heads are DTensors; materialize them once with full_tensor() for the fused kernel, as the
            # chunked path does. No-op off FSDP2; the ZeRO-3 (non-DTensor) case is handled by maybe_gather_lm_head_ctx.
            student_weight, student_bias = student_lm_head.weight, student_lm_head.bias
            teacher_weight, teacher_bias = teacher_lm_head.weight, teacher_lm_head.bias
            if isinstance(student_weight, torch.distributed.tensor.DTensor):
                student_weight = student_weight.full_tensor()
                if student_bias is not None:
                    student_bias = student_bias.full_tensor()
            if isinstance(teacher_weight, torch.distributed.tensor.DTensor):
                teacher_weight = teacher_weight.full_tensor()
                if teacher_bias is not None:
                    teacher_bias = teacher_bias.full_tensor()
            # ZeRO-3 shards the heads and the fused kernel reads them directly, so gather them for the call.
            with maybe_gather_lm_head_ctx(
                student_lm_head.weight, student_lm_head.bias, teacher_lm_head.weight, teacher_lm_head.bias
            ):
                loss = self.liger_loss(
                    student_input=student_hidden_states.reshape(-1, student_hidden_states.size(-1)),
                    student_weight=student_weight,
                    teacher_input=teacher_hidden_states.reshape(-1, teacher_hidden_states.size(-1)),
                    teacher_weight=teacher_weight,
                    true_labels=true_labels,
                    student_bias=student_bias,
                    teacher_bias=teacher_bias,
                )
            # Liger normalizes by the local valid-token count; rescale to the global count for grad-accum correctness.
            if num_items_in_batch is not None:
                num_valid_local = (true_labels != -100).sum().clamp_min(1)
                if isinstance(num_items_in_batch, torch.Tensor):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss * num_valid_local / num_items_in_batch
            # The fused kernel produces no entropy; `compute_loss` logs none for the Liger path.
            return loss, None, None

        # On VLMs the logit post-processing lives on `text_config`, so read it through `get_text_config()`.
        student_config = unwrapped_student.config.get_text_config()
        teacher_config = unwrapped_teacher.config.get_text_config()
        # `logit_scale` is None on models that don't scale (e.g. MPT); read that as unscaled (1.0). A real 0.0 is kept
        # as-is: the Liger guard rejects it, and the chunked path applies it faithfully. Muse Glimmer applies the same
        # pre-softcap multiplier under the name `output_multiplier`.
        student_logit_scale = getattr(student_config, "logit_scale", None)
        if student_logit_scale is None:
            student_logit_scale = getattr(student_config, "output_multiplier", None)
        teacher_logit_scale = getattr(teacher_config, "logit_scale", None)
        if teacher_logit_scale is None:
            teacher_logit_scale = getattr(teacher_config, "output_multiplier", None)
        student_logit_scale = 1.0 if student_logit_scale is None else student_logit_scale
        teacher_logit_scale = 1.0 if teacher_logit_scale is None else teacher_logit_scale
        loss, entropy_sum, n_valid = _chunked_divergence_loss(
            student_hidden_states,
            teacher_hidden_states,
            student_lm_head.weight,
            teacher_lm_head.weight,
            completion_mask,
            self.beta,
            _CHUNKED_LM_HEAD_CHUNK_SIZE,
            num_items_in_batch=num_items_in_batch,
            student_lm_head_bias=student_lm_head.bias,
            teacher_lm_head_bias=teacher_lm_head.bias,
            student_logit_scale=student_logit_scale,
            teacher_logit_scale=teacher_logit_scale,
            student_final_logit_softcapping=getattr(student_config, "final_logit_softcapping", None),
            teacher_final_logit_softcapping=getattr(teacher_config, "final_logit_softcapping", None),
            temperature=self.temperature,
        )
        # Return the raw entropy sum and valid-token count for `compute_loss` to aggregate and log after the forward
        # returns (see there). Detached: the metric is gradient-free.
        return loss, entropy_sum.detach(), n_valid

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        # Average the metrics
        metrics = {}
        for key, val in self._metrics[mode].items():
            # Filter out NaN values before averaging. With logging_steps > 1, a naive sum()/len() would let a single
            # NaN contaminate valid data from other batches. Only return None when no valid values remain (e.g. JSON
            # loggers crash on float NaN).
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    {},
                    None,
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            import pandas as pd

            table = {
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
            }
            df_base = pd.DataFrame(table)
            df_base.to_parquet(
                os.path.join(
                    self.args.output_dir,
                    "completions",
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            for logging_backend in logging_backends:
                df = df_base
                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
