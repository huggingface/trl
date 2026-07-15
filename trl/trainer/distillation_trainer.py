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
import importlib.resources as pkg_resources
import inspect
import math
import os
import textwrap
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather_object, is_peft_model, set_seed
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from huggingface_hub import CommitScheduler, DatasetCard, DatasetCardData, create_repo
from packaging.version import Version
from torch.utils.data import Sampler
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
from transformers.utils import is_peft_available, is_rich_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from ..distributed import DistributedBackend
from ..extras.profiling import profiling_context, profiling_decorator
from ..generation.vllm_generation import VLLMGeneration
from ..import_utils import is_liger_kernel_available
from ..models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from ..models.utils import _ForwardRedirection, disable_gradient_checkpointing
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
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)


if is_liger_kernel_available():
    pass


if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model


if is_trackio_available():
    import trackio


if is_wandb_available():
    import wandb


logger = get_logger(__name__)

# Number of valid tokens whose logits are materialized at a time. Peak memory scales linearly with it.
_CHUNKED_LM_HEAD_CHUNK_SIZE = 256


def _chunk(h_s, w_s, b_s, h_t, w_t, b_t, valid, beta):
    """Project one chunk of hidden states to vocabulary and reduce it to a scalar divergence.

    Both projections happen here, inside the checkpointed body, so that only the `(C, H)` hidden states are saved for
    backward. Computing the teacher's logits outside would keep a `(C, V)` tensor alive per chunk and defeat the point.

    A chunk's tail may hold masked positions: `valid` zeroes their contribution to both the loss and the entropy.
    """
    with maybe_gather_lm_head_ctx(w_s, b_s):
        student_logits = h_s.float() @ w_s.float().t()
        if b_s is not None:
            student_logits = student_logits + b_s.float()
    with maybe_gather_lm_head_ctx(w_t, b_t):
        teacher_logits = h_t.float() @ w_t.float().t()
        if b_t is not None:
            teacher_logits = teacher_logits + b_t.float()

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is
    # swapped compared to that defined in the paper.
    if beta == 0.0:
        divergence = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        divergence = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        # Compute the log of the mixture distribution: log(a + b) = log(exp(log(a)) + exp(log(b)))
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]), dim=0
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        divergence = beta_t * kl_teacher + (1 - beta_t) * kl_student

    entropy = -(student_log_probs.exp() * student_log_probs).sum(-1)
    return (divergence.sum(-1) * valid).sum(), (entropy * valid).sum()


def _chunked_divergence_loss(
    student_hidden: torch.Tensor,
    student_lm_head_weight: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_lm_head_weight: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
    chunk_size: int,
    num_items_in_batch: torch.Tensor | int | None = None,
    student_lm_head_bias: torch.Tensor | None = None,
    teacher_lm_head_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient generalized Jensen-Shannon divergence between a student and a teacher.

    The full `lm_head` projections are never materialized. Valid (unmasked) tokens are packed to the front (via
    `argsort` on the mask, a static-shape op) and processed in chunks of `chunk_size`, rounding the count up to a whole
    chunk so masked positions land in a skippable tail. Each chunk's `[chunk_size, vocab_size]` logits are kept alive
    only during its own forward/backward via gradient checkpointing, so peak logits memory is
    `2 * chunk_size * vocab_size` instead of `2 * batch_size * seq_len * vocab_size`.

    Args:
        student_hidden (`torch.Tensor`):
            Student decoder output of shape `(B, K, H)`, i.e. before the `lm_head` projection, aligned with the
            completion tokens.
        student_lm_head_weight (`torch.Tensor`):
            Weight of the student's `lm_head` linear layer, shape `(V, H)`.
        teacher_hidden (`torch.Tensor`):
            Teacher decoder output of shape `(B, K, H_t)`, aligned with the completion tokens.
        teacher_lm_head_weight (`torch.Tensor`):
            Weight of the teacher's `lm_head` linear layer, shape `(V, H_t)`.
        mask (`torch.Tensor`):
            Completion mask of shape `(B, K)`. Positions equal to `0` are excluded from both the `lm_head` matmuls and
            the loss.
        beta (`float`):
            Interpolation coefficient in `[0.0, 1.0]`. `0.0` is the forward KL, `1.0` the reverse KL.
        chunk_size (`int`):
            Number of valid tokens processed per chunk. Peak memory scales linearly with this.
        num_items_in_batch (`torch.Tensor`, `int` or `None`, *optional*):
            Total number of valid tokens across the global batch. When provided, the loss is reduced as
            `sum / num_items_in_batch`, which is gradient-accumulation-correct. When `None`, reduction is `mean` over
            local valid tokens.
        student_lm_head_bias (`torch.Tensor`, *optional*):
            Bias of the student's `lm_head` linear layer, shape `(V,)`.
        teacher_lm_head_bias (`torch.Tensor`, *optional*):
            Bias of the teacher's `lm_head` linear layer, shape `(V,)`.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: scalar loss, sum of the student's per-token Shannon entropy
        (in nats), and the number of valid target tokens — the last two as raw local sums, so callers can reduce them
        correctly across ranks.
    """
    hidden_s = student_hidden.reshape(-1, student_hidden.size(-1))
    hidden_t = teacher_hidden.reshape(-1, teacher_hidden.size(-1))
    flat_mask = mask.reshape(-1).bool()
    n_valid = flat_mask.sum()

    entropy_sum = hidden_s.new_zeros((), dtype=torch.float32)
    if n_valid == 0:
        # Whole micro-batch masked (e.g. every completion truncated). Keep the loss connected to the autograd graph
        # through every trainable parameter so `.backward()` succeeds and DDP / FSDP gradient sync doesn't hang.
        with maybe_gather_lm_head_ctx(student_lm_head_weight, student_lm_head_bias):
            loss = (student_hidden.float().sum() + student_lm_head_weight.float().sum()) * 0.0
            if student_lm_head_bias is not None:
                loss = loss + student_lm_head_bias.float().sum() * 0.0
        return loss, entropy_sum, n_valid

    # Pack valid tokens to the front so masked positions form whole trailing chunks. `argsort` on the boolean mask is
    # a static-shape op (unlike `hidden[mask]`, whose output shape is data-dependent and poisons XLA compilation).
    order = flat_mask.to(torch.int8).argsort(descending=True, stable=True)
    hidden_s = hidden_s[order]
    hidden_t = hidden_t[order]
    valid_sorted = flat_mask[order]

    # Process only the whole chunks covering the valid prefix: bounds XLA recompiles and drops fully-masked chunks.
    n_padded = (n_valid / chunk_size).ceil().to(torch.int64) * chunk_size

    loss = hidden_s.new_zeros((), dtype=torch.float32)
    for start in range(0, n_padded, chunk_size):
        chunk_loss, chunk_entropy = torch.utils.checkpoint.checkpoint(
            _chunk,
            hidden_s[start : start + chunk_size],
            student_lm_head_weight,
            student_lm_head_bias,
            hidden_t[start : start + chunk_size],
            teacher_lm_head_weight,
            teacher_lm_head_bias,
            valid_sorted[start : start + chunk_size],
            beta,
            use_reentrant=False,
        )
        loss = loss + chunk_loss
        entropy_sum = entropy_sum + chunk_entropy

    if num_items_in_batch is None:
        loss = loss / n_valid
    else:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss, entropy_sum, n_valid


class DistillationTrainer(_BaseTrainer):
    """
    Trainer for on-policy knowledge distillation.

    The student generates completions from its own policy, the teacher scores them, and the student is trained to
    minimize a per-token divergence against the teacher's next-token distribution. Training on the student's own
    samples is what distinguishes this from supervised fine-tuning on teacher-generated text: the student is corrected
    on the states it actually visits at inference time. This is the approach introduced in [On-Policy Distillation of
    Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649).

    The divergence is selected with `beta`: `0.0` is the forward KL, `1.0` (the default) is the reverse KL, and
    intermediate values give the generalized Jensen-Shannon divergence.

    Example:

    ```python
    >>> from datasets import load_dataset
    >>> from trl import DistillationTrainer

    >>> dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    >>> trainer = DistillationTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
    ...     teacher_model="Qwen/Qwen2.5-7B-Instruct",
    ...     train_dataset=dataset,
    ... )
    >>> trainer.train()
    ```

    Args:
        model (`str` or [`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Student model to be trained. Can be either:

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
        teacher_model (`str` or [`~transformers.PreTrainedModel`]):
            Teacher model whose next-token distribution the student is trained to match. Can be either a *model id* (or
            path), loaded with the keyword arguments in `args.teacher_model_init_kwargs`, or an already instantiated
            [`~transformers.PreTrainedModel`]. It must share the student's vocabulary; for cross-tokenizer
            distillation, use [`experimental.gold.GOLDTrainer`] instead.
        args ([`DistillationConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
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
        teacher_model: "str | PreTrainedModel | None" = None,
        args: DistillationConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DistillationConfig(f"{model_name}-Distillation")

        # Model
        if isinstance(model, str):
            model_init_kwargs = dict(args.model_init_kwargs or {})  # copy to avoid mutating model_init_kwargs
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
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistillationConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
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
                get_config_model_id(model.config),
                truncation_side="left",
                padding_side="left",
                trust_remote_code=args.trust_remote_code,
            )

        if args.use_transformers_continuous_batching and isinstance(processing_class, ProcessorMixin):
            raise ValueError(
                "`use_transformers_continuous_batching` does not support multimodal models. Use `use_vllm` instead."
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

        # Resolve vision placeholder token IDs once. Used by the forward pass to rebuild mm_token_type_ids
        # when tool responses inject images into the completion (see _generate forward_kwargs block).
        self._image_pad_token_id = None
        self._video_pad_token_id = None
        if self._is_vlm:
            for candidate in ("<|image_pad|>", "<|image|>"):
                tid = self._tokenizer.convert_tokens_to_ids(candidate)
                if tid != self._tokenizer.unk_token_id:
                    self._image_pad_token_id = tid
                    break
            tid = self._tokenizer.convert_tokens_to_ids("<|video_pad|>")
            if tid != self._tokenizer.unk_token_id:
                self._video_pad_token_id = tid

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

        self.chat_template = None

        # Training arguments
        self.max_completion_length = args.max_completion_length
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_continuous_batching = args.use_transformers_continuous_batching
        if self.use_transformers_continuous_batching:
            if not Version(transformers.__version__) >= Version("5.8.0"):
                raise ImportError(
                    "Using `use_transformers_continuous_batching` requires transformers>=5.8.0. "
                    "Please upgrade with `pip install --upgrade transformers`."
                )
            from transformers.generation import ContinuousBatchingConfig

            cb_kwargs = dict(args.transformers_continuous_batching_config or {})
            # The transformers default (0.9) leaves almost no VRAM for the training backward pass;
            # use a training-aware default unless the user has set it explicitly.
            cb_kwargs.setdefault("max_memory_percent", 0.5)
            self.continuous_batching_config = ContinuousBatchingConfig(**cb_kwargs)
        else:
            self.continuous_batching_config = None
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.beta = args.beta

        # MoE load-balancing auxiliary loss, applied to Mixture-of-Experts models (no effect otherwise)
        text_config = model.config.get_text_config()
        is_moe = getattr(text_config, "output_router_logits", None) is not None
        self.aux_loss_enabled = is_moe and args.router_aux_loss_coef != 0.0
        self.router_aux_loss_coef = args.router_aux_loss_coef

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise ValueError(
                "Iterable datasets are not yet supported in DistillationTrainer. Please use a standard dataset instead."
            )

        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        self._buffered_inputs = None

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed here
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Teacher model
        if teacher_model is None:
            raise ValueError("`teacher_model` is required")
        if isinstance(teacher_model, str):
            teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs or {})
            # Distributed training requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                teacher_model_init_kwargs["device_map"] = None
            teacher_model_init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            teacher_model = create_model_from_path(teacher_model, **teacher_model_init_kwargs)
        elif args.teacher_model_init_kwargs is not None:
            raise ValueError(
                "You passed `teacher_model_init_kwargs` to the `DistillationConfig`, but your `teacher_model` is "
                "already instantiated. This argument can only be used when `teacher_model` is a model identifier."
            )
        self.teacher_model = teacher_model

        # The divergence compares the full next-token distribution of the student against the teacher's, so both must
        # be defined over the same vocabulary.
        student_vocab_size = self.model.config.get_text_config().vocab_size
        teacher_vocab_size = self.teacher_model.config.get_text_config().vocab_size
        if student_vocab_size != teacher_vocab_size:
            raise ValueError(
                f"The student model has vocab_size {student_vocab_size} but the teacher model has vocab_size "
                f"{teacher_vocab_size}. Distillation compares the teacher's full next-token distribution, which "
                f"requires a shared vocabulary. Use a teacher with the same vocab_size, or GOLD for cross-tokenizer "
                f"distillation."
            )

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            disable_dropout_in_model(self.teacher_model)

        # The chunked divergence calls the backbone and the lm_head directly rather than going through the wrapper's
        # forward. Redirect model.module forward to the model forward so pre-forward hooks still fire: DDP needs it to
        # arm `prepare_for_backward()`, and under ZeRO-3 the parameter coordinator gathers/reduces around the loss.
        self._forward_redirection = _ForwardRedirection()

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_multimodal = args.log_multimodal
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers. We could skip it if we use vLLM, but it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # Initialize vLLM generation backend
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
                * args.steps_per_generation,
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
                logprobs=0,  # we only need the generated token logprobs for the importance sampling correction
                generation_kwargs=args.generation_kwargs,
            )
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        else:
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

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        self._dist = DistributedBackend(self.accelerator)

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            self.teacher_model = prepare_fsdp(self.teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(repo_id, private=self.args.hub_private_repo, repo_type="dataset", exist_ok=True)
                template_path = pkg_resources.files("trl").joinpath("templates/completions_dataset_card.md")
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). Here we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.
    def get_train_dataloader(self):
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that repeats the batch multiple times to allow reusing generations across multiple updates.
        # Refer to _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. Each row shows the per-step batch returned by
        # `_prepare_inputs`; rows within a `steps_per_generation` block are slices of the same generated batch.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step   <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   1   2   3   4   5   <- Generate for the first `steps_per_generation` (prompts 0 to 23); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     6   7   8   9  10  11   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2    12  13  14  15  16  17   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3    18  19  20  21  22  23   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    24  25  26  27  28  29   <- Generate for the second `steps_per_generation` (prompts 24 to 47); store the completions; use the first slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=1,
            batch_size=self.args.generation_batch_size,
            repeat_count=self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(data_source=eval_dataset, mini_repeat_count=1, seed=self.args.seed)

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

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            if self._step % self.args.steps_per_generation == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is no batch grouping for generation, hence
            # local generation batch == local eval batch
            inputs = self._generate_completions(generation_batch)
        return inputs

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
                chat_template=self.chat_template,
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

            # Generate using vLLM with raw token IDs
            _, completion_ids, _, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=1,
                profiler=profiling_context(self, "vLLM.generate"),
            )

        elif self.use_transformers_continuous_batching:
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                all_outputs = unwrapped_model.generate_batch(
                    prompt_ids,
                    generation_config=self.generation_config,
                    continuous_batching_config=self.continuous_batching_config,
                    progress_bar=False,
                )
                unwrapped_model.train()
            completion_ids = [output.generated_tokens for output in all_outputs.values()]

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

        # Decode completions, used for logging only
        if is_conversational({"prompt": prompts[0]}):
            contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, normalizes the loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
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

        return prompt_ids, completion_ids, completions, total_completion_tokens, images

    def _generate_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, torch.Tensor | Any]:
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

        prompt_ids_list, completion_ids_list, completions, num_items_in_batch, images = self._generate(prompts)

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

        num_images = [len(img_list) if img_list else 0 for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs. Use the full processor pipeline, which returns
        # model-specific keys (image_sizes, pixel_attention_mask, etc.).
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
        prompts_text = gather_object(
            [maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]
        )
        completions_text = gather_object([c if isinstance(c, str) else c[0]["content"] for c in completions])
        self._logs["prompt"].extend(prompts_text)
        self._logs["completion"].extend(completions_text)
        if images is not None and self.log_multimodal:
            self._logs["images"].extend(gather_object(images))

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistillationTrainer does not support returning outputs")
        unwrapped_model = self.accelerator.unwrap_model(model)
        # `_compute_loss` calls the backbone and the lm_head directly instead of the wrapper's forward, so it must be
        # routed through `_forward_redirection` for DDP to arm `prepare_for_backward()` and for ZeRO-3 to gather.
        return self._forward_redirection(model, unwrapped_model, self._compute_loss, unwrapped_model, inputs)

    def _compute_loss(self, unwrapped_model, inputs):
        mode = "train" if self.model.training else "eval"

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        forward_kwargs = {
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "pixel_attention_mask": inputs.get("pixel_attention_mask"),
            "spatial_shapes": inputs.get("spatial_shapes"),
            "image_sizes": inputs.get("image_sizes"),
            "image_position_ids": inputs.get("image_position_ids"),
        }

        # Only the hidden states are kept for both models: the vocab projection is done chunk by chunk inside
        # `_chunked_divergence_loss`, so the (B, K, V) logits are never materialized in full.
        student_hidden = self._get_last_hidden_state(
            unwrapped_model, input_ids, attention_mask, logits_to_keep, **forward_kwargs
        )
        self.teacher_model.eval()
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        with (
            torch.no_grad(),
            disable_gradient_checkpointing(self.teacher_model, self.args.gradient_checkpointing_kwargs),
        ):
            teacher_hidden = self._get_last_hidden_state(
                unwrapped_teacher, input_ids, attention_mask, logits_to_keep, **forward_kwargs
            )

        student_head = unwrapped_model.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()

        loss, entropy_sum, num_valid = _chunked_divergence_loss(
            student_hidden=student_hidden,
            student_lm_head_weight=student_head.weight,
            teacher_hidden=teacher_hidden,
            teacher_lm_head_weight=teacher_head.weight,
            mask=completion_mask,
            beta=self.beta,
            chunk_size=_CHUNKED_LM_HEAD_CHUNK_SIZE,
            num_items_in_batch=inputs["num_items_in_batch"],
            student_lm_head_bias=student_head.bias,
            teacher_lm_head_bias=teacher_head.bias,
        )

        # Log the metrics
        total_entropy = self.accelerator.gather(entropy_sum).nansum()
        total_valid = self.accelerator.gather(num_valid).nansum()
        self._metrics[mode]["entropy"].append((total_entropy / total_valid.clamp(min=1.0)).item())
        return loss

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
            # Filter out NaN values before averaging. With logging_steps > 1, a naive sum()/len() would let a
            # single NaN contaminate valid data from other batches. Only return None when no valid values
            # remain (e.g. JSON loggers crash on float NaN).
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

            images_raw = self._logs["images"] or []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in self._logs["images"]:
                        if image_list:
                            images.append([logging_backend.Image(image) for image in image_list])
                        else:
                            images.append([])
                    df = pd.concat(
                        [df_base, pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
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
