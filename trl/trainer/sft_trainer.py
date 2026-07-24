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
import inspect
import json
import os
import types
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from packaging.version import Version
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..chat_template_utils import (
    clone_chat_template,
    get_training_chat_template,
    has_generation_markers,
    is_chat_template_stop_token_trained,
)
from ..data_utils import (
    _tokenize,
    apply_chat_template,
    get_dataset_column_names,
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    prepare_multimodal_messages,
)
from ..models import get_act_offloading_ctx_manager
from .base_trainer import _BaseTrainer
from .sft_config import SFTConfig
from .utils import (
    create_model_from_path,
    entropy_from_logits,
    flush_left,
    get_config_model_id,
    maybe_gather_lm_head_ctx,
    pad,
    selective_log_softmax,
)


if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, PeftType, get_peft_model


_CHUNKED_LM_HEAD_CHUNK_SIZE = 256


@dataclass
class _ChunkedCELMHeadOutput(CausalLMOutputWithPast):
    """`CausalLMOutputWithPast` with extra fields populated by the chunked-CE path."""

    num_correct_tokens: torch.Tensor | None = None
    entropy_sum: torch.Tensor | None = None
    num_valid_tokens: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None


def _chunk(h, w, b, lbl, logit_scale, final_logit_softcapping):
    with maybe_gather_lm_head_ctx(w, b):
        logits = h.float() @ w.float().t()
        if b is not None:
            logits = logits + b.float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if final_logit_softcapping is not None:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    log_p = F.log_softmax(logits, dim=-1)
    # A chunk's tail may be `-100` padding: `ignore_index` zeroes their loss; `valid` does the same for accuracy/entropy.
    chunk_loss = F.nll_loss(log_p, lbl, ignore_index=-100, reduction="sum")
    valid = lbl != -100
    chunk_correct = ((logits.argmax(dim=-1) == lbl) & valid).sum().float()
    chunk_entropy = (-(log_p.exp() * log_p).sum(dim=-1) * valid).sum()
    return chunk_loss, chunk_correct, chunk_entropy


def _chunked_cross_entropy_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    chunk_size: int,
    labels: torch.Tensor | None = None,
    shift_labels: torch.Tensor | None = None,
    num_items_in_batch: torch.Tensor | int | None = None,
    logit_scale: float = 1.0,
    final_logit_softcapping: float | None = None,
    lm_head_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient next-token cross-entropy over hidden states and an `lm_head` weight.

    The full `lm_head` projection is never materialized. Valid (non-`-100`) tokens are packed to the front (via
    `argsort` on the label mask, a static-shape op) and processed in chunks of `chunk_size`, rounding the count up to a
    whole chunk so masked positions land in a skippable tail. Each chunk's `[chunk_size, vocab_size]` logits are kept
    alive only during its own forward/backward via gradient checkpointing, so peak logits memory is `chunk_size *
    vocab_size` instead of `batch_size * seq_len * vocab_size`. Quantizing the chunk count to a multiple of
    `chunk_size` keeps this XLA/Neuron-safe (at most `total / chunk_size` distinct traced shapes, not one per
    valid-token count) while still dropping fully-masked chunks on GPU.

    At least one of `labels` or `shift_labels` must be provided. `labels` triggers the internal `labels[..., 1:]` /
    `hidden_states[..., :-1, :]` shift; `shift_labels` skips it, assuming the caller already aligned labels with hidden
    states (the contract under context / sequence parallelism). If both are given, `shift_labels` wins (matching
    [`~transformers.loss.ForCausalLMLoss`]).

    Args:
        hidden_states (`torch.Tensor`):
            Base decoder output of shape `(B, S, H)`, i.e. before the `lm_head` projection.
        lm_head_weight (`torch.Tensor`):
            Weight of the `lm_head` linear layer, shape `(V, H)`.
        chunk_size (`int`):
            Number of valid tokens processed per chunk. Peak memory scales linearly with this.
        labels (`torch.Tensor`, *optional*):
            Labels of shape `(B, S)`. Positions equal to `-100` are excluded from both the `lm_head` matmul and the
            loss. Mutually exclusive with `shift_labels`.
        shift_labels (`torch.Tensor`, *optional*):
            Pre-shifted labels of shape `(B, S)`, aligned with `hidden_states` (position `i` predicts
            `shift_labels[i]`). Mutually exclusive with `labels`.
        num_items_in_batch (`torch.Tensor`, `int` or `None`, *optional*):
            Total number of valid tokens across the global batch, as plumbed by [`~transformers.Trainer`]. When
            provided, the loss is reduced as `sum / num_items_in_batch`, matching the gradient-accumulation-correct
            behavior of HF's default cross-entropy. When `None`, reduction is `mean` over local valid tokens.
        logit_scale (`float`, *optional*, defaults to `1.0`):
            Multiplier applied to each chunk's logits before the cross-entropy, matching the `logit_scale` behavior of
            Cohere-style models.
        final_logit_softcapping (`float`, *optional*):
            If set, applies `softcap * tanh(logits / softcap)` to each chunk's logits before the cross-entropy,
            matching the `final_logit_softcapping` behavior of Gemma-style models. Applied after `logit_scale`.
        lm_head_bias (`torch.Tensor`, *optional*):
            Bias of the `lm_head` linear layer, shape `(V,)`. Added to each chunk's logits when provided.

    Returns:
        `tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`: scalar loss, number of correctly-predicted
        tokens (count), sum of per-token Shannon entropy (in nats), and number of valid (non-`-100`) target tokens —
        all over the local batch. Raw sums are returned so callers can reduce correctly across ranks.
    """
    if labels is None and shift_labels is None:
        raise ValueError("At least one of `labels` or `shift_labels` must be provided.")

    if shift_labels is not None:
        hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        labels = shift_labels.reshape(-1)
    else:
        hidden = hidden_states[..., :-1, :].reshape(-1, hidden_states.size(-1))
        labels = labels[..., 1:].reshape(-1)

    valid = labels != -100
    n_valid_tensor = valid.sum()

    correct = hidden.new_zeros((), dtype=torch.float32)
    entropy_sum = hidden.new_zeros((), dtype=torch.float32)
    if n_valid_tensor == 0:
        # Whole micro-batch masked (e.g. completion-only loss + truncation). Keep the loss connected
        # to the autograd graph through every trainable parameter so `.backward()` succeeds and DDP /
        # FSDP gradient sync doesn't hang on a missing param.
        with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
            loss = (hidden_states.float().sum() + lm_head_weight.float().sum()) * 0.0
            if lm_head_bias is not None:
                loss = loss + lm_head_bias.float().sum() * 0.0
        return loss, correct, entropy_sum, n_valid_tensor

    # Pack valid tokens to the front so masked positions form whole trailing chunks. `argsort` on the boolean mask is
    # a static-shape op (unlike `hidden[valid]`, whose output shape is data-dependent and poisons XLA compilation).
    order = valid.to(torch.int8).argsort(descending=True, stable=True)
    hidden = hidden[order]
    labels = labels[order]

    # Process only the whole chunks covering the valid prefix: bounds XLA recompiles and drops fully-masked chunks on GPU.
    n_padded = (n_valid_tensor / chunk_size).ceil().to(torch.int64) * chunk_size

    loss = hidden.new_zeros((), dtype=torch.float32)

    for start in range(0, n_padded, chunk_size):
        h_chunk = hidden[start : start + chunk_size]
        lbl_chunk = labels[start : start + chunk_size]
        chunk_loss, chunk_correct, chunk_entropy = torch.utils.checkpoint.checkpoint(
            _chunk,
            h_chunk,
            lm_head_weight,
            lm_head_bias,
            lbl_chunk,
            logit_scale,
            final_logit_softcapping,
            use_reentrant=False,
        )
        loss = loss + chunk_loss
        correct = correct + chunk_correct
        entropy_sum = entropy_sum + chunk_entropy

    if num_items_in_batch is None:
        loss = loss / n_valid_tensor
    else:
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss, correct, entropy_sum, n_valid_tensor


def _patch_chunked_ce_lm_head(model: torch.nn.Module, chunk_size: int, is_vlm: bool = False) -> None:
    """
    Patch `model.forward` to compute the LM loss via [`_chunked_cross_entropy_loss`].

    When `labels` (or pre-shifted `shift_labels`, for CP/SP) are provided, the patched forward runs the decoder up to
    `last_hidden_state` (skipping the `lm_head` matmul), drops `labels == -100` positions, and computes the
    cross-entropy in chunks of `chunk_size` valid tokens. Returns a [`_ChunkedCELMHeadOutput`] with `loss` set,
    `logits=None`, and `num_correct_tokens` / `entropy_sum` / `num_valid_tokens` over non-ignored tokens. For MoE
    models (`output_router_logits=True`), the load-balancing aux loss is added with the same coefficient and formula as
    the model's reference forward.

    Without labels, the original forward runs unchanged — generation and labels-free eval preserve any per-model logits
    post-processing (`logit_scale`, `final_logit_softcapping`, `logits_to_keep` slicing).

    Args:
        model (`torch.nn.Module`):
            Model to patch. For PEFT, pass `peft_model.get_base_model()` rather than the `PeftModel` wrapper, so
            prompt-learning variants (PromptTuning, PrefixTuning, PTuning) keep their virtual-token injection in
            `PeftModel.forward` before delegating into the patched forward.
        chunk_size (`int`):
            Number of valid tokens processed per CE chunk.
        is_vlm (`bool`):
            Set to `True` for VLMs. Only used to read `logit_scale` / `final_logit_softcapping` /
            `output_router_logits` from `model.config.text_config` instead of the top-level config.
    """
    # VLM scaling configs (`logit_scale`, `final_logit_softcapping`, MoE `output_router_logits`) live on `text_config`;
    # text-only models keep them on the top-level config.
    text_config = model.config.text_config if is_vlm else model.config
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    logit_scale = getattr(text_config, "logit_scale", 1.0)
    original_forward = model.forward
    lm_head = model.get_output_embeddings()

    def _chunked_ce_forward(
        self: torch.nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        shift_labels: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Without labels, fall back to the original forward so generation and labels-free evaluation
        # preserve any per-model logits post-processing (e.g. Cohere `logit_scale`, Gemma
        # `final_logit_softcapping`, `logits_to_keep` slicing).
        if labels is None and shift_labels is None:
            # MoE models: request router logits so the model returns `outputs.aux_loss`. VLM wrappers honor this only
            # as a forward kwarg (not from the model config), so it must be passed here.
            if output_router_logits is not None:
                kwargs["output_router_logits"] = output_router_logits
            return original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if output_router_logits is None:
            output_router_logits = getattr(text_config, "output_router_logits", False)

        kwargs.pop("use_cache", None)
        decoder_kwargs = {}
        # MoE models: request router logits so the model returns `outputs.aux_loss`. VLM wrappers honor this only
        # as a forward kwarg (not from the model config), so it must be passed here.
        if output_router_logits:
            decoder_kwargs["output_router_logits"] = True
        # `base_model` gives the backbone model (skipping `lm_head`) — text decoder for LMs, multimodal wrapper
        # for VLMs (so vision-token injection runs before the text decoder). `get_decoder()` won't do: on VLMs it
        # returns just the text stack and feeds image-placeholder IDs through it.
        # Pre-5.0 transformers VLMs set `base_model_prefix = ""` so `self.base_model is self` (re-runs `lm_head`).
        # Fall back to `self.model` there.
        if is_vlm and Version(transformers.__version__) < Version("5.0.0"):
            backbone = self.model
        else:
            backbone = self.base_model
        outputs: BaseModelOutputWithPast = backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **decoder_kwargs, **kwargs
        )
        hidden_states = outputs.last_hidden_state

        lm_head_weight = lm_head.weight
        lm_head_bias = lm_head.bias
        # Under FSDP2, lm_head.weight is a DTensor (Shard(0) or Replicate). Passing it directly
        # into the gradient-checkpointed chunk loop causes FSDP2 to re-gather it once per chunk
        # during backward recomputation. full_tensor() converts it to a plain tensor once; all
        # chunks reference that tensor, so only one all-gather occurs (in full_tensor()'s backward).
        if isinstance(lm_head_weight, torch.distributed.tensor.DTensor):
            lm_head_weight = lm_head_weight.full_tensor()
            if lm_head_bias is not None:
                lm_head_bias = lm_head_bias.full_tensor()
        loss, num_correct_tokens, entropy_sum, num_valid_tokens = _chunked_cross_entropy_loss(
            hidden_states,
            lm_head_weight,
            chunk_size,
            labels=labels,
            shift_labels=shift_labels,
            num_items_in_batch=num_items_in_batch,
            logit_scale=logit_scale,
            final_logit_softcapping=final_logit_softcapping,
            lm_head_bias=lm_head_bias,
        )

        aux_loss = None
        if output_router_logits:
            # Mirror the per-family MoE forward: add `router_aux_loss_coef * load_balancing_loss_func(...)` to
            # the main loss. Mixtral is the source of truth — every MoE family (Qwen3Moe, GptOss, OLMoE,
            # Qwen2Moe, DBRX, JetMoE, PhiMoE, …) pulls this function from mixtral via the modular system, so a
            # single import keeps us in lockstep with upstream for every family we test.
            from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

            if Version(transformers.__version__) < Version("5.0.0") and not is_vlm:
                num_experts = self.num_experts
                num_experts_per_tok = self.num_experts_per_tok
                router_aux_loss_coef = self.router_aux_loss_coef
            else:
                # Upstream bug AttributeError: 'GptOssConfig' object has no attribute 'num_experts'; see #5754
                if text_config.model_type == "gpt_oss" and Version("5.0.0") <= Version(
                    transformers.__version__
                ) < Version("5.6.0"):
                    num_experts = self.num_experts
                else:
                    num_experts = text_config.num_experts
                num_experts_per_tok = text_config.num_experts_per_tok
                router_aux_loss_coef = text_config.router_aux_loss_coef
            aux_loss = load_balancing_loss_func(
                outputs.router_logits, num_experts, num_experts_per_tok, attention_mask
            )
            loss = loss + router_aux_loss_coef * aux_loss.to(loss.device)

        return _ChunkedCELMHeadOutput(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            num_correct_tokens=num_correct_tokens,
            entropy_sum=entropy_sum,
            num_valid_tokens=num_valid_tokens,
            aux_loss=aux_loss,
        )

    # Keep the original forward signature so `generate`'s `_validate_model_kwargs` still sees the
    # model's real inputs (e.g. VLM `pixel_values`, `spatial_shapes`) and doesn't reject them. The
    # unbound `__func__` signature makes `MethodType`'s `self`-stripping land correctly.
    _chunked_ce_forward.__signature__ = inspect.signature(original_forward.__func__)
    model.forward = types.MethodType(_chunked_ce_forward, model)


logger = get_logger(__name__)


FLASH_ATTENTION_VARIANTS = {
    "flash_attention_2",
    "flash_attention_3",
    "kernels-community/flash-attn2",
    "kernels-community/flash-attn3",
    "kernels-community/vllm-flash-attn3",
}


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling data. Inputs are dynamically padded to the maximum length of a batch.

    This collator expects each example in the input list to be a dictionary containing at least the `"input_ids"` key.
    If the input contains `"labels"`, they are used as is (padded like the input IDs); otherwise the labels default to
    the input IDs. Tokens that shouldn't contribute to the loss are expected to be already set to `-100` in the labels;
    the [`SFTTrainer`] takes care of this during dataset preparation. The collator returns a dictionary containing the
    following keys:
    - `"input_ids"`: Tensor of input IDs, padded to the maximum length of the batch.
    - `"labels"`: Tensor of labels, padded with `-100` to the maximum length of the batch. If `padding_free` is set
    to `False`, the following key is also returned:
    - `"attention_mask"`: Tensor of attention masks, padded to the maximum length of the batch.
    If `padding_free` is set to `True`, the following key is also returned:
    - `"position_ids"`: Tensor of position IDs, padded to the maximum length of the batch.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        padding_free (`bool`, *optional*, defaults to `False`):
            If set to `True`, the sequences will be flattened into a single sequence, and the position IDs will be
            generated accordingly and returned instead of the attention mask.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0)
    >>> examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[   1,    2,    3],
                       [   4,    5, -100]])}

    >>> # With prebuilt labels
    >>> examples = [
    ...     {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
    ...     {"input_ids": [4, 5], "labels": [-100, 5]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[  1,  2,  3],
                          [  4,  5,  0]]),
     'attention_mask': tensor([[  1,  1,  1],
                               [  1,  1,  0]]),
     'labels': tensor([[-100,    2,    3],
                       [-100,    5, -100]])}

    >>> # With padding_free
    >>> collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    >>> collator(examples)
    {'input_ids': tensor([[ 1, 2, 3, 4, 5]]),
     'position_ids': tensor([[0, 1, 2, 0, 1]]),
     'labels': tensor([[-100, 2, 3, -100, 5]])}
    ```
    """

    pad_token_id: int
    padding_free: bool = False
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [example["input_ids"] for example in examples]
        batch_seq_lengths = [example["seq_lengths"] for example in examples] if "seq_lengths" in examples[0] else None
        labels = [example.get("labels", example["input_ids"]) for example in examples]

        # Convert to tensor
        input_ids = [torch.tensor(ids) for ids in input_ids]
        labels = [torch.tensor(lbl) for lbl in labels]

        # For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and
        # compute wrong cu_seq_lens from the all-1s mask
        if self.padding_free:
            if batch_seq_lengths is not None:
                position_ids = self.get_position_ids_from_packed_seq_lengths(batch_seq_lengths)
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        else:
            attention_mask = [torch.ones_like(ids) for ids in input_ids]

        # If padding_free, flatten everything into a single sequence
        output = {}
        if self.padding_free:
            input_ids = [torch.cat(input_ids, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            position_ids = [torch.cat(position_ids, dim=0)]

        # Pad
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.padding_free:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][output["position_ids"] == 0] = -100
        else:
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        return output

    @staticmethod
    def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        """
        Get position IDs for packed sequences.

        Args:
            batch_seq_lengths (`list[list[int]]`):
                A list of lists containing the lengths of each individual document in the packed batch.

        Return:
            `list[torch.Tensor]`:
                A list of tensors containing the position IDs for each packed sequence.
        """
        # Get lengths per row
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        # Flat list of lengths
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        # Reset position ids to 0 at the start of each sequence
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        # Split back into one tensor per example
        return list(position_ids.split(example_lengths))


@dataclass
class DataCollatorForVisionLanguageModeling(DataCollatorMixin):
    """
    Data collator for vision-language modeling tasks.

    Unlike text-only datasets, where the collator typically receives pre-tokenized inputs ready for batching,
    vision-language data processing involves converting images into pixel values. This conversion is disk-intensive,
    making upfront preprocessing of the entire dataset impractical. Therefore, this collator performs tokenization and
    image processing on-the-fly to efficiently prepare batches.

    Each input example should be a dictionary containing at least:
    - An `"images"` key holding a list of images, or an `"image"` key holding a single image.
    - [language modeling](#language-modeling) type: either a `"messages"` key for conversational inputs or a `"text"`
      key for standard text inputs.
    - [prompt-completion](#prompt-completion) type: keys `"prompt"` and `"completion"` for the prompt and completion.

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs.
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"pixel_values"`: Tensor representing image pixel values.
    - `"labels"`: Tensor for training labels.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"` or `"image_position_ids"`.

    Args:
        processor ([`~transformers.ProcessorMixin`]):
            The processor used to tokenize text and process images. It must be a subclass of
            [`~transformers.ProcessorMixin`] and include a `tokenizer` with a defined `pad_token_id`.
        max_length (`int`, *optional*):
            Maximum sequence length. Sequences longer than `max_length` are truncated to `max_length`. If `None`, no
            truncation is applied.
        completion_only_loss (`bool`, *optional*, defaults to `False`):
            Whether to compute loss only on the completion part of the sequence. When `True`, the labels for the prompt
            part are set to -100. It requires the dataset type to be prompt-completion.
        pad_to_multiple_of (`int`, *optional*):
            If set, the sequences will be padded to a multiple of this value.
        dataset_text_field (`str`, *optional*, defaults to `"text"`):
            Name of the column that contains text data in the dataset. This parameter is only relevant for [standard
            datasets format](dataset_formats#standard).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Example:
    ```python
    >>> from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    >>> from transformers import AutoProcessor

    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    >>> collator = DataCollatorForVisionLanguageModeling(processor)
    >>> examples = [
    ...     {"images": [Image.open("image_0.png")], "messages": [{"role": "user", "content": "What is this?"}]},
    ...     {"images": [Image.open("image_1.png")], "messages": [{"role": "user", "content": "Describe this image."}]},
    ... ]
    >>> collator(examples)
    {'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                              419,     30, 151645,    198],
                          [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                           151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                             2168,     13, 151645,    198]]),
     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
     'pixel_values': tensor([[-0.9893,  0.1785,  1.5362,  ..., -0.0582,  0.8661, -0.2431],
                             [-0.2302,  0.9522, -1.1061,  ...,  0.0555,  1.3354, -0.6412],
                             [ 1.2150,  0.9084,  0.7041,  ...,  0.2404, -0.8403, -0.5133],
                             ...,
                             [ 0.6895,  0.2807,  0.2515,  ..., -0.2004, -1.2100,  0.0555],
                             [ 0.8209, -0.9748,  1.5654,  ...,  1.6055, -0.4706,  0.5817],
                             [-1.0915,  0.4559,  0.9230,  ...,  0.5106,  0.0982, -0.1720]]),
     'image_grid_thw': tensor([[1, 4, 4],
                               [1, 4, 4]]),
     'labels': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                        151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,   3838,    374,
                           419,     30, 151645,    198],
                        [151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,  151645,    198,
                         151644,    872,    198, 151652, 151655, 151655, 151655,  151655, 151653,  74785,    419,
                           2168,     13, 151645,    198]])}
    ```
    """

    processor: ProcessorMixin
    max_length: int | None = None
    completion_only_loss: bool = False  # default not used in practice; SFTTrainer always passes the relevant value
    pad_to_multiple_of: int | None = None
    dataset_text_field: str = "text"
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "messages" in examples[0] or self.dataset_text_field in examples[0]:
            if self.completion_only_loss:
                raise ValueError(
                    "The `completion_only_loss` argument is not supported for language modeling datasets."
                )
            return self._collate_language_modeling(examples)
        elif "prompt" in examples[0] and "completion" in examples[0]:
            return self._collate_prompt_completion(examples)
        else:
            raise KeyError(f"Unexpected input keys in examples: {list(examples[0].keys())}.")

    def _collate_language_modeling(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "image" in examples[0]:
            for example in examples:
                example["images"] = [example.pop("image")]
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None

        if "messages" in examples[0]:  # conversational case
            messages = [
                prepare_multimodal_messages(example["messages"], images=example["images"]) for example in examples
            ]
            texts = self.processor.apply_chat_template(messages)
        elif self.dataset_text_field in examples[0]:  # standard case
            texts = [example[self.dataset_text_field] for example in examples]
        else:
            raise KeyError(
                "The input examples must contain either 'messages' for conversational data or 'text' for standard "
                "data."
            )

        output = self.processor(
            images=images,
            text=texts,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS twice, see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100
        # We mask only padding tokens (-100) in the labels. Vision tokens are left unchanged because their handling in
        # loss computation has to be done by the model, and masking them here would be infeasible in practice as vision
        # token definitions vary across architectures.
        output["labels"] = labels
        return output

    def _collate_prompt_completion(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError(
                "Padding to a multiple of a value is not yet implemented for vision-language modeling and "
                "prompt-completion data."
            )
        if "image" in examples[0]:
            for example in examples:
                example["images"] = [example.pop("image")]
        images = [example["images"] for example in examples]
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                example["prompt"] = prepare_multimodal_messages(example["prompt"], images=example["images"])
                example["completion"] = prepare_multimodal_messages(example["completion"])
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS twice, see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS twice, see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, prompt_mask = processed_prompts["input_ids"], processed_prompts["attention_mask"]
        completion_ids, completion_mask = processed_completions["input_ids"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
        if "token_type_ids" in processed_prompts:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions["token_type_ids"]
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)
        if "mm_token_type_ids" in processed_prompts:  # special case for Qwen2.5-VL
            prompt_mm_token_type_ids = processed_prompts["mm_token_type_ids"]
            mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, torch.zeros_like(completion_ids)), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts and "mm_token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids, mm_token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids, mm_token_type_ids
            )
        elif "token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, token_type_ids
            )
        elif "mm_token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, mm_token_type_ids = flush_left(
                attention_mask, input_ids, completion_mask, mm_token_type_ids
            )
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]
            if "token_type_ids" in processed_prompts:
                token_type_ids = token_type_ids[:, : self.max_length]
            if "mm_token_type_ids" in processed_prompts:
                mm_token_type_ids = mm_token_type_ids[:, : self.max_length]

        # Create labels and mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            labels[completion_mask == 0] = -100

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids
        if "mm_token_type_ids" in processed_prompts:
            output["mm_token_type_ids"] = mm_token_type_ids
        return output


def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT loss function, as presented in [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward
    Rectification](https://huggingface.co/papers/2508.05629)
    """
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:]
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss


class SFTTrainer(_BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning (SFT) method.

    This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.

    Example:

    ```python
    >>> from trl import SFTTrainer
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    >>> trainer = SFTTrainer(
    ...     model="Qwen/Qwen2.5-0.5B-Instruct",
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
        args ([`SFTConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`~trainer.sft_trainer.DataCollatorForLanguageModeling`] if the model is a language model
            and [`~trainer.sft_trainer.DataCollatorForVisionLanguageModeling`] if the model is a vision-language model.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. This trainer supports both [language modeling](#language-modeling) type and
            [prompt-completion](#prompt-completion) type. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).

            The trainer also supports pre-tokenized datasets, recognized by a required `input_ids` column. An optional
            `labels` column (`-100` on tokens excluded from the loss) is used as is if present; otherwise labels are
            built from the optional `assistant_masks` / `completion_mask` columns (which are folded in then dropped,
            `completion_mask` only when `completion_only_loss=True`), or default to a copy of `input_ids`. Sequences
            are truncated to `max_length` during preparation. With `skip_prepare_dataset=True`, preparation is skipped
            and the collator is expected to handle the dataset as is.

            When `train_dataset` is an [`~datasets.IterableDataset`] (e.g. a streaming dataset), `max_steps` must be
            set in the training arguments, since its length cannot be inferred and the total number of training steps
            is required to bound the training loop and configure the learning rate scheduler.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`], [`~datasets.DatasetDict`], [`~datasets.IterableDatasetDict`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoProcessor.from_pretrained`]. A padding token, `tokenizer.pad_token`, must be set.
            If the processing class has not set a padding token, `tokenizer.eos_token` will be used as the default.
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss
            function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618)
            used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.EvalPrediction`] and return a dictionary string to metric values. When passing
            [`SFTConfig`] with `batch_eval_metrics` set to `True`, your `compute_metrics` function must take a boolean
            `compute_result` argument. This will be triggered after the last eval batch to signal that the function
            needs to calculate and return the global summary statistics rather than accumulating the batch-level
            statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
            model and a scheduler given by [`~transformers.get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.

            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before
            initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        quantization_config ([`~transformers.BitsAndBytesConfig`], *optional*):
            Quantization configuration used when loading the model from a model identifier. Combine with `peft_config`
            for QLoRA training. Ignored if the model is already instantiated.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        formatting_func (`Callable`, *optional*):
            Formatting function applied to the dataset before tokenization. Applying the formatting function explicitly
            converts the dataset into a [language modeling](#language-modeling) type.
    """

    _tag_names = ["trl", "sft"]
    _name = "SFT"

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        args: SFTConfig | TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        quantization_config: "BitsAndBytesConfig | None" = None,
        peft_config: "PeftConfig | None" = None,
        formatting_func: Callable[[dict], str] | None = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            if Version(transformers.__version__) < Version("5.0.0"):
                dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `SFTConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        elif not isinstance(train_dataset, Dataset):
            raise TypeError(
                f"`train_dataset` must be a `Dataset` or `IterableDataset`, got `{type(train_dataset).__name__}`."
            )

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
                    "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
            if quantization_config is not None:
                logger.warning(
                    "You passed `quantization_config` to the trainer, but your model is already instantiated. The "
                    "`quantization_config` will be ignored."
                )
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        _is_quantized_model = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), trust_remote_code=args.trust_remote_code
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

        if args.eos_token is not None:
            if args.eos_token not in self._tokenizer.get_vocab():
                raise ValueError(
                    f"The specified `eos_token` ('{args.eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            self._tokenizer.eos_token = args.eos_token

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Catch some wrong configurations related to VLMs
        if self._is_vlm and args.packing:
            raise ValueError(
                "Packing is not supported for vision-language models. Please set `packing=False` in the SFTConfig."
            )
        if self._is_vlm and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision-language models. Please set "
                "`padding_free=False` in the `SFTConfig`."
            )
        if self._is_vlm and args.assistant_only_loss:
            raise ValueError(
                "Assistant-only loss is not yet supported for vision-language models. Please set "
                "`assistant_only_loss=False` in the `SFTConfig`."
            )
        if self._is_vlm and args.max_length is not None and args.truncation_mode == "keep_end":
            raise ValueError(
                "truncation_mode='keep_end' is not supported for vision-language models. Image tokens reside "
                "inside the prompt portion of the sequence; depending on the example, keep_end may silently "
                "drop them, causing pixel_values to be forwarded to the model with no corresponding visual "
                "tokens in input_ids. Use truncation_mode='keep_start' (the default) or set max_length=None."
            )

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
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)
                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")
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

        # In Prompt Tuning a small set of trainable virtual tokens (continuous prompt embeddings) is prepended to the
        # input. We store the number of these tokens so we can account for them correctly when calculating accuracy.
        self.num_virtual_tokens = 0
        if is_peft_model(model):
            if model.active_adapter in model.peft_config:
                peft_model_config = model.peft_config[model.active_adapter]
                self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)

        # Data collator
        # BFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing
        # FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask.
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy in {"bfd", "bfd_split"})
        use_flash_attention = model.config._attn_implementation in FLASH_ATTENTION_VARIANTS
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                logger.warning(
                    "You are passing `padding_free=True` with the 'wrapped' packing strategy, which is not "
                    "recommended. Please refer to the documentation to understand why this is not recommended."
                )
            if not use_flash_attention:
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to a supported "
                    "Flash Attention variant. Padding-free training flattens batches into a single sequence, and only "
                    "the following implementations are known to reliably support this: "
                    f"{', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. Using other implementations may lead to "
                    "unexpected behavior. To ensure compatibility, set `attn_implementation` in the model "
                    "configuration to one of these supported options or verify that your attention mechanism can "
                    "handle flattened sequences."
                )

            if args.per_device_train_batch_size == 1 and not args.packing:
                logger.warning(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 annihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )

        # Decide whether to use completion-only loss: if not specified, then it is set to True if the dataset format
        # is prompt-completion, and False if the dataset format is language modeling.
        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )

        if data_collator is None and not self._is_vision_dataset:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or self._tokenizer.pad_token or self._tokenizer.eos_token
            if pad_token not in self._tokenizer.get_vocab():
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            self._tokenizer.pad_token = pad_token
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=self._tokenizer.pad_token_id,
                padding_free=self.padding_free,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionLanguageModeling(
                processor=processing_class,
                max_length=args.max_length,
                completion_only_loss=self.completion_only_loss,
                pad_to_multiple_of=args.pad_to_multiple_of,
                dataset_text_field=args.dataset_text_field,
            )

        if args.packing and args.packing_strategy in {"bfd", "bfd_split"} and not use_flash_attention:
            logger.warning(
                "You are using packing, but the attention implementation is not set to a supported Flash Attention "
                "variant. Packing gathers multiple samples into a single sequence, and only the following "
                f"implementations are known to reliably support this: {', '.join(sorted(FLASH_ATTENTION_VARIANTS))}. "
                "Using other implementations may lead to cross-contamination between samples. To avoid this, either "
                "disable packing by setting `packing=False`, or set `attn_implementation` in the model configuration "
                "to one of these supported options."
            )
        if args.assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError(
                "You set `assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )

        # When assistant_only_loss is enabled, swap in a training chat template with {% generation %} markers
        # if the current template doesn't already have them.
        if args.assistant_only_loss and not has_generation_markers(processing_class.chat_template):
            self.chat_template = get_training_chat_template(processing_class)
        else:
            self.chat_template = None

        # A template can define generation markers and still attribute the assistant's end-of-turn token to the next
        # message, leaving it out of the assistant mask so the model is never trained to stop.
        if args.assistant_only_loss and not is_chat_template_stop_token_trained(
            processing_class, chat_template=self.chat_template
        ):
            logger.warning(
                "The chat template does not include the assistant turn's end-of-turn token in the loss mask; "
                "the model may not learn to stop."
            )

        # Dataset
        if self.padding_free and not args.packing and args.max_length is not None and not self._is_vision_dataset:
            raise ValueError(
                "When `padding_free=True` without packing, `max_length` is not enforced. Either enable packing "
                "(e.g., `packing=True, packing_strategy='bfd'`), provide already truncated inputs, or set "
                "`max_length=None`."
            )
        # Skip dataset preparation if `skip_prepare_dataset=True` in `dataset_kwargs`, or if it's a VLM, where
        # preprocessing (e.g., image-to-pixel conversion) is too costly and done on the fly instead.
        self._skip_prepare_dataset = (
            args.dataset_kwargs is not None
            and args.dataset_kwargs.get("skip_prepare_dataset", False)
            or self._is_vision_dataset
        )
        # Kept on the instance so that `evaluate` can preprocess freshly-passed eval datasets the same way.
        self._formatting_func = formatting_func
        eval_datasets = (
            eval_dataset if isinstance(eval_dataset, dict) else {"eval": eval_dataset} if eval_dataset else {}
        )
        self._reject_skip_prepare_without_labels({"train": train_dataset, **eval_datasets}, data_collator)
        if not self._skip_prepare_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Loss function
        if not args.use_liger_kernel:  # liger supports dft loss by just passing use_token_scaling=True
            if args.loss_type == "nll":
                pass  # use the default loss
            elif args.loss_type == "dft":
                if compute_loss_func is not None:
                    raise ValueError(
                        "You passed a `compute_loss_func` together with `loss_type='dft'` to the `SFTTrainer`. "
                        "When using `loss_type='dft'`, the loss function is internally set to the DFT loss, so "
                        "passing a `compute_loss_func` is not allowed."
                    )
                compute_loss_func = dft_loss
            elif args.loss_type == "chunked_nll":
                # Same math as `"nll"` but the `lm_head` matmul is skipped on ignored tokens and the CE is computed in
                # chunks of tokens. Implemented by patching the model's forward before `super().__init__` so accelerate
                # wraps the patched forward.
                # For PEFT, patch the inner causal LM rather than the `PeftModel` wrapper. LoRA / IA³ /
                # `modules_to_save` adapters live in the module tree, so they're hit even when we bypass
                # `PeftModel.forward`. Prompt-learning variants need `PeftModel.forward` to run first (to inject
                # virtual tokens), then it delegates into the patched inner forward.
                target = model.get_base_model() if is_peft_model(model) else model
                # The chunked path reads the output projection weight directly, which would silently drop the
                # adapter delta (and starve its parameters of gradients) if the head itself is a PEFT tuner layer.
                if is_peft_model(model):
                    from peft.tuners.tuners_utils import BaseTunerLayer

                    if isinstance(target.get_output_embeddings(), BaseTunerLayer):
                        raise ValueError(
                            "`loss_type='chunked_nll'` is not supported when `lm_head` is wrapped by a PEFT adapter "
                            "(e.g. `target_modules='all-linear'` or explicitly including `'lm_head'`). Either remove "
                            "`lm_head` from `target_modules`, or switch to `loss_type='nll'`. If this is a real use "
                            "case for you, please open an issue at https://github.com/huggingface/trl/issues."
                        )
                _patch_chunked_ce_lm_head(target, chunk_size=_CHUNKED_LM_HEAD_CHUNK_SIZE, is_vlm=self._is_vlm)
            else:
                raise ValueError(
                    f"Invalid `loss_type` {args.loss_type} passed. Supported values are 'nll', 'dft', and "
                    "'chunked_nll'."
                )
        elif args.loss_type == "chunked_nll":
            raise ValueError("`loss_type='chunked_nll'` is not compatible with `use_liger_kernel=True`.")

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
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # MoE load-balancing auxiliary loss, applied to Mixture-of-Experts models (no effect otherwise)
        text_config = model.config.get_text_config()
        is_moe = getattr(text_config, "output_router_logits", None) is not None
        self.aux_loss_enabled = is_moe and self.args.router_aux_loss_coef != 0.0
        if is_moe:
            # The native and chunked forwards add the aux loss from the model config, so keep the config in sync with
            # the coef: enable it (and propagate the coef) when non-zero, disable it otherwise. This overrides any
            # `output_router_logits` the model was loaded with, so `router_aux_loss_coef=0.0` reliably turns it off.
            text_config.output_router_logits = self.aux_loss_enabled
            text_config.router_aux_loss_coef = self.args.router_aux_loss_coef

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        args: SFTConfig,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        if isinstance(dataset, Dataset) and dataset.format["type"] == "custom":
            raise ValueError(
                "SFTTrainer cannot prepare a dataset that uses `Dataset.with_transform()`. The preparation pipeline "
                "calls `Dataset.map()`, which reads through the transform and can bake a random or stateful transform "
                "into the tokenized columns. Pass `dataset_kwargs={'skip_prepare_dataset': True}` and make the "
                "transform return trainer-ready examples, including tokenized fields, or materialize deterministic "
                "transforms with `Dataset.map()` before constructing the trainer."
            )

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = get_dataset_column_names(dataset)
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = get_dataset_column_names(dataset)
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Add EOS token if needed: non-conversational only
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": self._tokenizer.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize_fn(example, processing_class, dataset_text_field, assistant_only_loss, chat_template):
                    tools = example.get("tools")
                    tools = json.loads(tools) if isinstance(tools, str) else tools
                    apply_chat_template_kwargs = {
                        "chat_template": chat_template,
                        "tools": tools,
                        **example.get("chat_template_kwargs", {}),
                    }
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            prompt_ids = _tokenize(
                                processing_class,
                                example["prompt"],
                                add_generation_prompt=True,
                                **apply_chat_template_kwargs,
                            )["input_ids"]
                            prompt_completion_processed = _tokenize(
                                processing_class,
                                example["prompt"] + example["completion"],
                                return_assistant_tokens_mask=assistant_only_loss,
                                **apply_chat_template_kwargs,
                            )
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = _tokenize(processing_class, example["prompt"], chat_template=chat_template)[
                                "input_ids"
                            ]
                            prompt_completion_ids = _tokenize(
                                processing_class,
                                example["prompt"] + example["completion"],
                                chat_template=chat_template,
                            )["input_ids"]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask

                    else:  # language modeling case
                        if is_conversational(example):
                            processed = _tokenize(
                                processing_class,
                                example["messages"],
                                return_assistant_tokens_mask=assistant_only_loss,
                                **apply_chat_template_kwargs,
                            )
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {
                                "input_ids": _tokenize(
                                    processing_class, example[dataset_text_field], chat_template=chat_template
                                )["input_ids"]
                            }

                    if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                        raise RuntimeError(
                            "You're using `assistant_only_loss=True`, but at least one example has no assistant "
                            "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
                            "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
                            "ensure it's correctly configured to support assistant masking."
                        )
                    return output

                dataset = dataset.map(
                    tokenize_fn,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                        "chat_template": self.chat_template,
                    },
                    **map_kwargs,
                )

            # Build a "labels" column, setting tokens that shouldn't contribute to the loss to -100 based on the
            # available masks: "assistant_masks" always applies, "completion_mask" only when completion_only_loss
            # is enabled. With no applicable mask, every token contributes (labels == input_ids). A dataset that
            # already provides a "labels" column is left as is.
            column_names = get_dataset_column_names(dataset)
            if "labels" not in column_names:
                mask_columns = []
                if self.completion_only_loss and "completion_mask" in column_names:
                    mask_columns.append("completion_mask")
                if "assistant_masks" in column_names:
                    mask_columns.append("assistant_masks")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Building labels for {dataset_name} dataset"

                def build_labels(example, mask_columns):
                    masks = [example[column] for column in mask_columns]
                    labels = [
                        token_id if all(bits) else -100
                        for token_id, *bits in zip(example["input_ids"], *masks, strict=False)
                    ]
                    return {"labels": labels}

                dataset = dataset.map(
                    build_labels,
                    fn_kwargs={"mask_columns": mask_columns},
                    remove_columns=mask_columns,
                    **map_kwargs,
                )

            # Truncate to max_length. Skipped when packing, since packing already chunks sequences to max_length.
            # Done here, during preparation, so the result is cached. When preparation is skipped
            # (`skip_prepare_dataset=True`), no truncation is applied and the dataset must already be truncated.
            if args.max_length is not None and not packing:
                if args.truncation_mode == "keep_start":
                    sl = slice(None, args.max_length)
                elif args.truncation_mode == "keep_end":
                    sl = slice(-args.max_length, None)
                else:
                    raise ValueError(
                        f"Unsupported truncation mode: {args.truncation_mode}, expected 'keep_start' or 'keep_end'"
                    )
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"

                def truncate(example, sl):
                    return {"input_ids": example["input_ids"][sl], "labels": example["labels"][sl]}

                dataset = dataset.map(truncate, fn_kwargs={"sl": sl}, **map_kwargs)

                # Drop examples left fully masked by truncation (e.g. a prompt alone filling `max_length` with
                # `truncation_mode="keep_start"`), since they contribute no loss.
                if isinstance(dataset, Dataset):  # `IterableDataset.filter` does not support `desc`
                    map_kwargs["desc"] = f"Dropping fully masked examples from {dataset_name} dataset"

                dataset = dataset.filter(
                    lambda example: any(label != -100 for label in example["labels"]), **map_kwargs
                )

            # Pack
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                dataset = dataset.select_columns(["input_ids", "labels"])

                # Shuffle the dataset before packing. When using wrapped packing, it's important to shuffle before
                # packing as well to avoid correlations between sequences packed together.
                if args.shuffle_dataset:
                    dataset = dataset.shuffle(seed=args.seed)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "labels"}
                column_names = get_dataset_column_names(dataset)
                dataset = dataset.select_columns(collator_expected_keys.intersection(column_names))

        if args.shuffle_dataset:
            dataset = dataset.shuffle(seed=args.seed)

        return dataset

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids",
        # "attention_mask" and "labels"). Dataset preparation also produces a "seq_lengths" column (for packing /
        # padding-free), so we override the default signature columns to keep it alongside the model inputs.
        if self._signature_columns is None:
            if self._is_vision_dataset:
                self._signature_columns = ["messages", "prompt", "completion", "image", "images"]
            else:
                self._signature_columns = ["input_ids", "labels", "seq_lengths"]

    def _reject_skip_prepare_without_labels(self, datasets: dict[str, Dataset], data_collator) -> None:
        # This guard may look defensive, but it covers a behavior change introduced when label building moved from
        # the collator to dataset preparation: the collator used to consume the mask columns directly, so a
        # skipped-preparation dataset carrying masks trained correctly. Now labels are built during preparation, which
        # is skipped here, and the collator ignores the mask columns. Without a "labels" column, such a dataset would
        # silently optimize the loss over the full sequence, so we fail loudly instead. Checked both at init and in
        # `evaluate`, since a dataset passed directly to `evaluate` also skips preparation.
        if not (
            self._skip_prepare_dataset
            and not self._is_vision_dataset
            and isinstance(data_collator, DataCollatorForLanguageModeling)
        ):
            return
        for name, dataset in datasets.items():
            cols = get_dataset_column_names(dataset)
            if "labels" not in cols and ("completion_mask" in cols or "assistant_masks" in cols):
                raise ValueError(
                    f"The {name} dataset has mask columns but no 'labels', and `skip_prepare_dataset=True` skips "
                    "label building, so it would train on the full sequence. Add a 'labels' column (-100 for "
                    "non-loss tokens) or drop `skip_prepare_dataset`."
                )

    def evaluate(
        self,
        eval_dataset: Dataset
        | IterableDataset
        | DatasetDict
        | IterableDatasetDict
        | dict[str, Dataset | IterableDataset]
        | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # When a dataset is passed directly to `evaluate` (e.g. a held-out test set), preprocess it the same way
        # `__init__` does, so that `evaluate` accepts the same dataset types as the trainer (language modeling,
        # prompt-completion, etc.). `_prepare_dataset` is idempotent: it skips datasets that are already tokenized. A
        # `str` selects a dataset that was already prepared at init time, so it's left untouched.
        if not self._skip_prepare_dataset and eval_dataset is not None and not isinstance(eval_dataset, str):
            packing = self.args.packing if self.args.eval_packing is None else self.args.eval_packing
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(
                        dataset, self.processing_class, self.args, packing, self._formatting_func, key
                    )
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(
                    eval_dataset, self.processing_class, self.args, packing, self._formatting_func, "eval"
                )
        eval_datasets = (
            eval_dataset
            if isinstance(eval_dataset, dict)
            else {"eval": eval_dataset}
            if eval_dataset is not None and not isinstance(eval_dataset, str)
            else {}
        )
        self._reject_skip_prepare_without_labels(eval_datasets, self.data_collator)
        return super().evaluate(
            eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        prediction_loss_only = inputs.pop("_prediction_loss_only", None)

        # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
        # This can be removed when this issue is fixed.
        # When using CP or SP, labels are pre-shifted, we must use shift_labels instead.
        labels = inputs["labels"] if "shift_labels" not in inputs else None

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False

        # MoE models: request router logits so the model returns `outputs.aux_loss`. VLM wrappers honor this only
        # as a forward kwarg (not from the model config), so it must be passed here.
        if self.aux_loss_enabled:
            inputs["output_router_logits"] = True

        # Request token accuracy from Liger kernel and set token scaling if using DFT loss
        if self.args.use_liger_kernel:
            # Avoid materializing full logits during eval unless explicitly needed.
            # By default, liger kernel only skips logits during training (self.training=True).
            # When only loss is needed for eval (no compute_metrics), we can safely skip logits.
            # prediction_step communicates whether logits are expected via `_prediction_loss_only`;
            # this prevents skipping logits during `predict()` where outputs are requested.
            # Keep logits when preprocess_logits_for_metrics is set, even if compute_metrics is None.
            # to prevent massive vRAM spikes from the lm_head projection.
            # See: https://github.com/huggingface/trl/issues/4679
            inputs["skip_logits"] = (
                self.model.training
                or self.args.prediction_loss_only
                or (
                    self.compute_metrics is None
                    and self.preprocess_logits_for_metrics is None
                    and prediction_loss_only is not False
                )
            )
            inputs["return_token_accuracy"] = True
            inputs["use_token_scaling"] = self.args.loss_type == "dft"

        try:
            (loss, outputs) = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        except ValueError as e:
            if "Image features and image tokens do not match" in str(e) and self.args.max_length is not None:
                raise ValueError(
                    f"The current `max_length` ({self.args.max_length}) is too short and causes image placeholder "
                    f"tokens in `input_ids` to be truncated, while the corresponding image features remain intact. "
                    f"Please increase `max_length` or set it to `None` to disable truncation."
                ) from e
            raise

        # Compute entropy
        if self.args.loss_type == "chunked_nll":
            # Use `num_valid_tokens` from the patched forward rather than recomputing from `labels`. Prompt-learning
            # PEFT (PromptTuning, P-Tuning) prepends `-100`-padded virtual tokens before delegating into the patched
            # forward, so the valid-token count over the padded labels can differ from the un-padded `labels[..., 1:]`
            # count by up to one per sequence; using the patched output keeps numerator and denominator aligned.
            num_valid = self.accelerator.gather_for_metrics(outputs.num_valid_tokens).sum()
            entropy_sum = self.accelerator.gather_for_metrics(outputs.entropy_sum).sum()
            entropy = (entropy_sum / num_valid).item() if num_valid > 0 else 0.0
            self._metrics[mode]["entropy"].append(entropy)
        elif not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP or SP, labels are pre-shifted.
                    shift_logits = outputs.logits
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :]
                    shift_labels = labels[..., 1:]

                # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                per_token_entropy = entropy_from_logits(shift_logits)
                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100

                entropy_sum = (per_token_entropy * mask).sum()
                total_tokens = mask.sum()
                correct_predictions = (predictions == shift_labels) & mask
                correct_tokens = correct_predictions.sum()

                # Gather counts across ranks and weight-average
                entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
                total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0

                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["entropy"].append(entropy)
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        if self.args.loss_type == "chunked_nll":
            correct = self.accelerator.gather_for_metrics(outputs.num_correct_tokens).sum()
            accuracy = (correct / num_valid).item() if num_valid > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        elif self.args.use_liger_kernel:
            if hasattr(outputs, "token_accuracy") and outputs.token_accuracy is not None:
                token_accuracy = self.accelerator.gather_for_metrics(outputs.token_accuracy).mean().item()
                self._metrics[mode]["mean_token_accuracy"].append(token_accuracy)
            else:
                warnings.warn(
                    "liger-kernel did not return token_accuracy when requested. The mean_token_accuracy metric will "
                    "not be logged. This is unexpected; please report it to the liger-kernel repository.",
                    stacklevel=2,
                )
        # Log auxiliary loss if enabled (applies to both Liger and non-Liger)
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
            self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Preserve the eval loop intent so compute_loss can decide whether logits are needed.
        inputs["_prediction_loss_only"] = prediction_loss_only
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
