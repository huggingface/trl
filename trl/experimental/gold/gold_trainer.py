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

import os
import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer, is_bitsandbytes_available
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from ...data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from ...extras.profiling import profiling_decorator
from ...extras.vllm_client import VLLMClient
from ...import_utils import is_vllm_available
from ...models import prepare_deepspeed
from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import (
    DataCollatorForChatML,
    create_model_from_path,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from .gold_config import GOLDConfig


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

if is_bitsandbytes_available():
    import bitsandbytes as bnb


def print_prompt_completions_sample_uld(
    prompts: list[str],
    completions: list[str],
    step: int,
    num_samples: int = None,
) -> None:
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        advantages (`list[float]`):
            List of advantages corresponding to the prompts and completions.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int` or `None`, *optional*, defaults to `None`):
            Number of random samples to display. If `None` (default), all items will be displayed.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample

    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> advantages = [0.987, 0.654]
    >>> print_prompt_completions_sample(prompts, completions, rewards, advantages, 42)
    ╭──────────────────────────── Step 42 ─────────────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ │
    │ ├────────────┼──────────────┼─────────────┼────────┼───────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ │
    │ └────────────┴──────────────┴─────────────┴────────┴───────────┘ │
    ╰──────────────────────────────────────────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]

    for i in range(len(prompts)):
        table.add_row(Text(prompts[i]), Text(completions[i]))
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def build_teacher_inputs_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: list[str],
    completion_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Tokenize teacher prompts/completions and produce tensors ready for GOLD loss."""

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_token_ids = tokenizer(completion_texts, add_special_tokens=False)["input_ids"]

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    prompt_lengths: list[int] = []

    for prompt_ids, completion_ids in zip(prompt_token_ids, completion_token_ids, strict=True):
        # Remove trailing EOS from prompt so completions can extend cleanly
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]

        prompt_lengths.append(len(prompt_ids))
        sequence = list(prompt_ids)
        sequence.extend(completion_ids)
        if eos_token_id is not None:
            sequence.append(eos_token_id)

        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones_like(seq_tensor))

        labels = seq_tensor.clone()
        labels[: len(prompt_ids)] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        labels_list.append(labels)

    teacher_input_ids = pad(
        sequences,
        padding_side="right",
        padding_value=pad_token_id if pad_token_id is not None else 0,
    )
    teacher_attention_mask = pad(attention_masks, padding_side="right", padding_value=0).bool()
    teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)

    if eos_token_id is not None:
        for row in range(teacher_attention_mask.size(0)):
            valid = (
                teacher_input_ids[row] != pad_token_id
                if pad_token_id is not None
                else teacher_attention_mask[row].bool()
            )
            if valid.any():
                last_idx = valid.nonzero(as_tuple=True)[0][-1]
                teacher_attention_mask[row, last_idx + 1 :] = False

    teacher_prompt_length = max(prompt_lengths) if prompt_lengths else 0

    return teacher_input_ids, teacher_labels, teacher_attention_mask, teacher_prompt_length


class ULDLoss(nn.Module):
    """
    Universal Logit Distillation Loss.
    """

    def __init__(self, config: GOLDConfig, student_tokenizer=None, teacher_tokenizer=None):
        super().__init__()
        self.crossentropy_weight = config.uld_crossentropy_weight
        self.distillation_weight = config.uld_distillation_weight
        self.student_temperature = config.uld_student_temperature
        self.teacher_temperature = config.uld_teacher_temperature
        self.skip_student_eos = config.uld_skip_student_eos
        self.skip_teacher_eos = config.uld_skip_teacher_eos
        self.use_extended_uld = config.use_extended_uld
        self.ignore_index = -100

        # Add tokenizers for enhanced alignment
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Hybrid ULD configuration
        self.use_hybrid_loss = getattr(config, "uld_use_hybrid_loss", False)
        self.hybrid_matched_weight = getattr(config, "uld_hybrid_matched_weight", None)
        self.hybrid_unmatched_weight = getattr(config, "uld_hybrid_unmatched_weight", None)
        self.beta = getattr(config, "beta", 1.0)  # For JSD loss in hybrid matched tokens

        # Initialize vocabulary mapping for hybrid loss
        self._vocab_mapping = None
        self._teacher_matched_ids = None
        self._student_matched_ids = None
        if self.use_hybrid_loss and student_tokenizer is not None and teacher_tokenizer is not None:
            self._initialize_vocabulary_mapping()

    def __call__(
        self, student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    ):
        """
        Compute ULD loss with GKD trainer interface.

        Args:
            student_logits: Student model logits [batch_size, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            student_labels: Student target labels [batch_size, seq_len]
            teacher_labels: Teacher target labels [batch_size, seq_len]
            student_input_ids: Student input token IDs [batch_size, seq_len]
            teacher_input_ids: Teacher input token IDs [batch_size, seq_len]

        Returns:
            Total loss (cross-entropy + distillation)
        """
        # Compute cross-entropy loss for student
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            crossentropy_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = 0.0

        # Compute distillation loss using ULD approximation
        distillation_loss = self._compute_distillation_loss(
            student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
        )

        return crossentropy_loss + distillation_loss

    def _initialize_vocabulary_mapping(self):
        """Initialize vocabulary mapping for hybrid ULD loss."""
        # Computing vocabulary mapping for hybrid ULD

        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()

        # Create reverse mapping for student
        student_token_to_id = dict(student_vocab.items())

        vocab_mapping = {}
        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_id = student_token_to_id[token_str]
                vocab_mapping[teacher_id] = student_id
                teacher_matched_ids.add(teacher_id)
                student_matched_ids.add(student_id)

        self._vocab_mapping = vocab_mapping
        self._teacher_matched_ids = teacher_matched_ids
        self._student_matched_ids = student_matched_ids

    def _compute_distillation_loss(
        self, student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    ):
        """
        Compute the Universal Logit Distillation loss with token mapping.

        This version uses actual input_ids for accurate token mapping and multiplies probabilities for split tokens.
        Both student_input_ids and teacher_input_ids are required for optimal alignment.
        """
        # Get answer regions (same as original)
        student_answer_index, student_answer_size = self._get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(teacher_labels)

        if self.skip_student_eos:
            student_answer_size = [size - 1 for size in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [size - 1 for size in teacher_answer_size]

        # Handle edge case where all answer sizes are 0
        if (
            not student_answer_size
            or not teacher_answer_size
            or max(max(student_answer_size), max(teacher_answer_size)) <= 0
        ):
            return torch.zeros(1, device=student_logits.device, requires_grad=True) * student_logits.sum() * 1e-8

        batch_size = student_logits.size(0)
        distillation_losses = []

        for i in range(batch_size):
            # Get answer regions for this batch item
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            if student_size <= 0 or teacher_size <= 0:
                loss_i = student_logits[i].sum() * 0.0
                distillation_losses.append(loss_i)
                continue

            # Extract answer logits
            student_answer_logits = student_logits[i, student_start : student_start + student_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size]

            # Convert to probabilities
            student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)

            # Get token IDs for mapping (always use actual input_ids)
            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()

            if self.use_extended_uld:
                # Build alignment groups directly from token ids using greedy text matching
                student_alignment_groups, teacher_alignment_groups = self._build_alignment_groups_from_ids(
                    student_token_ids, teacher_token_ids
                )

                # Merge student probabilities using student alignment groups
                student_aligned = self._merge_probabilities_with_alignment_groups(
                    student_probs, student_alignment_groups
                )

                # Merge teacher probabilities using teacher alignment groups
                teacher_aligned = self._merge_probabilities_with_alignment_groups(
                    teacher_probs, teacher_alignment_groups
                )
            else:
                min_length = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_length, :]
                teacher_aligned = teacher_probs[:min_length, :]

            # Apply ULD loss computation
            if self.use_hybrid_loss and self._vocab_mapping is not None:
                # Use hybrid approach: direct comparison for matched tokens, sorting for unmatched
                aligned_loss = self._compute_hybrid_uld_loss(student_aligned, teacher_aligned)
            else:
                # Original approach: sort all probabilities
                student_sorted = student_aligned.sort(dim=-1, descending=True).values
                teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

                # Pad vocabularies to same size
                student_vocab_size = student_sorted.size(-1)
                teacher_vocab_size = teacher_sorted.size(-1)
                max_vocab_size = max(student_vocab_size, teacher_vocab_size)

                if student_vocab_size < max_vocab_size:
                    student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
                if teacher_vocab_size < max_vocab_size:
                    teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

                # Compute L1 distance (ULD approach)
                aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
                aligned_loss /= student_aligned.size(0)  # Normalize by sequence length
            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()
        return self.distillation_weight * distillation_loss

    def _build_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids):
        """
        Build alignment groups using a greedy substring-equality algorithm on decoded token pieces.

        Args:
            student_token_ids: List[int]
            teacher_token_ids: List[int]

        Returns:
            Tuple[List[List[int]], List[List[int]]]: student and teacher alignment groups
        """

        def to_canonical_pieces(tok, ids):
            pieces = []
            prev = ""
            for k in range(len(ids)):
                # IMPORTANT: Do NOT skip special tokens - we need to align them too
                cur = tok.decode(ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                # Extract the incremental addition (may include spaces/ZWJ/etc.)
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        s_pieces = to_canonical_pieces(self.student_tokenizer, student_token_ids)
        t_pieces = to_canonical_pieces(self.teacher_tokenizer, teacher_token_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group = []
        t_group = []
        s_groups = []
        t_groups = []

        def flush():
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        # Greedily accumulate pieces until substrings match, then flush
        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group = []
                t_group = []
                continue

            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        # Flush any remainder if both sides accumulated something
        if s_buf == t_buf and s_group and t_group:
            flush()
        elif s_group or t_group:
            # Handle remaining unmatched tokens by forcing a flush
            # This ensures both sides have the same number of alignment groups
            if s_group or t_group:
                # Ensure both groups have content (even if empty list)
                if not s_group:
                    s_group = []
                if not t_group:
                    t_group = []
                # Force flush even if buffers don't match
                if s_group or t_group:
                    s_groups.append(s_group.copy() if s_group else [])
                    t_groups.append(t_group.copy() if t_group else [])

        return s_groups, t_groups

    def _merge_probabilities_with_alignment_groups(self, probs, alignment_groups):
        """
        Merge probabilities based on alignment groups.

        Args:
            probs: Probability tensor [seq_len, vocab_size]
            alignment_groups: List of alignment groups (each group is a list of positions to merge)

        Returns:
            Merged probability tensor [num_groups, vocab_size]
        """
        if not alignment_groups:
            return probs

        # Create aligned tensor
        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

        # Process each alignment group
        for group_idx, group in enumerate(alignment_groups):
            # Handle probability merging
            if len(group) > 1:
                # Multiple tokens map to this group - merge them
                eps = 1e-8
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                # No tokens map to this group
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs

    def _compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):
        """
        Compute hybrid ULD loss on aligned probability distributions. This method:
        1. Directly compares probabilities for tokens with matching vocabulary entries
        2. Uses sorting approach only for tokens with different vocabulary entries

        Args:
            student_aligned: Aligned student probabilities [seq_len, student_vocab_size]
            teacher_aligned: Aligned teacher probabilities [seq_len, teacher_vocab_size]
        Returns:
            Combined hybrid loss
        """
        device = student_aligned.device
        # seq_len = student_aligned.size(0)  # Unused variable
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        # Convert sets to sorted tensors for indexing
        if self._teacher_matched_ids:
            teacher_matched_indices = torch.tensor(sorted(self._teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_indices = torch.tensor(
                [self._vocab_mapping[tid.item()] for tid in teacher_matched_indices], dtype=torch.long, device=device
            )
        else:
            teacher_matched_indices = torch.tensor([], dtype=torch.long, device=device)
            student_matched_indices = torch.tensor([], dtype=torch.long, device=device)

        # Create masks for unmatched tokens
        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_indices) > 0:
            teacher_matched_mask[teacher_matched_indices] = True
            student_matched_mask[student_matched_indices] = True

        # 1. JSD loss for matched vocabulary tokens (direct semantic correspondence)
        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_indices) > 0:
            # Extract probabilities for matched tokens
            teacher_matched_probs = teacher_aligned[:, teacher_matched_indices]  # [seq_len, num_matched]
            student_matched_probs = student_aligned[:, student_matched_indices]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)

            # Use JSD loss for semantically aligned tokens
            # Convert probabilities back to logits for JSD computation

            # Apply generalized JSD loss to matched tokens
            matched_loss = self._compute_jsd_loss_for_matched_tokens(student_matched_probs, teacher_matched_probs)

        # 2. Sorted comparison loss for unmatched vocabulary tokens
        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_mask]  # [seq_len, num_teacher_unmatched]
        student_unmatched_probs = student_aligned[:, student_unmatched_mask]  # [seq_len, num_student_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
            # Sort unmatched probabilities
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            # Pad to same size if needed
            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            # L1 loss on sorted unmatched tokens
            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            unmatched_loss /= student_aligned.size(0)  # Normalize by sequence length

        # 3. Combine losses with weights
        if self.hybrid_matched_weight is None:
            # Use adaptive weighting based on vocabulary overlap
            hybrid_matched_weight = matched_token_count / max(1, teacher_vocab_size)
            hybrid_unmatched_weight = 1.0 - hybrid_matched_weight
        else:
            # Use fixed weights provided in config
            hybrid_matched_weight = self.hybrid_matched_weight
            hybrid_unmatched_weight = self.hybrid_unmatched_weight

        total_loss = hybrid_matched_weight * matched_loss + hybrid_unmatched_weight * unmatched_loss

        # Store matched/unmatched components for logging
        self.last_matched_loss = matched_loss
        self.last_unmatched_loss = unmatched_loss

        return total_loss

    def _compute_jsd_loss_for_matched_tokens(self, student_logits, teacher_logits):
        """
        Compute JSD loss for matched vocabulary tokens.

        Args:
            student_logits: Student logits for matched tokens [seq_len, num_matched]
            teacher_logits: Teacher logits for matched tokens [seq_len, num_matched]
        Returns:
            JSD loss for matched tokens
        """
        # Reshape to [batch_size * seq_len, vocab_size] format expected by generalized_jsd_loss
        batch_seq_len, num_matched = student_logits.shape

        student_logits_reshaped = student_logits.view(-1, num_matched)
        teacher_logits_reshaped = teacher_logits.view(-1, num_matched)

        # Use the GOLD generalized JSD loss implementation that accepts probability inputs
        jsd_loss = GOLDTrainer.generalized_jsd_loss(
            student_logits_reshaped,
            teacher_logits_reshaped,
            labels=None,  # No masking needed for matched tokens
            beta=self.beta,  # Standard JSD beta
            temperature=1.0,  # Already applied in main computation
            reduction="batchmean",
            logits_are_probs=True,
        )

        return jsd_loss

    def _get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            answer_mask = answer.ne(self.ignore_index)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue

            valid_indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(valid_indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if hasattr(self.trainer.accelerator, "sync_gradients") and self.trainer.accelerator.sync_gradients:
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class GOLDTrainer(SFTTrainer):
    _tag_names = ["trl", "gold"]
    _name = "GOLD"
    _paper = {
        "title": "Unlocking On-Policy Distillation for Any Model Family",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @misc{patino2025unlocking,
                title        = {{Unlocking On-Policy Distillation for Any Model Family}},
                author       = {Carlos Miguel Patiño and Kashif Rasul and Quentin Gallouédec and Ben Burtenshaw and Sergio Paniego and Vaibhav Srivastav and Thibaud Frere and Ed Beeching and Lewis Tunstall and Leandro von Werra and Thomas Wolf},
                year         = 2025,
                url          = {https://huggingface.co/spaces/HuggingFaceH4/general-on-policy-logit-distillation},
            }"""),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(tokenizer=processing_class, max_length=args.max_length)

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GOLDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        if args.use_uld_loss and args.teacher_tokenizer_name_or_path is None:
            if isinstance(teacher_model, str):
                args.teacher_tokenizer_name_or_path = teacher_model
            else:
                raise ValueError(
                    "`teacher_tokenizer_name_or_path` must be set when using ULD loss with a pre-instantiated teacher model."
                )

        if isinstance(teacher_model, str):
            init_kwargs = dict(teacher_model_init_kwargs)
            if "torch_dtype" in init_kwargs and "dtype" not in init_kwargs:
                init_kwargs["dtype"] = init_kwargs.pop("torch_dtype")
            teacher_model = create_model_from_path(teacher_model, **init_kwargs)
        self.use_uld_loss = args.use_uld_loss
        self.teacher_tokenizer = None
        if args.use_uld_loss and args.teacher_tokenizer_name_or_path is not None:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer_name_or_path)
            if not hasattr(self.teacher_tokenizer, "pad_token") or self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        # Hybrid ULD loss configuration is handled in ULDLoss class

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)
        if not args.use_uld_loss:
            teacher_model.resize_token_embeddings(self.model.config.vocab_size)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        # Hybrid ULD matched/unmatched accumulators (logged every step when ULD hybrid is used)
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        self.uld_loss_fn = None
        if self.use_uld_loss:
            self.uld_loss_fn = ULDLoss(
                config=args,
                student_tokenizer=processing_class,
                teacher_tokenizer=self.teacher_tokenizer,
            )

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                vllm_quantization = None
                if is_bitsandbytes_available():
                    for _, module in model.named_modules():
                        if isinstance(module, bnb.nn.Linear4bit):
                            vllm_quantization = "bitsandbytes"
                            break
                        elif isinstance(module, bnb.nn.Linear8bitLt):
                            raise ValueError("vLLM does not support in-flight 8-bit quantization.")

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                    quantization=vllm_quantization,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "prompts",
            "prompt_attention_mask",
            "messages",
            "chat_template_kwargs",
            "tools",
            "original_prompt_text",
            "original_completion_text",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """
        Override dataset preparation to preserve original text for cross-tokenizer distillation and ensure
        attention_mask is always added for DataCollatorForChatML compatibility.
        """
        # Check if dataset is already processed
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Use our enhanced dataset preparation for:
        # 1. ULD loss with cross-tokenizer (need original text preservation)
        # 2. Any unprocessed dataset (need attention_mask for DataCollatorForChatML)
        if not is_processed or (self.use_uld_loss and self.teacher_tokenizer is not None):
            # For unprocessed datasets, use our enhanced tokenization
            return self._prepare_dataset_with_original_text(
                dataset, processing_class, args, packing, formatting_func, dataset_name
            )

        # Use parent implementation for all other cases
        return super()._prepare_dataset(dataset, processing_class, args, packing, formatting_func, dataset_name)

    def _prepare_dataset_with_original_text(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """
        Prepare dataset while preserving original text for cross-tokenizer distillation.
        """
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in version 0.21.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            # Convert the dataset to ChatML if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            column_names = next(iter(dataset)).keys()
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns="conversations" if "conversations" in column_names else None,
                **map_kwargs,
            )

            # Apply the chat template if needed and preserve original text
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
                    fn_kwargs={"eos_token": processing_class.eos_token},
                    remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                    **map_kwargs,
                )

            # Tokenize the dataset while preserving original text
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset (preserving original text)"

            def tokenize_with_original_text(example, processing_class, dataset_text_field, assistant_only_loss):
                """Modified tokenization function that preserves original text."""
                result = {}

                if "prompt" in example:  # prompt-completion case
                    # Store original text
                    result["original_prompt_text"] = example["prompt"]
                    result["original_completion_text"] = example["completion"]

                    if is_conversational(example):
                        prompt_ids = processing_class.apply_chat_template(
                            example["prompt"], **example.get("chat_template_kwargs", {})
                        )
                        prompt_completion_ids = processing_class.apply_chat_template(
                            example["prompt"] + example["completion"], **example.get("chat_template_kwargs", {})
                        )
                    else:
                        prompt_ids = processing_class(text=example["prompt"]).input_ids
                        prompt_completion_ids = processing_class(
                            text=example["prompt"] + example["completion"]
                        ).input_ids

                    # Check if the tokenized prompt starts with the tokenized prompt+completion
                    if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                        warnings.warn(
                            "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                            "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                            "token handling. Verify that the tokenizer is processing text consistently."
                        )

                    # Create a completion mask
                    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                    result.update(
                        {
                            "input_ids": prompt_completion_ids,
                            "completion_mask": completion_mask,
                            "attention_mask": [1] * len(prompt_completion_ids),  # Add attention mask
                        }
                    )

                else:  # language modeling or conversational case
                    if is_conversational(example):
                        # For conversational data (ChatML), extract prompt and completion properly
                        messages = example["messages"]

                        # Extract user and assistant messages separately
                        user_messages = [msg for msg in messages if msg["role"] != "assistant"]
                        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]

                        if user_messages and assistant_messages:
                            # Apply chat template to get the prompt (everything up to assistant)
                            prompt_text = processing_class.apply_chat_template(
                                user_messages,
                                tokenize=False,
                                add_generation_prompt=True,  # Add assistant prompt
                                **example.get("chat_template_kwargs", {}),
                            )

                            # Get the full conversation with assistant response
                            full_text = processing_class.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                                **example.get("chat_template_kwargs", {}),
                            )

                            # Extract completion as everything after the prompt
                            # This ensures we capture any extra tokens (like <think> tags) that the template adds
                            if full_text.startswith(prompt_text):
                                completion_text = full_text[len(prompt_text) :]
                            else:
                                # Fallback: use assistant content + EOS
                                assistant_content = assistant_messages[0]["content"]
                                completion_text = (
                                    assistant_content + processing_class.eos_token
                                    if hasattr(processing_class, "eos_token")
                                    else assistant_content
                                )

                            # Store original text for cross-tokenizer distillation
                            result["original_prompt_text"] = prompt_text
                            result["original_completion_text"] = completion_text
                        else:
                            # Fallback: use empty prompt and full text as completion
                            full_text = processing_class.apply_chat_template(
                                messages, tokenize=False, **example.get("chat_template_kwargs", {})
                            )
                            result["original_prompt_text"] = ""
                            result["original_completion_text"] = full_text

                        # Process the conversation normally
                        processed = processing_class.apply_chat_template(
                            example["messages"],
                            return_dict=True,
                            return_assistant_tokens_mask=assistant_only_loss,
                            **example.get("chat_template_kwargs", {}),
                        )
                        if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                            raise RuntimeError(
                                "You're using `assistant_only_loss=True`, but at least one example has no "
                                "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                "generate assistant masks — it may be missing the `{% generation %}` tag. Please "
                                "check the template and ensure it's correctly configured to support assistant "
                                "masking."
                            )
                        result.update({k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed})
                        # Add attention_mask if not already present
                        if "attention_mask" not in result:
                            result["attention_mask"] = [1] * len(result["input_ids"])
                    else:
                        # For regular language modeling, store the full text as completion and empty prompt
                        result["original_prompt_text"] = ""
                        result["original_completion_text"] = example.get(dataset_text_field, example.get("text", ""))

                        tokenized = processing_class(text=example[dataset_text_field])
                        result.update(
                            {
                                "input_ids": tokenized.input_ids,
                                "attention_mask": getattr(tokenized, "attention_mask", [1] * len(tokenized.input_ids)),
                            }
                        )

                return result

            dataset = dataset.map(
                tokenize_with_original_text,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": args.dataset_text_field,
                    "assistant_only_loss": args.assistant_only_loss,
                },
                **map_kwargs,
            )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns_to_keep = ["input_ids", "original_prompt_text", "original_completion_text"]
                existing_columns = set(dataset.column_names)
                columns_to_select = [col for col in columns_to_keep if col in existing_columns]

                dataset = dataset.select_columns(columns_to_select)
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)

            if args.use_liger_kernel:
                required_columns = {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "completion_mask",
                    "assistant_masks",
                    "original_prompt_text",
                    "original_completion_text",
                }
                dataset = dataset.select_columns(required_columns.intersection(dataset.column_names))

        return dataset

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature
            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_uld_loss and self.teacher_tokenizer is not None:
            if "original_prompt_text" in inputs and "original_completion_text" in inputs:
                prompt_texts = inputs["original_prompt_text"]
                completion_texts = inputs["original_completion_text"]
                full_texts = [p + c for p, c in zip(prompt_texts, completion_texts, strict=True)]
            else:
                # Fallback: decode student input_ids (current approach)
                # WARNING: This may not work perfectly for cross-tokenizer distillation
                full_sequences = inputs["input_ids"]
                full_texts = self.processing_class.batch_decode(full_sequences, skip_special_tokens=False)

                # Try to split prompt/completion using original prompt length
                prompt_lengths = inputs["prompts"].shape[1]
                prompt_texts = self.processing_class.batch_decode(inputs["prompts"], skip_special_tokens=False)
                completion_texts = [
                    full.replace(prompt, "", 1) for full, prompt in zip(full_texts, prompt_texts, strict=True)
                ]

            (
                teacher_input_ids,
                teacher_labels,
                teacher_attention_mask,
                teacher_prompt_length,
            ) = build_teacher_inputs_from_texts(
                self.teacher_tokenizer,
                prompt_texts,
                completion_texts,
            )

            teacher_input_ids = teacher_input_ids.to(self.accelerator.device)
            teacher_labels = teacher_labels.to(self.accelerator.device)
            teacher_attention_mask = teacher_attention_mask.to(self.accelerator.device)

            outputs_student = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            self.teacher_model.eval()
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                )

            # These are not used for ULD loss but are needed if JSD loss were to be used in this branch
            student_prompt_length = inputs["prompts"].shape[1]
            shifted_student_logits = outputs_student.logits[:, student_prompt_length - 1 : -1, :]
            shifted_teacher_logits = outputs_teacher.logits[:, teacher_prompt_length - 1 : -1, :]
            shifted_labels = inputs["labels"][:, student_prompt_length:]
        else:
            if self.use_liger_gkd_loss:
                # Forward only through the base models (avoid lm_head to save memory)
                unwrapped_student = self.accelerator.unwrap_model(model)
                if hasattr(unwrapped_student, "get_decoder") and unwrapped_student.get_decoder() is not None:
                    base_student = unwrapped_student.get_decoder()
                else:
                    base_student = getattr(
                        unwrapped_student, getattr(unwrapped_student, "base_model_prefix", "model"), unwrapped_student
                    )

                student_outputs = base_student(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )

                self.teacher_model.eval()
                unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
                if hasattr(unwrapped_teacher, "get_decoder") and unwrapped_teacher.get_decoder() is not None:
                    base_teacher = unwrapped_teacher.get_decoder()
                else:
                    base_teacher = getattr(
                        unwrapped_teacher, getattr(unwrapped_teacher, "base_model_prefix", "model"), unwrapped_teacher
                    )
                with torch.no_grad():
                    teacher_outputs = base_teacher(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        use_cache=False,
                    )

                # hidden states (shifted)
                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                # Release full outputs to free memory
                del student_outputs, teacher_outputs

                # labels mask and labels (shifted)
                labels_mask = inputs["labels"] != -100
                masked_input_ids = torch.where(
                    labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
                )
                true_labels = masked_input_ids[:, 1:].contiguous()

                # heads
                student_head = unwrapped_student.get_output_embeddings()
                teacher_head = unwrapped_teacher.get_output_embeddings()

                # liger fused jsd loss
                loss = self.liger_jsd_loss(
                    student_input=student_hidden,
                    student_weight=student_head.weight,
                    teacher_input=teacher_hidden,
                    teacher_weight=teacher_head.weight,
                    true_labels=true_labels,
                    student_bias=getattr(student_head, "bias", None),
                    teacher_bias=getattr(teacher_head, "bias", None),
                )

                # Release hidden states after loss computation
                del student_hidden, teacher_hidden, true_labels
            else:
                # Original behavior for same tokenizer or when teacher_tokenizer is not provided
                outputs_student = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                self.teacher_model.eval()
                with torch.no_grad():
                    outputs_teacher = self.teacher_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )

                prompt_lengths = inputs["prompts"].shape[1]
                shifted_student_logits = outputs_student.logits[:, prompt_lengths - 1 : -1, :]
                shifted_teacher_logits = outputs_teacher.logits[:, prompt_lengths - 1 : -1, :]
                shifted_labels = inputs["labels"][:, prompt_lengths:]
                loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta=self.beta,
                )

        if self.use_uld_loss:
            student_input_ids = inputs["input_ids"]

            # Use the *teacher* labels created above, not the student's.
            teacher_labels_for_loss = teacher_labels if "teacher_labels" in locals() else inputs["labels"]
            teacher_input_ids_for_loss = teacher_input_ids if "teacher_input_ids" in locals() else inputs["input_ids"]

            # Create properly masked student labels (fixing batch size > 1 issue)
            student_labels = inputs["labels"].clone()
            if hasattr(self.processing_class, "pad_token_id") and self.processing_class.pad_token_id is not None:
                student_labels[student_labels == self.processing_class.pad_token_id] = -100

            # Also mask pad tokens in teacher labels for consistency
            if (
                hasattr(self, "teacher_tokenizer")
                and hasattr(self.teacher_tokenizer, "pad_token_id")
                and self.teacher_tokenizer.pad_token_id is not None
            ):
                teacher_labels[teacher_labels == self.teacher_tokenizer.pad_token_id] = -100

            loss = self.uld_loss_fn(
                student_logits=outputs_student.logits,
                teacher_logits=outputs_teacher.logits,
                student_labels=student_labels,
                teacher_labels=teacher_labels_for_loss,
                student_input_ids=student_input_ids,
                teacher_input_ids=teacher_input_ids_for_loss,
            )

            # If ULD hybrid mode produced per-step matched/unmatched components, accumulate them for logging.
            # Use gradient_accumulation_steps to mirror Trainer's windowing behavior.
            if hasattr(self.uld_loss_fn, "last_matched_loss") and hasattr(self.uld_loss_fn, "last_unmatched_loss"):
                try:
                    ga = max(1, int(self.args.gradient_accumulation_steps))
                except Exception:
                    ga = 1
                step_eq = 1.0 / ga
                # read scalar values for logging
                matched_val = (
                    self.uld_loss_fn.last_matched_loss.item()
                    if self.uld_loss_fn.last_matched_loss is not None
                    else 0.0
                )
                unmatched_val = (
                    self.uld_loss_fn.last_unmatched_loss.item()
                    if self.uld_loss_fn.last_unmatched_loss is not None
                    else 0.0
                )

                self._matched_sum += matched_val
                self._unmatched_sum += unmatched_val
                self._matched_step_eq += step_eq
                self._unmatched_step_eq += step_eq

        empty_cache()

        return (loss, outputs_student) if return_outputs else loss

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        # Generate output with respect to the prompt only
        if self.use_transformers_paged:
            previous_attn = self.model.config._attn_implementation
            if is_flash_attn_2_available():
                model.config._attn_implementation = "paged_attention"
            else:
                model.config._attn_implementation = "sdpa_paged"
            prompt_mask = inputs.get("prompt_attention_mask")
            prompts_tensor = inputs["prompts"]
            if prompt_mask is not None:
                prompt_sequences = [
                    row[mask.bool()].detach().cpu().tolist()
                    for row, mask in zip(prompts_tensor, prompt_mask, strict=True)
                ]
            else:
                prompt_sequences = [row.detach().cpu().tolist() for row in prompts_tensor]
            generated_outputs = model.generate_batch(prompt_sequences, generation_config=generation_config)
            model.config._attn_implementation = previous_attn

            completion_ids = [output.generated_tokens for output in generated_outputs.values()]
            generated_tokens = torch.stack([torch.tensor(ids, device=model.device) for ids in completion_ids])
        else:
            generated_outputs = model.generate(
                input_ids=inputs["prompts"],
                attention_mask=inputs.get("prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences

        batch_size = generated_tokens.size(0)
        device = generated_tokens.device

        prompt_mask = inputs.get("prompt_attention_mask")
        pad_token_id = pad_token_id if pad_token_id is not None else self.processing_class.pad_token_id

        if prompt_mask is not None:
            prompt_lengths = prompt_mask.sum(dim=1).to(torch.long)
        else:
            if pad_token_id is not None:
                prompt_lengths = (inputs["prompts"] != pad_token_id).sum(dim=1).to(torch.long)
            else:
                prompt_lengths = torch.full(
                    (batch_size,),
                    inputs["prompts"].shape[1],
                    dtype=torch.long,
                    device=device,
                )

        new_input_ids = generated_tokens
        new_attention_mask = torch.ones_like(new_input_ids)
        if pad_token_id is not None:
            new_attention_mask[new_input_ids == pad_token_id] = 0

        new_labels = torch.full_like(new_input_ids, -100)
        for idx in range(batch_size):
            length = int(prompt_lengths[idx].item())
            new_labels[idx, length:] = new_input_ids[idx, length:]

        if pad_token_id is not None:
            new_labels[new_input_ids == pad_token_id] = -100

        prompt_texts = []
        completion_texts = []
        for idx in range(batch_size):
            length = int(prompt_lengths[idx].item())
            prompt_tokens = inputs["prompts"][idx]
            if prompt_mask is not None:
                prompt_tokens = prompt_tokens[prompt_mask[idx].bool()]
            elif pad_token_id is not None:
                prompt_tokens = prompt_tokens[prompt_tokens != pad_token_id]
            prompt_texts.append(
                self.processing_class.decode(
                    prompt_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )
            completion_tokens = new_input_ids[idx, length:]
            completion_texts.append(
                self.processing_class.decode(
                    completion_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        return new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        device = self.accelerator.device

        # Decode prompts for vLLM (without special tokens - vLLM expects clean text)
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=True,
            # clean_up_tokenization_spaces=False # Keep this commented unless specific issues arise
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm]

        # Also decode prompts WITH special tokens for ULD loss computation
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["prompts"],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )["completion_ids"]
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.vllm_guided_decoding_regex)
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full((padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Mask prompt tokens in labels
        prompt_lengths = prompt_ids.shape[1]
        new_labels[:, :prompt_lengths] = -100

        # IMPORTANT: Preserve original text for cross-tokenizer ULD loss
        # Use prompts_text_with_special (with special tokens) for ULD loss computation
        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])
            # Work around for https://github.com/vllm-project/vllm/issues/29341
            self.vllm_engine.collective_rpc("reload_weights")

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for the General Online Logit Distillation (GOLD) model.

        This method implements the on-policy learning approach described in the GOLD blog post. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the offline original inputs.
        """
        on_policy = False
        if random.random() <= self.lmbda:
            on_policy = True
            if self.use_vllm:
                self._wake_vllm_if_needed()
                result = self._generate_on_policy_outputs_vllm(
                    inputs, self.generation_config, self.processing_class.pad_token_id
                )
                new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts = result
            else:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    result = self.generate_on_policy_outputs(
                        unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                    )
                    new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts = result

            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = new_attention_mask
            inputs["labels"] = new_labels

            # CRITICAL: Preserve original text for cross-tokenizer ULD loss
            # This ensures both off-policy (dataset) and on-policy (generated) samples
            # can use proper text-based alignment for different tokenizers
            inputs["original_prompt_text"] = prompt_texts
            inputs["original_completion_text"] = completion_texts

            # Log prompt and completion texts
            self._textual_logs["prompt"].extend(gather_object(prompt_texts))
            self._textual_logs["completion"].extend(gather_object(completion_texts))

        loss = super().training_step(model, inputs, num_items_in_batch)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # include matched/unmatched accumulators for distributed reduction
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                    self._matched_sum,
                    self._unmatched_sum,
                    self._matched_step_eq,
                    self._unmatched_step_eq,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
                matched_sum,
                unmatched_sum,
                matched_eq,
                unmatched_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # matched/unmatched averaged over same logging window (if present)
            if matched_eq > 0:
                logs["matched_loss"] = round(matched_sum / matched_eq, 4)
            if unmatched_eq > 0:
                logs["unmatched_loss"] = round(unmatched_sum / unmatched_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0
            self._matched_sum = self._unmatched_sum = 0.0
            self._matched_step_eq = self._unmatched_step_eq = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            if is_rich_available():
                print_prompt_completions_sample_uld(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
