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

import random
import textwrap
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from itertools import takewhile
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainerCallback
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.utils import is_datasets_available, is_liger_kernel_available, is_peft_available, is_rich_available

from ...data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset
from ...extras.profiling import profiling_decorator
from ...generation.vllm_generation import VLLMGeneration
from ...import_utils import is_vllm_available
from ...models import prepare_deepspeed
from ...models.utils import _ForwardRedirection, unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import RepeatSampler, create_model_from_path, disable_dropout_in_model, pad, split_tensor_dict
from ..utils import DataCollatorForChatML, empty_cache, encode_with_byte_offsets, pad_byte_offsets, piece_byte_len
from .gold_config import GOLDConfig


if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss


if is_peft_available():
    from peft import PeftConfig


if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


if is_wandb_available():
    import wandb


def print_prompt_completions_sample_uld(
    prompts: list[str],
    completions: list[str],
    step: int,
    num_samples: int = None,
) -> None:
    """
    Print out a sample of model completions to the console.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int` or `None`, *optional*, defaults to `None`):
            Number of random samples to display. If `None` (default), all items will be displayed.

    Example:
    ```python
    >>> from trl.experimental.gold.gold_trainer import print_prompt_completions_sample_uld

    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> print_prompt_completions_sample_uld(prompts, completions, 42)
    ╭─────────── Step 42 ───────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩ │
    │ │ The sky is │  blue.       │ │
    │ ├────────────┼──────────────┤ │
    │ │ The sun is │  in the sky. │ │
    │ └────────────┴──────────────┘ │
    ╰───────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample_uld` requires the `rich` library. Please install it with "
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize teacher prompts/completions and produce tensors ready for GOLD loss.

    Returns ``(input_ids, labels, attention_mask, byte_offsets)``. ``byte_offsets`` is a ``[batch, seq, 2]`` tensor of
    UTF-8 byte ``(start, end)`` for each token: prompt and padding positions are filled with ``(0, 0)``; completion
    tokens carry offsets relative to the corresponding ``completion_text``; the appended EOS gets ``(content_len,
    content_len)``. Byte offsets are derived from the fast tokenizer's char offsets via ``encode_with_byte_offsets``.
    """

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    backend = tokenizer.backend_tokenizer

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_encs = encode_with_byte_offsets(backend, completion_texts, add_special_tokens=False)

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    offsets_list: list[list[tuple[int, int]]] = []

    for prompt_ids, (enc_ids, enc_offs), completion_text in zip(
        prompt_token_ids, completion_encs, completion_texts, strict=True
    ):
        # Remove trailing EOS from prompt so completions can extend cleanly
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]

        completion_ids = list(enc_ids)
        completion_offs = list(enc_offs)
        content_len = len(completion_text.encode("utf-8"))

        sequence = list(prompt_ids) + completion_ids
        offsets = [(0, 0)] * len(prompt_ids) + completion_offs
        if eos_token_id is not None:
            sequence.append(eos_token_id)
            offsets.append((content_len, content_len))

        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones_like(seq_tensor))
        offsets_list.append(offsets)

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

    target_len = teacher_input_ids.size(1)
    teacher_byte_offsets = torch.stack(
        [pad_byte_offsets(offs, target_len, padding_side="right") for offs in offsets_list],
        dim=0,
    )

    return teacher_input_ids, teacher_labels, teacher_attention_mask, teacher_byte_offsets


class ULDLoss(nn.Module):
    """
    Universal Logit Distillation Loss.
    """

    def __init__(self, config: GOLDConfig, student_tokenizer=None, teacher_tokenizer=None, device=None):
        super().__init__()
        self.device = device
        self.crossentropy_weight = config.uld_crossentropy_weight
        self.distillation_weight = config.uld_distillation_weight
        self.student_temperature = config.uld_student_temperature
        self.teacher_temperature = config.uld_teacher_temperature
        self.skip_student_eos = config.uld_skip_student_eos
        self.skip_teacher_eos = config.uld_skip_teacher_eos
        self.use_extended_uld = config.use_extended_uld
        self.token_merge_strategy = config.uld_token_merge_strategy
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
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=None,
        teacher_byte_offsets=None,
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
            student_byte_offsets: Per-token UTF-8 byte offsets ``[batch, seq, 2]``
                from the data collator (relative to the rendered chat-template message). Required.
            teacher_byte_offsets: Per-sample list of completion-relative byte
                offsets (one list per batch item) from ``build_teacher_inputs_from_texts``. Required.

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
            student_logits,
            teacher_logits,
            student_labels,
            teacher_labels,
            student_input_ids,
            teacher_input_ids,
            student_byte_offsets=student_byte_offsets,
            teacher_byte_offsets=teacher_byte_offsets,
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

        max_matched_teacher_id = max(self._vocab_mapping.keys())
        self.mapping_tensor = torch.full((max_matched_teacher_id + 1,), -1, dtype=torch.long)  # -1 for unmapped ids
        for k, v in self._vocab_mapping.items():
            self.mapping_tensor[k] = v
        if self.device is not None:
            self.mapping_tensor = self.mapping_tensor.to(self.device)

    def _compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=None,
        teacher_byte_offsets=None,
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

            # The "bayesian" shift has no predictor logit for an answer span starting at index 0 (front-truncation can
            # drop the whole prompt). Skip that unscorable leading span on both sides to keep them on a shared span.
            if self.token_merge_strategy == "bayesian" and (student_start == 0 or teacher_start == 0):
                if self.use_extended_uld:
                    if student_byte_offsets is None or teacher_byte_offsets is None:
                        raise ValueError("Byte offsets are required when `use_extended_uld=True`.")
                    s_answer = student_byte_offsets[i, student_start : student_start + student_size].tolist()
                    t_answer = teacher_byte_offsets[i, teacher_start : teacher_start + teacher_size].tolist()
                    student_groups, teacher_groups = self._align_by_byte_offsets(s_answer, t_answer)
                    if not student_groups or not teacher_groups:
                        distillation_losses.append(student_logits[i].sum() * 0.0)
                        continue
                    # Drop the first aligned group pair (advance by the tokens it covered).
                    student_drop, teacher_drop = len(student_groups[0]), len(teacher_groups[0])
                else:
                    # Drop the first positional target from both sides.
                    student_drop = teacher_drop = 1

                student_start, student_size = student_start + student_drop, student_size - student_drop
                teacher_start, teacher_size = teacher_start + teacher_drop, teacher_size - teacher_drop
                if student_size <= 0 or teacher_size <= 0:
                    distillation_losses.append(student_logits[i].sum() * 0.0)
                    continue

            # Extract answer logits. "bayesian" starts one position earlier so probs[k] predicts token_ids[k].
            if self.token_merge_strategy == "bayesian":
                student_answer_logits = student_logits[i, student_start - 1 : student_start + student_size - 1]
                teacher_answer_logits = teacher_logits[i, teacher_start - 1 : teacher_start + teacher_size - 1]
            else:
                student_answer_logits = student_logits[i, student_start : student_start + student_size]
                teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size]

            # Convert to probabilities
            student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)

            # Pass actual input_ids so split-token groups can multiply conditional probabilities.
            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()

            if self.use_extended_uld:
                if student_byte_offsets is None or teacher_byte_offsets is None:
                    raise ValueError("Byte offsets are required when `use_extended_uld=True`.")

                # Both sides are completion-relative, so plain slicing gives a shared byte coordinate system.
                s_answer = student_byte_offsets[i, student_start : student_start + student_size].tolist()
                t_answer = teacher_byte_offsets[i, teacher_start : teacher_start + teacher_size].tolist()
                student_groups, teacher_groups = self._align_by_byte_offsets(s_answer, t_answer)
                # Drop degenerate pairs where either side is empty — e.g. teacher's trailing zero-width EOS at
                # ``(content_len, content_len)`` paired with an empty student group merges to a zero distribution
                # and inflates the loss (only reachable when ``skip_teacher_eos=False``).
                paired = [(sg, tg) for sg, tg in zip(student_groups, teacher_groups, strict=False) if sg and tg]
                student_groups = [sg for sg, _ in paired]
                teacher_groups = [tg for _, tg in paired]
                student_aligned = self._merge_probabilities_with_alignment_groups(
                    student_probs, student_groups, student_token_ids
                )
                teacher_aligned = self._merge_probabilities_with_alignment_groups(
                    teacher_probs, teacher_groups, teacher_token_ids
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

    @staticmethod
    def _align_by_byte_offsets(s_offsets, t_offsets):
        """
        Walk both byte-offset arrays, advancing the side whose current token ends earlier. A group closes when both
        sides reach the same byte boundary — the points where the two tokenizers agree on a split.
        """
        s_groups, t_groups = [], []
        s_start = t_start = s = t = 0
        n_s, n_t = len(s_offsets), len(t_offsets)
        while s < n_s and t < n_t:
            s_end, t_end = s_offsets[s][1], t_offsets[t][1]
            if s_end < t_end:
                s += 1
            elif s_end > t_end:
                t += 1
            else:
                s += 1
                t += 1
                s_groups.append(list(range(s_start, s)))
                t_groups.append(list(range(t_start, t)))
                s_start, t_start = s, t
        if s < n_s or t < n_t:
            s_groups.append(list(range(s_start, n_s)))
            t_groups.append(list(range(t_start, n_t)))
        return s_groups, t_groups

    def _merge_probabilities_with_alignment_groups(self, probs, alignment_groups, token_ids=None):
        """
        Merge probabilities based on alignment groups, using either the "observed" or "bayesian" strategy
        (`self.token_merge_strategy`).

        For a group merging tokens at positions [i, ..., i+k]:
        - "observed": multiply the marginal distribution at the FIRST position by the scalar conditional probabilities
          of the actual later tokens.
        - "bayesian": multiply the full distribution at the LAST position (conditioned on the actual prefix tokens) by
          the scalar probabilities of the actual earlier tokens, following the chain rule.

        Both produce an unnormalized distribution that preserves correct relative probabilities.

        Args:
            probs: Probability tensor [seq_len, vocab_size]
            alignment_groups: List of alignment groups (each group is a list of positions to merge)
            token_ids: Actual token IDs that were generated [seq_len]. REQUIRED when any group has
                      len(group) > 1. If None when multi-token groups exist, raises ValueError.

        Returns:
            Merged probability tensor [num_groups, vocab_size]

        Raises:
            ValueError: If token_ids is None when merging multi-token groups
        """
        if not alignment_groups:
            return probs

        # Create aligned tensor
        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device, dtype=probs.dtype)
        eps = 1e-8

        # Process each alignment group
        for group_idx, group in enumerate(alignment_groups):
            # Handle probability merging
            if len(group) > 1:
                # Multiple tokens map to this group - merge by multiplying in the scalar probabilities of the tokens
                if token_ids is None:
                    raise ValueError(
                        "token_ids must be provided when merging multi-token groups. "
                        "They are needed to extract the scalar probabilities of the actually generated tokens."
                    )

                if self.token_merge_strategy == "bayesian":
                    base_probs = probs[group[-1]]  # last position's full distribution
                    scalar_positions = group[:-1]
                else:
                    base_probs = probs[group[0]]  # first position's marginal distribution
                    scalar_positions = group[1:]

                # Multiply base_probs by the scalar probabilities of the actual tokens at scalar_positions
                conditional_prob_product = 1.0
                for idx in scalar_positions:
                    actual_token_id = token_ids[idx]
                    token_prob = probs[idx, actual_token_id].clamp_min(eps)
                    conditional_prob_product *= token_prob

                merged_probs = base_probs * conditional_prob_product
                aligned_probs[group_idx] = merged_probs

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
            student_matched_indices = self.mapping_tensor[teacher_matched_indices]
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
        self.model_revision = (args.model_init_kwargs or {}).get("revision")

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
                weight_hard_loss=0.0,
                weight_soft_loss=1.0,
            )
            self.use_liger_gkd_loss = True
            self._forward_redirection = _ForwardRedirection()

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GOLDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
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
            if args.teacher_model_revision is not None:
                init_kwargs.setdefault("revision", args.teacher_model_revision)
            init_kwargs.setdefault("trust_remote_code", args.trust_remote_code)
            teacher_model = create_model_from_path(teacher_model, **init_kwargs)
        self.use_uld_loss = args.use_uld_loss
        self.teacher_tokenizer = None
        if args.use_uld_loss and args.teacher_tokenizer_name_or_path is not None:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                args.teacher_tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
            )
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
            teacher_model.resize_token_embeddings(self.model.config.get_text_config().vocab_size)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.num_generations = args.num_generations

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        # Buffering for rollouts across gradient accumulation steps
        self._buffered_inputs = None
        self._buffered_on_policy = None
        self._buffered_text_logs = None
        self._step = 0

        # Hybrid ULD matched/unmatched accumulators (logged every step when ULD hybrid is used)
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        self.uld_loss_fn = None
        if self.use_uld_loss:
            self.uld_loss_fn = ULDLoss(
                config=args,
                student_tokenizer=processing_class,
                teacher_tokenizer=self.teacher_tokenizer,
                device=self.accelerator.device,
            )

        generation_kwargs = {
            "max_new_tokens": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "top_k": args.top_k,
            "pad_token_id": self.processing_class.pad_token_id,
        }
        self.generation_config = GenerationConfig(**generation_kwargs)
        # Keep training-specific generation kwargs to overwrite model's original generation config
        self.generation_kwargs = generation_kwargs
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
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
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
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length or args.max_length,
                max_num_seqs=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                repetition_penalty=getattr(args, "repetition_penalty", 1.0),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=getattr(args, "min_p", 0.0),
                max_completion_length=args.max_completion_length,
                logprobs=None,
            )
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -self.vllm_sync_frequency

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
            "byte_offsets",
            "completion_mask",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _get_train_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size * self.accelerator.num_processes,
            repeat_count=self.args.gradient_accumulation_steps,
            shuffle=True,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        """
        Override Trainer.get_train_dataloader to load one generation batch per optimizer window.

        The dataloader yields local batches of size `per_device_train_batch_size * gradient_accumulation_steps`. The
        `RepeatSampler` (with `repeat_count=gradient_accumulation_steps`) ensures each generation batch is sampled
        `gradient_accumulation_steps` times so Trainer's loop iterates the correct number of times. Only the first
        batch in each window triggers `_fill_buffer`; the rest are ignored by `_prepare_inputs`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        if not self.model.training:
            return generation_batch

        buffer_steps = self.args.gradient_accumulation_steps
        if self._step % buffer_steps == 0 or self._buffered_inputs is None:
            self._fill_buffer(generation_batch, buffer_steps)

        slice_idx = self._step % buffer_steps
        inputs = self._buffered_inputs[slice_idx]
        self._step += 1
        return inputs

    @staticmethod
    def _build_sequence_batch(
        new_input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        pad_token_id: int | None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build attention mask and labels from full sequences and prompt lengths."""
        prompt_lengths = prompt_lengths.to(device=new_input_ids.device, dtype=torch.long)
        positions = torch.arange(new_input_ids.shape[1], device=new_input_ids.device).unsqueeze(0)
        completion_mask = positions >= prompt_lengths.unsqueeze(1)

        if attention_mask is not None:
            new_attention_mask = attention_mask.to(device=new_input_ids.device, dtype=new_input_ids.dtype)
        else:
            new_attention_mask = torch.ones_like(new_input_ids)
            if pad_token_id is not None:
                new_attention_mask[new_input_ids == pad_token_id] = 0

        new_labels = torch.full_like(new_input_ids, -100)
        new_labels[completion_mask & new_attention_mask.bool()] = new_input_ids[
            completion_mask & new_attention_mask.bool()
        ]
        if attention_mask is None and pad_token_id is not None:
            new_labels[new_input_ids == pad_token_id] = -100

        return new_attention_mask, new_labels

    def _maybe_add_completion_byte_offsets(self, updated_slice: dict[str, torch.Tensor | Any]) -> None:
        """Attach completion-relative byte offsets to on-policy ULD batches.

        Derived from the sampled ids via ``piece_byte_len`` (no decode→re-encode round-trip).
        """
        if not (
            self.use_uld_loss
            and self.teacher_tokenizer is not None
            and self.uld_loss_fn is not None
            and self.uld_loss_fn.use_extended_uld
        ):
            return

        new_input_ids = updated_slice["input_ids"]
        new_labels = updated_slice["labels"]
        seq_len = new_input_ids.shape[1]

        rows: list[list[tuple[int, int]]] = []
        for row_ids, row_labels in zip(new_input_ids.cpu().tolist(), new_labels.cpu().tolist(), strict=True):
            offs: list[tuple[int, int]] = [(0, 0)] * seq_len
            cumulative = 0
            for pos, (tid, label) in enumerate(zip(row_ids, row_labels, strict=True)):
                if label == -100:
                    continue
                nb = piece_byte_len(self.processing_class.convert_ids_to_tokens([tid])[0])
                offs[pos] = (cumulative, cumulative + nb)
                cumulative += nb
            rows.append(offs)
        updated_slice["byte_offsets"] = torch.tensor(rows, dtype=torch.long, device=new_input_ids.device)

    @profiling_decorator
    def _fill_buffer(self, generation_batch: dict[str, torch.Tensor | Any], buffer_steps: int):
        slices = split_tensor_dict(generation_batch, buffer_steps)

        if self.accelerator.is_main_process:
            on_policy_flags = [random.random() <= self.lmbda for _ in range(buffer_steps)]
        else:
            on_policy_flags = [False] * buffer_steps

        on_policy_flags = broadcast_object_list(on_policy_flags, from_process=0)
        on_policy_indices = [i for i, flag in enumerate(on_policy_flags) if flag]

        self._buffered_inputs = [None] * buffer_steps
        self._buffered_on_policy = on_policy_flags
        self._buffered_text_logs = [None] * buffer_steps

        for i, flag in enumerate(on_policy_flags):
            if not flag:
                slice_inputs = slices[i]

                if (
                    self.use_uld_loss
                    and self.teacher_tokenizer is not None
                    and ("original_prompt_text" not in slice_inputs or "original_completion_text" not in slice_inputs)
                ):
                    raise ValueError(
                        "Off-policy batch missing 'original_prompt_text' or 'original_completion_text' fields. "
                        "Use the default DataCollatorForChatML (or a collator that emits these fields) so the "
                        "teacher tokenizer has source text to align against."
                    )
                if (
                    self.use_uld_loss
                    and self.teacher_tokenizer is not None
                    and self.uld_loss_fn.use_extended_uld
                    and "byte_offsets" not in slice_inputs
                ):
                    raise ValueError(
                        "Off-policy batch missing `byte_offsets`. Use the default DataCollatorForChatML or set "
                        "`use_extended_uld=False`."
                    )

                self._buffered_inputs[i] = slice_inputs

        if on_policy_indices:
            self._generate_on_policy_for_slices(slices, on_policy_indices)

    @profiling_decorator
    def _generate_on_policy_for_slices(
        self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]
    ):
        prompt_ids_list = []
        local_slice_indices = []
        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            prompt_attention_mask = slice_inputs.get("prompt_attention_mask")
            for prompt_idx, prompt in enumerate(slice_inputs["prompts"]):
                if prompt_attention_mask is not None:
                    prompt = prompt[prompt_attention_mask[prompt_idx].bool()]
                prompt_ids_list.append(prompt.tolist())
                local_slice_indices.append(slice_idx)

        prompts_text = self.processing_class.batch_decode(
            prompt_ids_list,
            skip_special_tokens=False,
        )

        if not self.use_vllm:
            self._generate_non_vllm_for_slices(slices, on_policy_indices)
            return

        if (
            self.state.global_step != self._last_vllm_sync_step
            and self.state.global_step >= self._last_vllm_sync_step + self.vllm_sync_frequency
        ):
            self.vllm_generation.sync_weights()
            self._last_vllm_sync_step = self.state.global_step

        _, completion_ids, _, _ = self.vllm_generation.generate(
            prompts=prompt_ids_list,
            images=None,
            num_generations=self.num_generations,
        )

        self._process_completions_to_buffer(
            slices,
            on_policy_indices,
            local_slice_indices,
            completion_ids,
            prompt_ids_list,
            prompts_text,
            self.generation_config.max_new_tokens,
        )

    def _generate_non_vllm_for_slices(self, slices: list[dict[str, torch.Tensor | Any]], on_policy_indices: list[int]):
        """Fallback generation without vLLM (uses model.generate per slice)."""
        with unwrap_model_for_generation(
            self.model,
            self.accelerator,
            generation_kwargs=self.generation_kwargs,
        ) as unwrapped_model:
            for slice_idx in on_policy_indices:
                slice_inputs = slices[slice_idx]
                result = self.generate_on_policy_outputs(
                    unwrapped_model,
                    slice_inputs,
                    self.generation_config,
                    self.processing_class.pad_token_id,
                )
                new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts = result

                updated_slice = dict(slice_inputs)
                updated_slice["input_ids"] = new_input_ids
                updated_slice["attention_mask"] = new_attention_mask
                updated_slice["labels"] = new_labels
                updated_slice["original_prompt_text"] = prompt_texts
                updated_slice["original_completion_text"] = completion_texts
                self._maybe_add_completion_byte_offsets(updated_slice)

                self._buffered_inputs[slice_idx] = updated_slice
                self._buffered_text_logs[slice_idx] = (prompt_texts, completion_texts)

    def _process_completions_to_buffer(
        self,
        slices: list[dict[str, torch.Tensor | Any]],
        on_policy_indices: list[int],
        local_slice_indices: list[int],
        completion_ids: list,
        prompt_ids_list: list[list[int]],
        prompts_text: list[str],
        max_completion_length: int,
    ):
        """
        Process vLLM completions and update buffered inputs for on-policy slices.
        """
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0

        slice_completions = {idx: [] for idx in on_policy_indices}
        slice_prompt_ids = {idx: [] for idx in on_policy_indices}
        slice_prompts = {idx: [] for idx in on_policy_indices}

        for i, slice_idx in enumerate(local_slice_indices):
            slice_completions[slice_idx].append(completion_ids[i])
            slice_prompt_ids[slice_idx].append(prompt_ids_list[i])
            slice_prompts[slice_idx].append(prompts_text[i])

        for slice_idx in on_policy_indices:
            slice_inputs = slices[slice_idx]
            completion_ids_for_slice = slice_completions[slice_idx]
            prompt_ids_for_slice = slice_prompt_ids[slice_idx]
            prompt_txts = slice_prompts[slice_idx]

            prompt_max_length = max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
            truncated_prompt_ids = []
            prompt_attention_masks = []
            truncation_side = getattr(self.processing_class, "truncation_side", "right")
            for prompt_ids in prompt_ids_for_slice:
                if prompt_max_length and len(prompt_ids) > prompt_max_length:
                    if truncation_side == "left":
                        prompt_ids = prompt_ids[-prompt_max_length:]
                    else:
                        prompt_ids = prompt_ids[:prompt_max_length]
                prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
                truncated_prompt_ids.append(prompt_tensor)
                prompt_attention_masks.append(torch.ones(len(prompt_ids), device=device, dtype=torch.long))

            prompt_ids = pad(truncated_prompt_ids, padding_side="left", padding_value=pad_token_id)
            prompt_attention_mask = pad(prompt_attention_masks, padding_side="left", padding_value=0)

            # Decode the truncated prompt so the teacher conditions on the same context the student saw.
            # `clean_up_tokenization_spaces=False` matches the completion decode below so byte counts stay aligned.
            prompt_txts_with_special = self.processing_class.batch_decode(
                [ids.tolist() for ids in truncated_prompt_ids],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids_for_slice]
            completion_ids_for_text: list[list[int]] = []
            padded_completion_ids_list = []
            completion_attention_masks = []
            for completion_tensor in completion_ids_tensors:
                if len(completion_tensor) > max_completion_length:
                    truncated_completion_tensor = completion_tensor[:max_completion_length]
                    padded_completion_ids_list.append(truncated_completion_tensor)
                    completion_ids_for_text.append(truncated_completion_tensor.tolist())
                    completion_attention_masks.append(
                        torch.ones(len(truncated_completion_tensor), device=device, dtype=torch.long)
                    )
                elif len(completion_tensor) < max_completion_length:
                    padding_needed = max_completion_length - len(completion_tensor)
                    padded_tensor = torch.cat(
                        [
                            completion_tensor,
                            torch.full(
                                (padding_needed,),
                                pad_token_id,
                                device=device,
                                dtype=completion_tensor.dtype,
                            ),
                        ]
                    )
                    padded_completion_ids_list.append(padded_tensor)
                    completion_ids_for_text.append(completion_tensor.tolist())
                    completion_attention_masks.append(
                        torch.cat(
                            [
                                torch.ones(len(completion_tensor), device=device, dtype=torch.long),
                                torch.zeros(padding_needed, device=device, dtype=torch.long),
                            ]
                        )
                    )
                else:
                    padded_completion_ids_list.append(completion_tensor)
                    completion_ids_for_text.append(completion_tensor.tolist())
                    completion_attention_masks.append(
                        torch.ones(len(completion_tensor), device=device, dtype=torch.long)
                    )

            completion_ids_padded = torch.stack(padded_completion_ids_list)
            completion_attention_mask = torch.stack(completion_attention_masks)

            new_input_ids = torch.cat([prompt_ids, completion_ids_padded], dim=1)
            new_attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
            prompt_lengths = torch.full((prompt_ids.shape[0],), prompt_ids.shape[1], device=device)
            new_attention_mask, new_labels = self._build_sequence_batch(
                new_input_ids,
                prompt_lengths,
                pad_token_id,
                attention_mask=new_attention_mask,
            )

            completion_texts = self.processing_class.batch_decode(
                completion_ids_for_text,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            updated_slice = dict(slice_inputs)
            updated_slice["input_ids"] = new_input_ids
            updated_slice["attention_mask"] = new_attention_mask
            updated_slice["labels"] = new_labels
            updated_slice["original_prompt_text"] = prompt_txts_with_special
            updated_slice["original_completion_text"] = completion_texts
            self._maybe_add_completion_byte_offsets(updated_slice)

            self._buffered_inputs[slice_idx] = updated_slice
            self._buffered_text_logs[slice_idx] = (prompt_txts, completion_texts)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Preserve original text fields for ULD when needed."""
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        if packing and self.use_uld_loss and self.teacher_tokenizer is not None:
            raise ValueError(
                "Packing is not supported with cross-tokenizer ULD because byte-offset alignment is defined per "
                "prompt/completion example."
            )

        if not is_processed or (self.use_uld_loss and self.teacher_tokenizer is not None):
            return self._prepare_dataset_with_original_text(
                dataset, processing_class, args, packing, formatting_func, dataset_name
            )

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

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            # Convert the dataset to ChatML if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            column_names = next(iter(dataset)).keys()
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
                    fn_kwargs={"eos_token": processing_class.eos_token},
                    remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                    **map_kwargs,
                )

            # Tokenize the dataset while preserving original text
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset (preserving original text)"

            def tokenize_with_original_text(example, processing_class, dataset_text_field, max_length):
                """Emit input_ids, attention_mask, byte_offsets, completion_mask, and the original prompt/completion
                text. Byte offsets and input_ids come from a single ``encode_with_byte_offsets`` call."""
                backend = processing_class.backend_tokenizer
                result = {}

                if "prompt" in example:  # prompt-completion case
                    if is_conversational(example):
                        prompt_text = processing_class.apply_chat_template(
                            example["prompt"],
                            add_generation_prompt=True,
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        full_text = processing_class.apply_chat_template(
                            example["prompt"] + example["completion"],
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_text = "".join(
                            x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_text, full_text, strict=False))
                        )
                        completion_text = full_text[len(prompt_text) :]
                    else:
                        prompt_text = example["prompt"]
                        completion_text = example["completion"]
                        full_text = prompt_text + completion_text
                    result["original_prompt_text"] = prompt_text
                    result["original_completion_text"] = completion_text
                elif is_conversational(example):
                    messages = example["messages"]
                    assistant_indices = [idx for idx, msg in enumerate(messages) if msg["role"] == "assistant"]
                    if assistant_indices:
                        completion_idx = assistant_indices[-1]
                        prompt_messages = messages[:completion_idx]
                        full_messages = messages[: completion_idx + 1]
                        if prompt_messages:
                            prompt_text = processing_class.apply_chat_template(
                                prompt_messages,
                                add_generation_prompt=True,
                                tokenize=False,
                                **example.get("chat_template_kwargs", {}),
                            )
                        else:
                            prompt_text = ""
                        full_text = processing_class.apply_chat_template(
                            full_messages,
                            add_generation_prompt=False,
                            tokenize=False,
                            **example.get("chat_template_kwargs", {}),
                        )
                        prompt_text = "".join(
                            x for x, _ in takewhile(lambda x: x[0] == x[1], zip(prompt_text, full_text, strict=False))
                        )
                        completion_text = full_text[len(prompt_text) :]
                        result["original_prompt_text"] = prompt_text
                        result["original_completion_text"] = completion_text
                    else:
                        full_text = processing_class.apply_chat_template(
                            messages, tokenize=False, **example.get("chat_template_kwargs", {})
                        )
                        prompt_text = ""
                        result["original_prompt_text"] = ""
                        result["original_completion_text"] = full_text
                else:
                    text = example.get(dataset_text_field, example.get("text", ""))
                    prompt_text = ""
                    full_text = text
                    result["original_prompt_text"] = ""
                    result["original_completion_text"] = text

                # Single backend call: ids and char-derived byte offsets from the same encoding,
                # so input_ids[i] is described by full_offs[i] without any boundary slop.
                [(input_ids, full_offs)] = encode_with_byte_offsets(backend, [full_text], add_special_tokens=False)
                prompt_byte_len = len(prompt_text.encode("utf-8"))
                completion_start = next(
                    (idx for idx, (s, _) in enumerate(full_offs) if s >= prompt_byte_len),
                    len(input_ids),
                )
                # Completion-relative: prompt positions zeroed, completion offsets shifted to
                # the assistant content's first byte (matches build_teacher_inputs_from_texts).
                byte_offsets = [(0, 0)] * completion_start + [
                    (s - prompt_byte_len, e - prompt_byte_len) for s, e in full_offs[completion_start:]
                ]

                # Keep the last `max_length` tokens (the completion end). `completion_mask` tracks the
                # boundary so it survives truncation without re-tokenizing the prompt.
                if max_length is not None and len(input_ids) > max_length:
                    drop = len(input_ids) - max_length
                    input_ids = input_ids[drop:]
                    byte_offsets = byte_offsets[drop:]
                    completion_start = max(0, completion_start - drop)
                    # If truncation ate into the completion, rebase the kept completion offsets so they're
                    # relative to the new (truncated) `original_completion_text` the teacher will re-encode.
                    if completion_start < len(byte_offsets):
                        base = byte_offsets[completion_start][0]
                        if base > 0:
                            byte_offsets = byte_offsets[:completion_start] + [
                                (s - base, e - base) for s, e in byte_offsets[completion_start:]
                            ]
                    # Resync the strings the teacher will re-encode with the ids the student kept.
                    decode = partial(
                        processing_class.decode, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    result["original_prompt_text"] = decode(input_ids[:completion_start])
                    result["original_completion_text"] = decode(input_ids[completion_start:])

                result["input_ids"] = input_ids
                result["attention_mask"] = [1] * len(input_ids)
                result["byte_offsets"] = byte_offsets
                result["completion_mask"] = [0] * completion_start + [1] * (len(input_ids) - completion_start)
                return result

            dataset = dataset.map(
                tokenize_with_original_text,
                fn_kwargs={
                    "processing_class": processing_class,
                    "dataset_text_field": args.dataset_text_field,
                    "max_length": args.max_length,
                },
                **map_kwargs,
            )

            # Pack if requested. Truncation already happened in `tokenize_with_original_text`, keeping
            # the completion end — so the generic front-truncating `truncate_dataset` is intentionally
            # not applied here (it would drop the completion).
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

            if args.use_liger_kernel:
                required_columns = {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "completion_mask",
                    "messages",
                    "original_prompt_text",
                    "original_completion_text",
                    "byte_offsets",
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
        num_items_in_batch=None,
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
        if num_items_in_batch is not None:
            # Normalize by the global number of valid tokens for gradient-accumulation-correct loss (see issue #4719).
            jsd_sum = jsd.sum()
            if isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
            return jsd_sum / num_items_in_batch
        if reduction == "batchmean":
            # clamp_min(1) avoids 0/0 -> nan when a sample has no unmasked positions
            # (e.g. completion fully truncated). jsd[mask] is empty -> jsd.sum() == 0,
            # so 0/1 == 0 with a valid grad path.
            denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
            return jsd.sum() / denom
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_uld_loss and self.teacher_tokenizer is not None:
            # Both DataCollatorForChatML and the on-policy generation path attach these
            # fields, so cross-tokenizer ULD never has to round-trip through batch_decode.
            prompt_texts = inputs["original_prompt_text"]
            completion_texts = inputs["original_completion_text"]

            (
                teacher_input_ids,
                teacher_labels,
                teacher_attention_mask,
                teacher_completion_byte_offsets,
            ) = build_teacher_inputs_from_texts(self.teacher_tokenizer, prompt_texts, completion_texts)

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
        else:
            if self.use_liger_gkd_loss:
                # Forward only through the base models (avoid lm_head to save memory).
                # Route through the DDP/FSDP wrapper via _forward_redirection so that
                # DDP.forward() is called and prepare_for_backward() fires correctly.
                unwrapped_student = self.accelerator.unwrap_model(model)
                student_outputs = self._forward_redirection(
                    model, unwrapped_student, self._liger_student_forward, unwrapped_student, inputs
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

                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                del student_outputs, teacher_outputs

                student_hidden = student_hidden.reshape(-1, student_hidden.shape[-1])
                teacher_hidden = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])

                labels_mask = inputs["labels"] != -100
                masked_input_ids = torch.where(
                    labels_mask, inputs["input_ids"], torch.full_like(inputs["input_ids"], -100)
                )
                true_labels = masked_input_ids[:, 1:].reshape(-1)

                student_head = unwrapped_student.get_output_embeddings()
                teacher_head = unwrapped_teacher.get_output_embeddings()

                loss = self.liger_jsd_loss(
                    student_input=student_hidden,
                    student_weight=student_head.weight,
                    teacher_input=teacher_hidden,
                    teacher_weight=teacher_head.weight,
                    true_labels=true_labels,
                    student_bias=getattr(student_head, "bias", None),
                    teacher_bias=getattr(teacher_head, "bias", None),
                )

                # The Liger JSD loss normalizes by the local number of valid tokens. Under gradient accumulation we
                # want the global normalization, so rescale by `num_valid_local / num_items_in_batch`.
                if num_items_in_batch is not None:
                    num_valid_local = (true_labels != -100).sum().clamp_min(1)
                    if isinstance(num_items_in_batch, torch.Tensor):
                        num_items_in_batch = num_items_in_batch.to(loss.device)
                    loss = loss * num_valid_local / num_items_in_batch

                del student_hidden, teacher_hidden, true_labels
            else:
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

                # Standard causal shift: logits at position i predict the token at i + 1. The `labels != -100` mask
                # inside `generalized_jsd_loss` already excludes prompt (and padding) positions, so we do not slice by
                # prompt length. Slicing by `inputs["prompts"].shape[1]` (the batch-max prompt width) would drop real
                # completion tokens for samples whose prompt is shorter than the batch maximum, since `labels` is
                # padded to the full-sequence width independently of `prompts`.
                shifted_student_logits = outputs_student.logits[:, :-1, :]
                shifted_teacher_logits = outputs_teacher.logits[:, :-1, :]
                shifted_labels = inputs["labels"][:, 1:]
                loss = self.generalized_jsd_loss(
                    student_logits=shifted_student_logits,
                    teacher_logits=shifted_teacher_logits,
                    labels=shifted_labels,
                    beta=self.beta,
                    temperature=self.temperature,
                    num_items_in_batch=num_items_in_batch,
                )

        if self.use_uld_loss and self.teacher_tokenizer is not None:
            student_labels = inputs["labels"].clone()
            if self.processing_class.pad_token_id is not None:
                student_labels[student_labels == self.processing_class.pad_token_id] = -100
            if self.teacher_tokenizer.pad_token_id is not None:
                teacher_labels[teacher_labels == self.teacher_tokenizer.pad_token_id] = -100

            student_byte_offsets = inputs.get("byte_offsets")
            if self.uld_loss_fn.use_extended_uld and student_byte_offsets is None:
                raise ValueError("Input batches must include `byte_offsets` when `use_extended_uld=True`.")

            loss = self.uld_loss_fn(
                student_logits=outputs_student.logits,
                teacher_logits=outputs_teacher.logits,
                student_labels=student_labels,
                teacher_labels=teacher_labels,
                student_input_ids=inputs["input_ids"],
                teacher_input_ids=teacher_input_ids,
                student_byte_offsets=student_byte_offsets,
                teacher_byte_offsets=teacher_completion_byte_offsets,
            )

            if hasattr(self.uld_loss_fn, "last_matched_loss") and hasattr(self.uld_loss_fn, "last_unmatched_loss"):
                ga = max(1, int(self.args.gradient_accumulation_steps))
                step_eq = 1.0 / ga
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

        # model.generate() returns full sequences (prompt + completion), so completions start
        # after the full padded prompt width.
        prompt_lengths = torch.full((batch_size,), inputs["prompts"].shape[1], dtype=torch.long, device=device)

        new_input_ids = generated_tokens
        new_attention_mask, new_labels = self._build_sequence_batch(new_input_ids, prompt_lengths, pad_token_id)

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
            completion_tokens = completion_tokens[new_labels[idx, length:] != -100]
            completion_texts.append(
                self.processing_class.decode(
                    completion_tokens.tolist(),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            )

        return new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts

    def _liger_student_forward(self, student, inputs):
        """Decoder-only forward used by the Liger JSD path (skips lm_head to save memory)."""
        if hasattr(student, "get_decoder") and student.get_decoder() is not None:
            decoder = student.get_decoder()
        else:
            decoder = getattr(student, getattr(student, "base_model_prefix", "model"), student)
        return decoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )

    def _get_liger_zero3_lm_head_gather_ctx(self, model: nn.Module):
        if not self.use_liger_gkd_loss:
            return nullcontext()

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None or deepspeed_plugin.zero_stage != 3:
            return nullcontext()

        import deepspeed

        unwrapped_student = self.accelerator.unwrap_model(model)
        unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
        student_head = unwrapped_student.get_output_embeddings()
        teacher_head = unwrapped_teacher.get_output_embeddings()
        params = [student_head.weight, teacher_head.weight]
        if student_head.bias is not None:
            params.append(student_head.bias)
        if teacher_head.bias is not None:
            params.append(teacher_head.bias)
        return deepspeed.zero.GatheredParameters(params, modifier_rank=None)

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
        buffer_steps = self.args.gradient_accumulation_steps

        # Keep lm_head gathered across forward+backward for Liger + ZeRO-3.
        with self._get_liger_zero3_lm_head_gather_ctx(model):
            loss = super().training_step(model, inputs, num_items_in_batch)

        slice_idx = (self._step - 1) % buffer_steps

        on_policy = False
        if self._buffered_on_policy is not None and slice_idx < len(self._buffered_on_policy):
            on_policy = self._buffered_on_policy[slice_idx]

        if on_policy and self._buffered_text_logs is not None and self._buffered_text_logs[slice_idx] is not None:
            prompt_texts, completion_texts = self._buffered_text_logs[slice_idx]
            self._textual_logs["prompt"].extend(gather_object(prompt_texts))
            self._textual_logs["completion"].extend(gather_object(completion_texts))

        loss_scalar = float(loss.detach())
        step_equiv = 1.0 / self.args.gradient_accumulation_steps

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

            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            if matched_eq > 0:
                logs["matched_loss"] = round(matched_sum / matched_eq, 4)
            if unmatched_eq > 0:
                logs["unmatched_loss"] = round(unmatched_sum / unmatched_eq, 4)

            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0
            self._matched_sum = self._unmatched_sum = 0.0
            self._matched_step_eq = self._unmatched_step_eq = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs.update(metrics)
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
                    "step": [self.state.global_step] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
