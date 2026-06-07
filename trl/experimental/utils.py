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

# This file contains utility classes and functions that are used across more than one experimental trainer or feature.

import inspect
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.types
import torch
from accelerate.utils import is_peft_model
from packaging.version import Version
from pyarrow import compute as pc
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.processing_utils import ProcessorMixin
from transformers.utils import (
    is_peft_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..data_utils import (
    DatasetType,
    _get_dataset_format,
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from ..trainer.utils import flush_left, pad


if is_peft_available():
    import peft
    from peft import PeftConfig, PeftModel, get_peft_model


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        is_encoder_decoder (`bool` or `None`, `optional`, defaults to `None`):
            Whether you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    is_encoder_decoder: bool | None = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = -100
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = -100
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


def pad_byte_offsets(offsets: list[tuple[int, int]], target_length: int, padding_side: str) -> torch.Tensor:
    """Build a ``[target_length, 2]`` long tensor from ``(start, end)`` byte-offset tuples,
    padding with ``(0, 0)`` on the requested side."""
    offs = torch.tensor(offsets, dtype=torch.long).reshape(-1, 2)
    pad_len = target_length - offs.size(0)
    if pad_len <= 0:
        return offs
    pad_block = torch.zeros(pad_len, 2, dtype=torch.long)
    return torch.cat([pad_block, offs], dim=0) if padding_side == "left" else torch.cat([offs, pad_block], dim=0)


def is_byte_level_tokenizer(backend) -> bool:
    """Whether ``backend`` is a ByteLevel BPE tokenizer (Llama-3 family, SmolLM, Qwen, \u2026) \u2014 its pieces are in
    byte\u2192unicode space, one char per source byte. Detected via the pre-tokenizer / decoder repr."""
    return "ByteLevel" in repr(backend.pre_tokenizer) or "ByteLevel" in repr(backend.decoder)


def piece_byte_len(piece: str) -> int:
    """UTF-8 byte length of a ByteLevel BPE token piece \u2014 each char maps 1:1 to one source byte.

    Cross-tokenizer ULD targets ByteLevel BPE pairs (Llama-3, Qwen, SmolLM, Phi, Mistral v0.3+, \u2026); SentencePiece
    students are out of scope here and would need the loss-level projection from X-Token to align."""
    return len(piece)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
    bs += list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


_BYTE_LEVEL_DECODER = {ch: b for b, ch in _bytes_to_unicode().items()}


def _byte_level_piece_len(piece: str, text_bytes: bytes, start: int) -> int | None:
    piece_bytes = []
    for ch in piece:
        if ch not in _BYTE_LEVEL_DECODER:
            return None
        piece_bytes.append(_BYTE_LEVEL_DECODER[ch])
    if piece_bytes and piece_bytes[0] == ord(" ") and text_bytes[start : start + 1] != b" ":
        piece_bytes = piece_bytes[1:]
    return len(piece_bytes)


def _split_repeated_byte_offsets(byte_offsets: list[tuple[int, int]], tokens: list[str]) -> list[tuple[int, int]]:
    """Split repeated char-derived spans for byte-fallback or byte-level tokens."""
    normalized = list(byte_offsets)
    i = 0
    while i < len(byte_offsets):
        j = i + 1
        while j < len(byte_offsets) and byte_offsets[j] == byte_offsets[i]:
            j += 1

        if j - i > 1:
            start, end = byte_offsets[i]
            piece_lengths = [piece_byte_len(token) for token in tokens[i:j]]
            if sum(piece_lengths) == end - start:
                cursor = start
                for offset_idx, length in enumerate(piece_lengths, start=i):
                    normalized[offset_idx] = (cursor, cursor + length)
                    cursor += length

        i = j
    return normalized


def _normalize_byte_offsets(
    byte_offsets: list[tuple[int, int]], tokens: list[str], text_bytes: bytes
) -> list[tuple[int, int]]:
    byte_offsets = _split_repeated_byte_offsets(byte_offsets, tokens)
    normalized = []
    cursor = 0

    for idx, (start, end) in enumerate(byte_offsets):
        if start == end:
            normalized.append((cursor, cursor))
            continue

        piece_len = _byte_level_piece_len(tokens[idx], text_bytes, start)
        next_start = byte_offsets[idx + 1][0] if idx + 1 < len(byte_offsets) else None
        has_overlap = start < cursor or (next_start is not None and next_start < end)

        if piece_len is not None and (has_overlap or piece_len == end - start):
            candidate_start = max(start, cursor)
            candidate_end = candidate_start + piece_len
            if candidate_end <= end:
                start = candidate_start
                end = candidate_end

        if start < cursor or end < start:
            raise ValueError(
                "Tokenizer produced overlapping byte offsets that could not be normalized. "
                "Cross-tokenizer ULD requires monotonic byte offsets."
            )

        normalized.append((start, end))
        cursor = end

    return normalized


def encode_with_byte_offsets(backend, texts: list[str], add_special_tokens: bool = False):
    """Encode ``texts`` and return per-text ``(ids, byte_offsets)`` pairs.

    Byte offsets are derived from the fast tokenizer's character offsets via an O(N) char-to-byte cumulative table.
    Overlapping spans from byte-level and byte-fallback tokens are split across their byte pieces."""
    if not is_byte_level_tokenizer(backend):
        raise NotImplementedError(
            "Cross-tokenizer ULD currently supports only ByteLevel BPE tokenizers "
            "(Llama-3, Qwen, SmolLM, Phi, Mistral v0.3+, …). The given tokenizer is not ByteLevel."
        )
    encs = backend.encode_batch(texts, add_special_tokens=add_special_tokens)
    out = []
    for text, enc in zip(texts, encs, strict=True):
        char_to_byte = [0]
        for ch in text:
            char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
        byte_offsets = [(char_to_byte[s], char_to_byte[e]) for s, e in enc.offsets]
        byte_offsets = _normalize_byte_offsets(byte_offsets, enc.tokens, text.encode("utf-8"))
        out.append((list(enc.ids), byte_offsets))
    return out


@dataclass
class DataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []
        byte_offsets: list[list[tuple[int, int]]] = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, example.get("original_prompt_text", None))
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, add_generation_prompt=False, tokenize=False
                )
                if is_byte_level_tokenizer(self.tokenizer.backend_tokenizer):
                    [(message_input_ids_full, full_offs)] = encode_with_byte_offsets(
                        self.tokenizer.backend_tokenizer, [formatted_message], add_special_tokens=False
                    )
                    prompt_byte_len = len(formatted_prompt.encode("utf-8"))
                    completion_start_idx_full = next(
                        (idx for idx, (start, _) in enumerate(full_offs) if start >= prompt_byte_len),
                        len(message_input_ids_full),
                    )
                else:
                    # Non-ByteLevel tokenizer: byte offsets are unnecessary (cross-tokenizer ULD requires ByteLevel
                    # anyway). Fall back to plain tokenization so GKD / non-ULD trainers with SentencePiece or
                    # Unigram tokenizers still work through this collator.
                    message_input_ids_full = self.tokenizer(
                        formatted_message, add_special_tokens=False, return_tensors=None
                    )["input_ids"]
                    completion_start_idx_full = len(
                        self.tokenizer(formatted_prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
                    )
                    full_offs = [(0, 0)] * len(message_input_ids_full)
                    prompt_byte_len = 0

                # Keep the last max_length tokens — drops oldest prompt context first,
                # never drops from the END (the model's recent context).
                if self.max_length is not None and len(message_input_ids_full) > self.max_length:
                    sample_ids = message_input_ids_full[-self.max_length :]
                    sample_offs = full_offs[-self.max_length :]
                    current_prompt_len = max(
                        0, completion_start_idx_full - (len(message_input_ids_full) - self.max_length)
                    )
                else:
                    sample_ids = message_input_ids_full
                    sample_offs = full_offs
                    current_prompt_len = completion_start_idx_full

                # Make completion-relative: prompt positions zeroed, completion offsets shifted. If truncation
                # ate into the completion (no prompt tokens kept and the first kept token is mid-completion),
                # rebase to byte 0 of the kept completion so teacher/student share the same coordinate system.
                kept_completion_offs = sample_offs[current_prompt_len:]
                base = (
                    kept_completion_offs[0][0] if kept_completion_offs and current_prompt_len == 0 else prompt_byte_len
                )
                completion_offs = [(s - base, e - base) for s, e in kept_completion_offs]
                sample_offs = [(0, 0)] * current_prompt_len + completion_offs

                input_ids.append(sample_ids)
                attention_mask.append([1] * len(sample_ids))
                current_prompt_ids = sample_ids[:current_prompt_len]
                byte_offsets.append(sample_offs)
            else:
                sample_ids = example["input_ids"]
                input_ids.append(sample_ids)
                attention_mask.append(example.get("attention_mask", [1] * len(sample_ids)))
                completion_mask = example.get("completion_mask")
                if completion_mask is not None:
                    # Use the tracked boundary directly: no re-tokenization, survives truncation.
                    prompt_len = completion_mask.index(1) if 1 in completion_mask else len(sample_ids)
                    current_prompt_ids = sample_ids[:prompt_len]
                else:
                    # No tracked boundary: tokenize the prompt and cap with a slice (avoid `truncation=True`,
                    # which would persist on the shared backend used by `encode_with_byte_offsets`).
                    tokenized_prompt = self.tokenizer(
                        formatted_prompt,
                        padding=False,
                        return_tensors=None,
                        add_special_tokens=False,
                    )
                    current_prompt_ids = tokenized_prompt["input_ids"][: len(sample_ids)]
                byte_offsets.append(example.get("byte_offsets", [(0, 0)] * len(sample_ids)))

            prompts_input_ids.append(current_prompt_ids)
            prompt_attention_mask.append([1] * len(current_prompt_ids))

            label = [self.ignore_index] * len(sample_ids)
            label[len(current_prompt_ids) :] = sample_ids[len(current_prompt_ids) :]
            labels.append(label)

        input_ids = pad(
            [torch.tensor(x, dtype=torch.long) for x in input_ids],
            padding_side="left",
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad(
            [torch.tensor(x, dtype=torch.long) for x in attention_mask], padding_side="left", padding_value=0
        )
        labels = pad(
            [torch.tensor(x, dtype=torch.long) for x in labels], padding_side="left", padding_value=self.ignore_index
        )
        prompts_input_ids = pad(
            [torch.tensor(x, dtype=torch.long) for x in prompts_input_ids],
            padding_side="left",
            padding_value=self.tokenizer.pad_token_id,
        )
        prompt_attention_mask = pad(
            [torch.tensor(x, dtype=torch.long) for x in prompt_attention_mask], padding_side="left", padding_value=0
        )

        target_len = input_ids.size(1)
        byte_offsets_tensor = torch.stack(
            [pad_byte_offsets(offs, target_len, padding_side="left") for offs in byte_offsets], dim=0
        )

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "byte_offsets": byte_offsets_tensor,
        }
        # Forward source text for cross-tokenizer ULD, when the dataset has it.
        if "original_prompt_text" in examples[0] and "original_completion_text" in examples[0]:
            out["original_prompt_text"] = [ex["original_prompt_text"] for ex in examples]
            out["original_completion_text"] = [ex["original_completion_text"] for ex in examples]
        return out


@dataclass
class DataCollatorForVisionLanguageChatML(DataCollatorMixin):
    """
    Data collator for GOLD VLM training.

    Combines image processing from [`~trainer.sft_trainer.DataCollatorForVisionLanguageModeling`] with the
    prompt-separation logic that GOLD needs for on-policy generation. Each input example should be a dictionary
    containing at least:
    - An `"images"` key holding a list of images, or an `"image"` key holding a single image.
    - Keys `"prompt"` and `"completion"` for the prompt and completion (conversational or plain text).

    The collator outputs a dictionary including:
    - `"input_ids"`: Tensor of token IDs (prompt + completion, concatenated).
    - `"attention_mask"`: Tensor indicating attention mask.
    - `"labels"`: Tensor for training labels (prompt tokens masked with -100).
    - `"prompts"`: Tensor of prompt-only token IDs (left-padded), used for on-policy generation.
    - `"prompt_attention_mask"`: Attention mask for prompts.
    - `"original_prompt_text"`: List of raw prompt text strings, used for ULD cross-tokenizer distillation.
    - `"original_completion_text"`: List of raw completion text strings, used for ULD cross-tokenizer distillation.
    - `"pixel_values"`: Tensor representing image pixel values.

    Additional keys may be present depending on the processor, such as `"image_grid_thw"` or `"image_position_ids"`.

    Args:
        processor ([`~transformers.ProcessorMixin`]):
            The processor used to tokenize text and process images.
        max_length (`int` or `None`, *optional*):
            Maximum sequence length for input tokens. If `None`, no truncation is applied.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The tensor type to return.
    """

    processor: ProcessorMixin
    max_length: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if "prompt" not in examples[0] or "completion" not in examples[0]:
            raise KeyError(
                "DataCollatorForVisionLanguageChatML requires 'prompt' and 'completion' keys in examples. "
                f"Got keys: {list(examples[0].keys())}."
            )

        # Normalize single image to list
        if "image" in examples[0]:
            for example in examples:
                example["images"] = [example.pop("image")]
        images = [example.get("images", []) for example in examples]
        if all(img_list == [] for img_list in images):
            images = None

        # Capture raw prompt/completion text before `apply_chat_template` mutates the examples.
        def _raw_text_from_messages(messages_or_str: Any) -> str:
            if isinstance(messages_or_str, str):
                return messages_or_str
            parts: list[str] = []
            for turn in messages_or_str:
                content = turn.get("content", "")
                if isinstance(content, str):
                    if content:
                        parts.append(content)
                    continue
                turn_parts: list[str] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            turn_parts.append(text)
                    elif isinstance(block, str):
                        turn_parts.append(block)
                if turn_parts:
                    parts.append("\n".join(turn_parts))
            return "\n".join(parts)

        raw_prompt_texts = [_raw_text_from_messages(example["prompt"]) for example in examples]
        raw_completion_texts = [_raw_text_from_messages(example["completion"]) for example in examples]

        # Apply chat template for conversational data
        if is_conversational(examples[0]):
            for example in examples:
                example["prompt"] = prepare_multimodal_messages(example["prompt"], images=example["images"])
                example["completion"] = prepare_multimodal_messages(example["completion"])
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        # Process prompts (with images) and completions (text only) separately
        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )

        # Concatenate prompts and completions
        prompt_ids, prompt_mask = (
            processed_prompts["input_ids"],
            processed_prompts["attention_mask"],
        )
        completion_ids, completion_mask = (
            processed_completions["input_ids"],
            processed_completions["attention_mask"],
        )
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
        if "token_type_ids" in processed_prompts:
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions.get("token_type_ids", torch.zeros_like(completion_ids))
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)
        if "mm_token_type_ids" in processed_prompts:
            prompt_mm_token_type_ids = processed_prompts["mm_token_type_ids"]
            completion_mm_token_type_ids = processed_completions.get(
                "mm_token_type_ids", torch.zeros_like(completion_ids)
            )
            mm_token_type_ids = torch.cat((prompt_mm_token_type_ids, completion_mm_token_type_ids), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts and "mm_token_type_ids" in processed_prompts:
            (
                attention_mask,
                input_ids,
                completion_mask,
                token_type_ids,
                mm_token_type_ids,
            ) = flush_left(
                attention_mask,
                input_ids,
                completion_mask,
                token_type_ids,
                mm_token_type_ids,
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

        # Create labels: mask padding and prompt tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[completion_mask == 0] = -100

        # Completion-relative byte offsets for cross-tokenizer ULD. Computed from the final input_ids/labels
        # via per-token piece byte length (the same primitive the on-policy / teacher path uses), so it is
        # layout-agnostic after flush_left/truncation and shares the teacher's byte coordinate system.
        tokenizer = self.processor.tokenizer
        piece_len_cache: dict[int, int] = {}
        byte_offset_rows: list[list[tuple[int, int]]] = []
        for row_ids, row_labels in zip(input_ids.tolist(), labels.tolist(), strict=True):
            offs: list[tuple[int, int]] = [(0, 0)] * len(row_ids)
            cumulative = 0
            for pos, (tid, label) in enumerate(zip(row_ids, row_labels, strict=True)):
                if label == -100:
                    continue
                if tid not in piece_len_cache:
                    piece_len_cache[tid] = piece_byte_len(tokenizer.convert_ids_to_tokens([tid])[0])
                nb = piece_len_cache[tid]
                offs[pos] = (cumulative, cumulative + nb)
                cumulative += nb
            byte_offset_rows.append(offs)
        byte_offsets = torch.tensor(byte_offset_rows, dtype=torch.long)

        # Build output with non-sequence vision keys from processed_prompts (pixel_values, image_grid_thw, etc.).
        output = {
            k: v
            for k, v in processed_prompts.items()
            if k not in ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")
        }
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        output["byte_offsets"] = byte_offsets
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids
        if (
            "mm_token_type_ids" in processed_prompts
        ):  # special case for ERNIE-VL from class DataCollatorForVisionLanguageModeling(DataCollatorMixin):
            output["mm_token_type_ids"] = mm_token_type_ids

        # GOLD-specific: separate prompt tensors for on-policy generation
        output["prompts"] = prompt_ids
        output["prompt_attention_mask"] = prompt_mask

        # GOLD-specific: raw text for ULD cross-tokenizer distillation.
        # These must be the untemplated text (no student chat-template markers) so the
        # teacher can re-render the prompt through its own chat template and tokenize the
        # completion cleanly.
        output["original_prompt_text"] = raw_prompt_texts
        output["original_completion_text"] = raw_completion_texts

        return output


def truncate_right(
    input_ids: torch.Tensor, stop_token_id: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates the input tensor from the right side after the first occurrence of the stop token.

    Args:
        input_ids (`torch.Tensor`):
            The tensor containing the responses to be truncated
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses

    Returns:
        tuple:
            - `output_ids` (`torch.Tensor`):
                The truncated responses tensor with pad tokens filled after the stop token
            - `mask` (`torch.Tensor`):
                The mask tensor to indicate the padding tokens
    """
    trunc_idxs = first_true_indices(input_ids == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(input_ids.size()) - 1) + [input_ids.shape[1]]
    idxs = torch.arange(input_ids.shape[1], device=input_ids.device).view(*new_size)
    output_ids = torch.masked_fill(input_ids, idxs > trunc_idxs, pad_token_id)
    mask = torch.masked_fill(torch.ones_like(input_ids), idxs > trunc_idxs, 0)
    return output_ids, mask


def add_bos_token_if_needed(
    bos_token_id: int | None,
    prompt_len_input_ids: int,
    prompt_tokens: dict[str, list[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: dict[str, list[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: dict[str, list[int]],
):
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int, chosen_tokens: dict[str, list[int]], rejected_tokens: dict[str, list[int]]
):
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    return chosen_tokens, rejected_tokens


def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving the position of the
    first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True in each row. If no True
            value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward logits and the rewards for a given model and query responses.

    Args:
        model (`torch.nn.Module`):
            The model used to compute the reward logits.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for the reward model.
            - `final_rewards` (`torch.Tensor`):
                The final rewards for each query response.
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    r"""
    Prepare a k-bit quantized transformers model for training (PEFT/QLoRA).
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    quant_methods = ["gptq", "aqlm", "eetq", "torchao", "hqq"]
    is_quantized = getattr(model, "quantization_method", None) in quant_methods or getattr(
        model, "hqq_quantized", False
    )

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for _, param in model.named_parameters():
        # freeze all parameters
        param.requires_grad = False

    # Enable gradient checkpointing if needed
    if (loaded_in_kbit or is_quantized) and use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # backward-compatible hook
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )
        gc_kwargs = {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs} if supports_gc_kwargs else {}
        model.gradient_checkpointing_enable(**gc_kwargs)

    return model


def enable_gradient_checkpointing(
    model: PreTrainedModel, gradient_checkpointing_kwargs: dict | None
) -> PreTrainedModel:
    """Enables gradient checkpointing for the model."""
    # Enable gradient checkpointing on the base model for PEFT
    if is_peft_model(model):
        model.base_model.gradient_checkpointing_enable()
    # Enable gradient checkpointing for non-PEFT models
    else:
        model.gradient_checkpointing_enable()

    gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
    use_reentrant = (
        "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
    )

    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def prepare_peft_model(
    model: PreTrainedModel, peft_config: "PeftConfig | None", args: TrainingArguments
) -> PreTrainedModel:
    """Prepares a model for PEFT training."""
    if not is_peft_available():
        raise ImportError("PEFT is required to use a peft model. Run `pip install peft`.")

    if isinstance(model, PeftModel) and peft_config is not None:
        raise ValueError(
            "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge and "
            "unload the existing adapter, save the resulting base model, and then pass that base model along with the "
            "new `peft_config` to the trainer."
        )

    # Handle quantized models (QLoRA)
    is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

    is_sharded_qlora = False
    if getattr(model, "is_loaded_in_4bit", False):
        # Check if model is sharded (FSDP/DS-Zero3)
        for _, param in model.named_parameters():
            if param.__class__.__name__ == "Params4bit":
                is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                break

    # Prepare model for kbit training if needed
    if is_qlora and not is_sharded_qlora and not isinstance(model, PeftModel):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs or {},
        )
        # Disable gradient checkpointing as it's handled by prepare_model_for_kbit_training
        args.gradient_checkpointing = False
    elif args.gradient_checkpointing:
        model = enable_gradient_checkpointing(model, args.gradient_checkpointing_kwargs)

    # Create PEFT model
    if peft_config is not None:
        if (
            Version(peft.__version__) >= Version("0.12")  # autocast_adapter_dtype introduced in 0.12
            and getattr(model, "is_loaded_in_4bit", False)
            and is_sharded_qlora
        ):
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(model, peft_config)

    # Handle bf16 casting for 4-bit models
    if args.bf16 and getattr(model, "is_loaded_in_4bit", False) and not is_sharded_qlora:
        peft_module_casting_to_bf16(model)

    return model


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float, dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (CUDA, MLU, MPS, NPU, XPU) and empties the
    cache of the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


def create_reference_model(
    model: nn.Module, num_shared_layers: int | None = None, pattern: str | None = None
) -> nn.Module:
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.

    Args:
        model ([`nn.Module`]): The model to be copied.
        num_shared_layers (`int`, *optional*):
            The number of initial layers that are shared between both models and kept frozen.
        pattern (`str`, *optional*): The shared layers are selected with a string pattern
            (e.g. "transformer.h.{layer}" for GPT2) and if a custom pattern is necessary it can be passed here.

    Returns:
        [`nn.Module`]
    """
    if is_deepspeed_zero3_enabled():
        raise ValueError(
            "DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()`. Please instantiate your reference model directly with `AutoModelForCausalLM.from_pretrained()`."
        )

    parameter_names = [n for n, _ in model.named_parameters()]
    ref_model = deepcopy(model)

    # if no layers are shared, return copy of model
    if num_shared_layers is None:
        for param_name in parameter_names:
            param = ref_model.get_parameter(param_name)
            param.requires_grad = False
        return ref_model.eval()

    # identify layer name pattern
    if pattern is not None:
        pattern = pattern.format(layer=num_shared_layers)
    else:
        for pattern_candidate in LAYER_PATTERNS:
            pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
            if any(pattern_candidate in name for name in parameter_names):
                pattern = pattern_candidate
                break

    if pattern is None:
        raise ValueError("Layer pattern could not be matched.")

    # divide parameters in shared and unshared parameter lists
    shared_param_list = []
    unshared_param_list = []

    shared_parameter = True
    for name, _param in model.named_parameters():
        if pattern in name:
            shared_parameter = False
        if shared_parameter:
            shared_param_list.append(name)
        else:
            unshared_param_list.append(name)

    # create reference of the original parameter if they are shared
    for param_name in shared_param_list:
        param = model.get_parameter(param_name)
        param.requires_grad = False

        _ref_param = ref_model.get_parameter(param_name)

    # for all other parameters just make sure they don't use gradients
    for param_name in unshared_param_list:
        param = ref_model.get_parameter(param_name)
        param.requires_grad = False

    if pattern is not None and len(unshared_param_list) == 0:
        logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()


def truncate_dataset(
    dataset: DatasetType,
    max_length: int,
    map_kwargs: dict[str, Any] | None = None,
) -> DatasetType:
    r"""
    Truncate sequences in a dataset to a specified `max_length`.

    Args:
        dataset ([`~datasets.Dataset`] or [`~datasets.DatasetDict`]):
            Dataset to truncate.
        max_length (`int`):
            Maximum sequence length to truncate to.
        map_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the dataset's map method when truncating examples.

    Returns:
        [`~datasets.Dataset`] or [`~datasets.DatasetDict`]: The dataset with truncated sequences.

    Example:
    ```python
    >>> from datasets import Dataset

    >>> examples = {
    ...     "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
    ...     "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
    ... }
    >>> dataset = Dataset.from_dict(examples)
    >>> truncated_dataset = truncate_dataset(dataset, max_length=2)
    >>> truncated_dataset[:]
    {'input_ids': [[1, 2], [4, 5], [8]],
     'attention_mask': [[0, 1], [0, 0], [1]]}
    ```
    """
    if map_kwargs is None:
        map_kwargs = {}

    def truncate(examples):
        truncated_columns = []
        for column in examples.columns:
            if pyarrow.types.is_list(column.type) or pyarrow.types.is_large_list(column.type):
                column = pc.list_slice(column, 0, max_length)
            truncated_columns.append(column)
        return pa.Table.from_arrays(truncated_columns, names=examples.column_names)

    format = _get_dataset_format(dataset)
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(truncate, batched=True, **map_kwargs)
    dataset = dataset.with_format(**format)
    return dataset
