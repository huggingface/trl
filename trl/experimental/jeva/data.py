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

# Jeva: System One Models from First Principles
# Copyright 2026 Qi Li and Xinchao Wang.

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TrainingRecord:
    state: str
    question: str
    decision_type: str
    options: tuple[str, ...]
    target_probs: tuple[float, ...]
    allow_null: bool = False
    values: tuple[float, ...] | None = None
    positive_index: int | None = None


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def normalize_record(row: Mapping[str, Any]) -> TrainingRecord:
    kind = "choice" if row.get("decision_type") is None else row["decision_type"]
    if kind not in {"choice", "noul", "score"}:
        raise ValueError("decision_type must be choice, noul, or score")
    state, question = row.get("state"), row.get("question")
    if not isinstance(state, str) or not isinstance(question, str) or not question.strip():
        raise ValueError("state and nonempty question must be strings")
    options = row.get("options") or (["false", "true"] if kind == "noul" else None)
    if not isinstance(options, (list, tuple)) or not options:
        raise ValueError("options must be a nonempty list of strings")
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("every option must be a nonempty string")
    k = len(options)
    allow_null = False if row.get("allow_null") is None else row["allow_null"]
    if not isinstance(allow_null, bool):
        raise ValueError("allow_null must be a boolean")
    if kind != "choice" and allow_null:
        raise ValueError("allow_null is supported only for choice records")
    supplied = [
        name
        for name in ("target_index", "target_probs", "p_true", "score")
        if name in row and (row[name] is not None or name == "target_index")
    ]
    if len(supplied) > 1 and row.get("target_index") is None:
        supplied.remove("target_index")
    if len(supplied) != 1:
        raise ValueError("provide exactly one of target_index, target_probs, p_true, or score")
    label = supplied[0]
    values, positive_index = None, None
    if kind == "noul":
        if k != 2:
            raise ValueError("noul requires exactly two options")
        positive_index = 1 if row.get("positive_index") is None else row["positive_index"]
        if type(positive_index) is not int or positive_index not in (0, 1):
            raise ValueError("positive_index must be 0 or 1")
    if kind == "score":
        raw_values = row.get("values") if row.get("values") is not None else list(range(k))
        if not isinstance(raw_values, (list, tuple)) or len(raw_values) != k or k < 2:
            raise ValueError("score requires at least two options and one value per option")
        values = tuple(_finite(value, "values") for value in raw_values)
        if any(b <= a for a, b in zip(values, values[1:], strict=False)):
            raise ValueError("score values must be strictly increasing in option order")
    probs = [0.0] * (k + 1)
    if label == "target_probs":
        raw_probs = row[label]
        if not isinstance(raw_probs, (list, tuple)) or len(raw_probs) != k + 1:
            raise ValueError("target_probs must contain null probability followed by one per option")
        probs = [_finite(value, "target_probs") for value in raw_probs]
        if min(probs) < 0 or not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("target_probs must be nonnegative and sum to one")
        total = sum(probs)
        probs = [value / total for value in probs]
    elif label == "target_index":
        index = row[label]
        if index is None:
            if not allow_null:
                raise ValueError("a null target requires allow_null=true")
            probs[0] = 1.0
        elif type(index) is not int or not 0 <= index < k:
            raise ValueError("target_index must be a zero-based option index or null")
        else:
            probs[index + 1] = 1.0
    elif label == "p_true":
        if kind != "noul":
            raise ValueError("p_true is only valid for noul")
        probability = _finite(row[label], label)
        if not 0 <= probability <= 1:
            raise ValueError("p_true must be between zero and one")
        probs[positive_index + 1] = probability
        probs[2 - positive_index] = 1.0 - probability
    else:
        if kind != "score":
            raise ValueError("score is only valid for score records")
        score = _finite(row[label], label)
        if not values[0] <= score <= values[-1]:
            raise ValueError("score must fall within the supplied rubric values")
        for index, value in enumerate(values):
            if score <= value:
                if index == 0:
                    probs[1] = 1.0
                else:
                    fraction = (score - values[index - 1]) / (value - values[index - 1])
                    probs[index] = 1.0 - fraction
                    probs[index + 1] = fraction
                break
    if not allow_null and probs[0] != 0:
        raise ValueError("null target mass requires allow_null=true")
    return TrainingRecord(state, question, kind, tuple(options), tuple(probs), allow_null, values, positive_index)


@dataclass
class DataCollatorForJeva:
    """
    Batch typed decisions with separate state and option tokenization.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer used by the shared encoder. Padding is applied on the right.
        max_state_length (`int`, *optional*, defaults to `9000`):
            Maximum length of the state and question text.
        max_option_length (`int`, *optional*, defaults to `8192`):
            Maximum length of each option text.
        truncation (`str`, *optional*, defaults to `"error"`):
            Reject overflowing text (`"error"`) or truncate it (`"truncate"`).
    """

    tokenizer: PreTrainedTokenizerBase
    max_state_length: int = 9000
    max_option_length: int = 8192
    truncation: str = "error"

    def __call__(self, features):
        rows = [normalize_record(row) for row in features]
        states = [f"Decision type: {r.decision_type}\nQuestion: {r.question}\nState:\n{r.state}" for r in rows]
        options = [f"Option for {r.decision_type}: {option}" for r in rows for option in r.options]
        encoded = []
        for texts, limit in ((states, self.max_state_length), (options, self.max_option_length)):
            inputs = self.tokenizer(texts, padding=False, truncation=False)
            if self.truncation == "error" and any(len(ids) > limit for ids in inputs["input_ids"]):
                raise ValueError(
                    f"Input exceeds its {limit} token limit; increase the limit or set truncation='truncate'"
                )
            if self.truncation == "truncate":
                inputs = self.tokenizer(texts, padding=False, truncation=True, max_length=limit)
            encoded.append(dict(self.tokenizer.pad(inputs, padding=True, return_tensors="pt", padding_side="right")))
        counts = torch.tensor([len(r.options) for r in rows])
        option_mask = torch.arange(counts.max())[None, :] < counts[:, None]
        labels = torch.zeros(len(rows), option_mask.shape[1] + 1)
        for i, row in enumerate(rows):
            labels[i, : len(row.target_probs)] = torch.tensor(row.target_probs)
        return {
            "state_inputs": encoded[0],
            "option_inputs": encoded[1],
            "option_counts": counts,
            "option_mask": option_mask,
            "labels": labels,
            "allow_null": torch.tensor([r.allow_null for r in rows]),
            "is_choice": torch.tensor([r.decision_type == "choice" for r in rows]),
        }
