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
from transformers.decision import DecisionConfig, Question, read_marks


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def normalize_target(row: Mapping[str, Any], question: Question) -> tuple[float, ...]:
    kind = question.type
    k = len(question.options)
    allow_null = question.allow_null
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
        positive_index = question.options.index("true")
    if kind == "score":
        values = tuple(range(k)) if question.values is None else tuple(question.values)
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
    return tuple(probs)


@dataclass
class DataCollatorForJeva:
    """
    Batch typed decisions with separate state and option tokenization.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer used by the shared encoder. Padding is applied on the right.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
        max_state_length (`int`, *optional*, defaults to `9000`):
            Maximum length of the state and question text.
        max_option_length (`int`, *optional*, defaults to `8192`):
            Maximum length of each option text.
        truncation (`str`, *optional*, defaults to `"error"`):
            Reject overflowing text (`"error"`) or truncate it (`"truncate"`).
    """

    tokenizer: PreTrainedTokenizerBase
    decision_config: DecisionConfig
    max_state_length: int = 9000
    max_option_length: int = 8192
    truncation: str = "error"

    def __post_init__(self):
        if self.max_state_length < 1 or self.max_option_length < 1:
            raise ValueError("max_state_length and max_option_length must be positive")
        if self.truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be error or truncate")

    def __call__(self, features):
        states, options, targets, questions = [], [], [], []
        for row in features:
            request = row["request"] if "request" in row else {"state": row["state"], "questions": row["questions"]}
            prompts = self.decision_config.chat_template_kwargs(request, self.tokenizer)
            if len(prompts) != 1 or len(prompts[0]["decision_questions"]) != 1:
                raise ValueError("Each training row must contain exactly one decision question")
            kwargs = prompts[0]
            question = kwargs.pop("decision_questions")[0]
            inputs = self.tokenizer.apply_chat_template(**kwargs, tokenize=True, return_dict=True, return_tensors="pt")
            inputs, marks = read_marks(dict(inputs), self.decision_config._sentinel_ids)
            if len(marks) != len(question.options) + 1:
                raise ValueError("The Jeva template must mark one state span and one span per option")
            segments = inputs["input_ids"][0].split(torch.diff(marks, prepend=marks.new_zeros(1)).tolist())
            states.append(segments[0].tolist())
            options.extend(segment.tolist() for segment in segments[1:])
            targets.append(normalize_target(row, question))
            questions.append(question)
        encoded = []
        for sequences, limit in ((states, self.max_state_length), (options, self.max_option_length)):
            if self.truncation == "error" and any(len(ids) > limit for ids in sequences):
                raise ValueError(
                    f"Input exceeds its {limit} token limit; increase the limit or set truncation='truncate'"
                )
            if self.truncation == "truncate":
                sequences = [ids[:limit] for ids in sequences]
            encoded.append(
                dict(
                    self.tokenizer.pad(
                        {"input_ids": sequences}, padding=True, return_tensors="pt", padding_side="right"
                    )
                )
            )
        counts = torch.tensor([len(question.options) for question in questions])
        option_mask = torch.arange(counts.max())[None, :] < counts[:, None]
        labels = torch.zeros(len(questions), option_mask.shape[1] + 1)
        for index, target in enumerate(targets):
            labels[index, : len(target)] = torch.tensor(target)
        return {
            **encoded[0],
            **{f"option_{key}": value for key, value in encoded[1].items()},
            "num_options": counts,
            "option_mask": option_mask,
            "labels": labels,
            "allow_null": torch.tensor([question.allow_null for question in questions]),
            "is_choice": torch.tensor([question.type == "choice" for question in questions]),
        }
