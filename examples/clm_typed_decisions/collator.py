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

from dataclasses import asdict, dataclass
from typing import Any

import torch

from trl.experimental.decision import DataCollatorForDecision


@dataclass
class DataCollatorForClm(DataCollatorForDecision):
    """
    Pack CLM questions with a shared pool of distinct candidates for contrastive training.

    Accepts the shared decision request schema with one question per row. Nullable decisions are unsupported.
    Preprocess with `dataset.map(collator.tokenize)` to render and tokenize each question once. Batching packs the
    states and unique candidates into one prompt and builds soft targets over the candidate pool. `option_mask`
    restricts reported probabilities to each question's own candidates; the model's contrastive loss uses the full
    pool. Candidate identity is determined by its token IDs after truncation.


    Args:
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer used to render and encode states and candidates.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
        max_state_length (`int`, *optional*, defaults to `2047`):
            Maximum state and question token length.
        max_option_length (`int`, *optional*, defaults to `2047`):
            Maximum candidate token length.
        truncation (`str`, *optional*, defaults to `"error"`):
            Reject overflowing text (`"error"`) or truncate it (`"truncate"`) using the tokenizer's truncation side.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of tensor to return. Only `"pt"` is supported.
    """

    max_state_length: int = 2047
    max_option_length: int = 2047
    truncation: str = "error"
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.max_state_length < 1 or self.max_option_length < 1:
            raise ValueError("max_state_length and max_option_length must be positive")
        if self.truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be error or truncate")

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render and tokenize one decision with the native Transformers template."""
        inputs, question, targets = self.prepare_example(example)
        marks = inputs["marks"]
        if len(marks) != len(question.options) + 1:
            raise ValueError("The decision template must mark one state span and one span per option")
        segments = inputs["input_ids"][0].split(torch.diff(marks, prepend=marks.new_zeros(1)).tolist())
        sequences = []
        for index, segment in enumerate(segments):
            limit = self.max_state_length if index == 0 else self.max_option_length
            if self.truncation == "error" and len(segment) > limit:
                raise ValueError(
                    f"Input exceeds its {limit} token limit; increase the limit or set truncation='truncate'"
                )
            segment = segment[-limit:] if self.tokenizer.truncation_side == "left" else segment[:limit]
            sequences.append(segment.tolist())
        features = {
            "input_ids": sequences[0],
            "option_input_ids": sequences[1:],
            "labels": list(targets),
            "allow_null": asdict(question).get("allow_null", False),
            "is_choice": question.type == "choice",
        }
        if features["allow_null"]:
            raise ValueError("CLM does not support nullable decisions")
        features["labels"] = features["labels"][1:]
        return features

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" not in features[0]:
            features = [self.tokenize(example) for example in features]
        pool, columns, sequences, counts = {}, [], [], []
        for example in features:
            sequences.append(example["input_ids"])
            row_columns, count = [], 0
            for ids in example["option_input_ids"]:
                key = tuple(ids)
                if key not in pool:
                    pool[key] = len(pool)
                    sequences.append(ids)
                    count += 1
                row_columns.append(pool[key])
            columns.append(row_columns)
            counts.append(count)
        labels = torch.zeros(len(features), len(pool))
        option_mask = torch.zeros_like(labels, dtype=torch.bool)
        for index, (example, row_columns) in enumerate(zip(features, columns, strict=True)):
            indices = torch.tensor(row_columns)
            labels[index].scatter_add_(0, indices, torch.tensor(example["labels"]))
            option_mask[index, indices] = True
        return {
            "input_ids": torch.tensor([[token for sequence in sequences for token in sequence]]),
            "marks": torch.tensor([[len(sequence) for sequence in sequences]]).cumsum(-1),
            "num_options": torch.tensor([counts]),
            "labels": labels,
            "option_mask": option_mask,
            "is_choice": torch.tensor([example["is_choice"] for example in features]),
        }
