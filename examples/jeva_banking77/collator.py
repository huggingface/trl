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
class DataCollatorForJeva(DataCollatorForDecision):
    """
    Batch typed decisions with separate state and option tokenization.

    Each example contains a `request` (a dictionary or JSON string), or `state` and `questions`, with one question.
    Supply one of `target_index`, `target_probs`, `p_true`, or `score` as the target. The collator returns padded state
    inputs, flattened option inputs, an option mask, and target distributions with null probability first.

    For repeated training or evaluation, preprocess once with `dataset.map(collator.tokenize)`. The collator also
    accepts these prepared examples and only pads them when building batches.

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
        option_chunk_size (`int`, *optional*):
            Number of options encoded in one forward pass.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of tensor to return. Only `"pt"` is supported.
    """

    max_state_length: int = 9000
    max_option_length: int = 8192
    truncation: str = "error"
    option_chunk_size: int | None = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.max_state_length < 1 or self.max_option_length < 1:
            raise ValueError("max_state_length and max_option_length must be positive")
        if self.option_chunk_size is not None and self.option_chunk_size < 1:
            raise ValueError("option_chunk_size must be positive or None")
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
        return {
            "input_ids": sequences[0],
            "option_input_ids": sequences[1:],
            "labels": list(targets),
            "allow_null": asdict(question).get("allow_null", False),
            "is_choice": question.type == "choice",
        }

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" not in features[0]:
            features = [self.tokenize(example) for example in features]
        states = [example["input_ids"] for example in features]
        options = [option for example in features for option in example["option_input_ids"]]
        encoded = [
            dict(self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt", padding_side="right"))
            for sequences in (states, options)
        ]
        counts = torch.tensor([len(example["option_input_ids"]) for example in features])
        option_mask = torch.arange(counts.max())[None, :] < counts[:, None]
        labels = torch.zeros(len(features), option_mask.shape[1] + 1)
        for index, example in enumerate(features):
            labels[index, : len(example["labels"])] = torch.tensor(example["labels"])
        return {
            "option_chunk_size": self.option_chunk_size,
            **encoded[0],
            **{f"option_{key}": value for key, value in encoded[1].items()},
            "num_options": counts,
            "option_mask": option_mask,
            "labels": labels,
            "allow_null": torch.tensor([example["allow_null"] for example in features]),
            "is_choice": torch.tensor([example["is_choice"] for example in features]),
        }
