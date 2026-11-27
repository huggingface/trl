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

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.decision import DecisionConfig as TransformersDecisionConfig
from transformers.decision import Question


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _normalize_target(row: Mapping[str, Any], question: Question) -> tuple[float, ...]:
    kind = question.type
    k = len(question.options)
    metadata = asdict(question)
    allow_null = metadata.get("allow_null", False)

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

    probs = [0.0] * (k + 1)
    if label == "target_probs":
        raw_probs = row[label]
        if isinstance(raw_probs, Mapping):
            # Arrow structs fill absent keys with None when rows have different options.
            named_probs = {option: value for option, value in raw_probs.items() if value is not None}
            if named_probs.keys() - set(question.options):
                raise ValueError("target_probs contains unknown option names")
            raw_probs = [0.0, *(named_probs.get(option, 0.0) for option in question.options)]
        elif not isinstance(raw_probs, (list, tuple)) or len(raw_probs) != k + 1:
            raise ValueError(
                "target_probs must be keyed by option name or contain null probability followed by one per option"
            )
        probs = [_finite(value, "target_probs") for value in raw_probs]
        total = sum(probs)
        if min(probs) < 0 or not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("target_probs must be nonnegative and sum to one")
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
        positive_index = question.options.index("true")
        probs[positive_index + 1] = probability
        probs[2 - positive_index] = 1.0 - probability
    else:
        if kind != "score":
            raise ValueError("score is only valid for score records")
        score = _finite(row[label], label)
        values = metadata.get("values") or range(k)
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
class DataCollatorForDecision(DataCollatorMixin):
    """Collate decision requests and normalized targets for a model's forward pass.

    Renders requests and normalizes dataset targets before calling the model's preparation hook.

    Args:
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer or processor used to render the decision request.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
        prepare_inputs_for_decision (`Callable`):
            Model hook that receives rendered prompts and a `targets` keyword argument with one normalized distribution
            per question. It returns the model's forward inputs, including labels.
    """

    tokenizer: PreTrainedTokenizerBase
    decision_config: TransformersDecisionConfig
    prepare_inputs_for_decision: Callable[..., dict[str, Any]]
    return_tensors = "pt"

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompts, targets, is_choice = [], [], []
        for example in features:
            request = (
                example["request"]
                if "request" in example
                else {"state": example["state"], "questions": example["questions"]}
            )
            rendered = self.decision_config.prepare_inputs(request, self.tokenizer)
            if len(rendered) != 1 or len(rendered[0][1]) != 1:
                raise ValueError("Each training row must contain exactly one decision question")

            question = rendered[0][1][0]
            prompts.append(rendered[0])
            targets.append(_normalize_target(example, question))
            is_choice.append(question.type == "choice")

        batch = self.prepare_inputs_for_decision(prompts, targets=targets)

        return {**batch, "is_choice": torch.tensor(is_choice, dtype=torch.bool)}
