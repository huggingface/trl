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
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

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


def normalize_target(row: Mapping[str, Any], question: Question) -> tuple[float, ...]:
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
    values, positive_index = None, None
    if kind == "noul":
        positive_index = question.options.index("true")
    if kind == "score":
        values = tuple(metadata.get("values") or range(k))
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
class DataCollatorForDecision(DataCollatorMixin):
    """Base for example collators that adapt Transformers decision inputs to a model's forward signature.

    Subclasses implement `tokenize` and `torch_call` for their model's input layout. Request rendering and question
    parsing are delegated to Transformers; targets are normalized independently of the model's loss.

    Args:
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer or processor used to render the decision request.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
    """

    tokenizer: PreTrainedTokenizerBase
    decision_config: TransformersDecisionConfig
    return_tensors = "pt"

    def prepare_example(self, example: dict[str, Any]):
        """Return native inputs, the question, and its normalized null-first target distribution."""
        request = (
            example["request"]
            if "request" in example
            else {"state": example["state"], "questions": example["questions"]}
        )
        prompts = self.decision_config.prepare_inputs(request, self.tokenizer)
        if len(prompts) != 1 or len(prompts[0][1]) != 1:
            raise ValueError("Each training row must contain exactly one decision question")
        inputs, questions = prompts[0]
        question = questions[0]
        return inputs, question, normalize_target(example, question)
