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

from dataclasses import dataclass
from typing import Any

import torch
from transformers.decision import Question

from trl.experimental.decision import DataCollatorForDecision


@dataclass
class DataCollatorForJev(DataCollatorForDecision):
    """Prepare one text decision prompt per microbatch for JEV's packed option loss.

    Args:
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer used to render the prompt.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
    """

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        inputs, question, targets = self.prepare_example(example)
        if targets[0]:
            raise ValueError("JEV does not support nullable decisions")
        return {
            "input_ids": inputs["input_ids"][0].tolist(),
            "marks": inputs["marks"].tolist(),
            "num_options": [len(question.options)],
            "question_type_ids": [Question.questions_types.index(question.type)],
            "labels": list(targets[1:]),
            "is_choice": question.type == "choice",
        }

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("JEV requires per_device_train_batch_size=1 and per_device_eval_batch_size=1")
        example = features[0] if "input_ids" in features[0] else self.tokenize(features[0])
        return {key: torch.tensor([value]) for key, value in example.items()}
