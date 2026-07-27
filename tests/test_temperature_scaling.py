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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trl import GRPOTrainer, RLOOTrainer


class LogitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(1, 5, 8))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits.repeat(input_ids.size(0), 1, 1))


def get_autograd_node_names(tensor):
    names = set()
    seen = set()
    pending = [tensor.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.add(type(node).__name__)
        pending.extend(next_node for next_node, _ in node.next_functions)
    return names


@pytest.mark.parametrize("trainer_cls", [GRPOTrainer, RLOOTrainer])
def test_temperature_scaling_does_not_materialize_logits_view(trainer_cls):
    trainer = SimpleNamespace(model_kwarg_keys=set(), temperature=0.7, _entropy_bonus_enabled=False)
    model = LogitsModel()
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])

    logps, _, _ = trainer_cls._get_per_token_logps_and_entropies(
        trainer,
        model,
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        logits_to_keep=3,
    )

    assert "CopySlices" not in get_autograd_node_names(logps)
