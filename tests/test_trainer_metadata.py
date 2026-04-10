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

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name", "expected"),
    [
        ("trl", "DPOTrainer", ("preference",)),
        ("trl", "GRPOTrainer", ("prompt-only",)),
        ("trl", "RewardTrainer", ("preference",)),
        ("trl", "RLOOTrainer", ("prompt-only",)),
        ("trl", "SFTTrainer", ("language-modeling", "prompt-completion")),
        ("trl.experimental.bco", "BCOTrainer", ("unpaired-preference", "preference")),
        ("trl.experimental.cpo", "CPOTrainer", ("preference",)),
        ("trl.experimental.gkd", "GKDTrainer", ("prompt-completion",)),
        ("trl.experimental.kto", "KTOTrainer", ("unpaired-preference", "preference")),
        ("trl.experimental.nash_md", "NashMDTrainer", ("prompt-only",)),
        ("trl.experimental.online_dpo", "OnlineDPOTrainer", ("prompt-only",)),
        ("trl.experimental.orpo", "ORPOTrainer", ("preference",)),
        ("trl.experimental.ppo", "PPOTrainer", ("tokenized-language-modeling",)),
        ("trl.experimental.prm", "PRMTrainer", ("stepwise-supervision",)),
        ("trl.experimental.xpo", "XPOTrainer", ("prompt-only",)),
    ],
)
def test_trainer_dataset_types(module_name, class_name, expected):
    module = import_module(module_name)
    trainer_class = getattr(module, class_name)

    assert trainer_class.dataset_types == expected
