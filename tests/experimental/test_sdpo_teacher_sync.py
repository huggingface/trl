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

import sys
import types
from unittest.mock import MagicMock, patch

import torch

from trl.experimental.sdpo.sdpo_trainer import SDPOTrainer
from trl.experimental.sdpo.teacher_sync import PEFTAdapterEMACallback


def test_peft_ema_teacher_is_initialized_before_callback_registration():
    trainer = object.__new__(SDPOTrainer)
    trainer.args = types.SimpleNamespace(
        teacher_model_kind="ema",
        teacher_update_rate=0.05,
        teacher_sync_steps=1,
    )
    trainer.model = MagicMock()
    trainer.accelerator = MagicMock()
    trainer._use_peft_ema_teacher_adapter = lambda: True
    events = []
    trainer.add_callback = lambda callback: events.append(("register", callback))

    callback = MagicMock()
    callback._initialize_teacher_adapter.side_effect = lambda: events.append(("initialize", callback))
    with patch("trl.experimental.sdpo.sdpo_trainer.PEFTAdapterEMACallback", return_value=callback):
        trainer._setup_teacher_model()

    assert events == [("initialize", callback), ("register", callback)]
    assert trainer.teacher_model is trainer.model


def test_peft_ema_student_state_is_gathered_and_cloned_under_zero3():
    parameter = MagicMock(requires_grad=True)
    parameter.numel.return_value = 0
    model = MagicMock()
    model.parameters.return_value = [parameter]
    accelerator = MagicMock()
    accelerator.unwrap_model.return_value = model
    accelerator.state.deepspeed_plugin.zero_stage = 3
    callback = PEFTAdapterEMACallback(model=model, accelerator=accelerator)
    source = torch.tensor([1.0])
    events = []

    class GatheredParameters:
        def __init__(self, parameters, modifier_rank):
            assert parameters == [parameter]
            assert modifier_rank is None

        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")

    deepspeed = types.SimpleNamespace(zero=types.SimpleNamespace(GatheredParameters=GatheredParameters))
    with (
        patch.dict(sys.modules, {"deepspeed": deepspeed}),
        patch("peft.get_peft_model_state_dict", return_value={"adapter.weight": source}),
    ):
        state_dict = callback._get_student_state_dict()

    assert events == ["enter", "exit"]
    assert torch.equal(state_dict["adapter.weight"], source)
    assert state_dict["adapter.weight"].data_ptr() != source.data_ptr()
