# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from unittest.mock import MagicMock

import pytest

from trl import DPOConfig, GRPOConfig, KTOConfig, RLOOConfig
from trl.trainer.callbacks import SyncRefModelCallback


@pytest.mark.parametrize("config_cls", [DPOConfig, GRPOConfig, KTOConfig, RLOOConfig])
@pytest.mark.parametrize("invalid_steps", [0, -1, -512])
def test_sync_ref_model_non_positive_steps_raises(tmp_path, config_cls, invalid_steps):
    with pytest.raises(ValueError, match="`ref_model_sync_steps` must be greater than 0"):
        config_cls(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            sync_ref_model=True,
            ref_model_sync_steps=invalid_steps,
        )


@pytest.mark.parametrize("config_cls", [DPOConfig, GRPOConfig, KTOConfig, RLOOConfig])
def test_sync_ref_model_positive_steps_succeeds(tmp_path, config_cls):
    config = config_cls(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        sync_ref_model=True,
        ref_model_sync_steps=256,
    )
    assert config.sync_ref_model is True
    assert config.ref_model_sync_steps == 256


@pytest.mark.parametrize("config_cls", [DPOConfig, GRPOConfig, KTOConfig, RLOOConfig])
def test_sync_ref_model_disabled_by_default(tmp_path, config_cls):
    config = config_cls(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
    )
    assert config.sync_ref_model is False
    assert config.ref_model_sync_steps == 512


def test_sync_ref_model_callback_defensive_on_zero_steps():
    cb = SyncRefModelCallback(ref_model=MagicMock(), accelerator=None)
    cb.sync_target_model = MagicMock()

    args = SimpleNamespace(ref_model_sync_steps=0, ref_model_mixup_alpha=0.6)
    state = SimpleNamespace(global_step=1)

    # Should not raise ZeroDivisionError
    cb.on_step_end(args=args, state=state, control=None, model=MagicMock())
    cb.sync_target_model.assert_not_called()


def test_sync_ref_model_callback_syncs_on_modulo_step():
    cb = SyncRefModelCallback(ref_model=MagicMock(), accelerator=None)
    cb.sync_target_model = MagicMock()

    args = SimpleNamespace(ref_model_sync_steps=10, ref_model_mixup_alpha=0.6)
    model = MagicMock()

    # Step not matching modulo
    state_no_sync = SimpleNamespace(global_step=7)
    cb.on_step_end(args=args, state=state_no_sync, control=None, model=model)
    cb.sync_target_model.assert_not_called()

    # Step matching modulo
    state_sync = SimpleNamespace(global_step=10)
    cb.on_step_end(args=args, state=state_sync, control=None, model=model)
    cb.sync_target_model.assert_called_once_with(model, cb.ref_model, 0.6)
