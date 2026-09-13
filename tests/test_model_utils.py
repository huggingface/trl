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

import types
from unittest.mock import Mock, patch

import accelerate
import pytest
import torch
from packaging.version import Version
from torch import nn
from transformers import AutoModelForCausalLM

from trl import DPOTrainer, KTOTrainer
from trl.import_utils import is_deepspeed_available
from trl.models.utils import disable_gradient_checkpointing, prepare_deepspeed, prepare_fsdp


@pytest.mark.skipif(not is_deepspeed_available(), reason="deepspeed is not installed")
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_prepare_deepspeed_strips_optimizer_for_cpu_offload(stage):
    pytest.importorskip("deepspeed")
    # prepare_deepspeed initializes eval-only models without an optimizer. If the deep-copied config still carries the
    # training-only blocks, DeepSpeed builds a CPUAdam on GPU params (`optimizer` + `offload_optimizer: cpu`) or an LR
    # scheduler with no optimizer (`scheduler`), and the run dies before training. They must be dropped before
    # `deepspeed.initialize`. Intercept the call to check the config it would receive.
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage, "offload_optimizer": {"device": "cpu"}},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-5}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-5, "warmup_num_steps": 10},
        },
    }
    accelerator = types.SimpleNamespace(
        state=types.SimpleNamespace(deepspeed_plugin=types.SimpleNamespace(deepspeed_config=ds_config))
    )

    captured = {}

    def fake_initialize(model, config):
        captured["config"] = config
        return types.SimpleNamespace(eval=lambda: None), None, None, None

    with patch("deepspeed.initialize", fake_initialize):
        prepare_deepspeed(model=None, accelerator=accelerator)

    assert "optimizer" not in captured["config"]
    assert "scheduler" not in captured["config"]
    assert "offload_optimizer" not in captured["config"]["zero_optimization"]


@pytest.mark.skipif(Version(accelerate.__version__) < Version("1.6.0"), reason="requires accelerate>=1.6.0")
def test_prepare_fsdp2_uses_accelerate_auto_wrap():
    model = nn.Linear(2, 2)
    accelerator = types.SimpleNamespace(state=types.SimpleNamespace(fsdp_plugin=types.SimpleNamespace(fsdp_version=2)))

    with patch("trl.models.utils.fsdp2_prepare_model", return_value=model) as prepare_model:
        result = prepare_fsdp(model, accelerator)

    prepare_model.assert_called_once_with(accelerator, model)
    assert result is model
    assert result.training is False


@pytest.mark.parametrize("trainer_cls", [DPOTrainer, KTOTrainer])
@pytest.mark.parametrize("optimizer_prepared_during_precompute", [False, True])
def test_precomputed_fsdp_model_is_reused_for_training(trainer_cls, optimizer_prepared_during_precompute):
    trainer = object.__new__(trainer_cls)
    model = nn.Linear(2, 2)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer._precompute_engine = model
    trainer.is_fsdp_enabled = True
    trainer._created_lr_scheduler = False
    trainer.lr_scheduler = None
    trainer.optimizer = optimizer if optimizer_prepared_during_precompute else None
    trainer.model = model
    trainer.model_wrapped = model
    trainer.callback_handler = types.SimpleNamespace()

    def create_optimizer():
        trainer.optimizer = optimizer

    trainer.create_optimizer = Mock(side_effect=create_optimizer)
    trainer.create_scheduler = Mock()
    trainer.accelerator = types.SimpleNamespace(
        prepare_model=Mock(return_value=model),
        prepare_optimizer=Mock(return_value=optimizer),
        parallelism_config=None,
    )
    train_dataloader = object()

    prepared_model, prepared_dataloader = trainer._prepare_for_training(4, train_dataloader, None)

    trainer.accelerator.prepare_model.assert_called_once_with(model)
    trainer.accelerator.prepare_optimizer.assert_called_once_with(optimizer)
    if optimizer_prepared_during_precompute:
        trainer.create_optimizer.assert_not_called()
    else:
        trainer.create_optimizer.assert_called_once_with()
    trainer.create_scheduler.assert_called_once_with(num_training_steps=4)
    assert prepared_model is trainer.model is trainer.model_wrapped is trainer._precompute_engine
    assert prepared_dataloader is trainer.callback_handler.train_dataloader is train_dataloader
    assert prepared_model.training is True


class TestDisableGradientCheckpointing:
    def test_when_disabled(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        assert model.is_gradient_checkpointing is False
        with disable_gradient_checkpointing(model):
            assert model.is_gradient_checkpointing is False
        assert model.is_gradient_checkpointing is False

    def test_when_enabled(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model.gradient_checkpointing_enable()
        assert model.is_gradient_checkpointing is True
        with disable_gradient_checkpointing(model):
            assert model.is_gradient_checkpointing is False
        assert model.is_gradient_checkpointing is True
