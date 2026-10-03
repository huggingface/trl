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
from unittest.mock import patch

import pytest
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

from trl.import_utils import is_deepspeed_available
from trl.models.utils import (
    _unwrap_model_for_generation,
    disable_gradient_checkpointing,
    prepare_deepspeed,
    unwrap_model_for_generation,
)


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

    @pytest.mark.parametrize("unwrap", [_unwrap_model_for_generation, unwrap_model_for_generation])
    @pytest.mark.parametrize("is_gradient_checkpointing", [False, True])
    @pytest.mark.parametrize(
        "checkpointing_kwargs",
        [
            None,
            {"use_reentrant": True, "preserve_rng_state": False},
            {"use_reentrant": False, "preserve_rng_state": False},
        ],
    )
    def test_unwrap_restores_kwargs(self, unwrap, is_gradient_checkpointing, checkpointing_kwargs):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        accelerator = Accelerator(cpu=True)
        if is_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
            original_kwargs = model.model._gradient_checkpointing_func.keywords.copy()

        with unwrap(model, accelerator, gradient_checkpointing_kwargs=checkpointing_kwargs) as unwrapped_model:
            assert unwrapped_model is model
            assert model.is_gradient_checkpointing is False

        assert model.is_gradient_checkpointing is is_gradient_checkpointing
        if is_gradient_checkpointing:
            assert model.model._gradient_checkpointing_func.keywords == original_kwargs

    @pytest.mark.parametrize("unwrap", [_unwrap_model_for_generation, unwrap_model_for_generation])
    @pytest.mark.parametrize("is_gradient_checkpointing", [False, True])
    @pytest.mark.parametrize(
        "checkpointing_kwargs",
        [
            None,
            {"use_reentrant": True, "preserve_rng_state": False},
            {"use_reentrant": False, "preserve_rng_state": False},
        ],
    )
    def test_unwrap_restores_kwargs_after_exception(self, unwrap, is_gradient_checkpointing, checkpointing_kwargs):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        accelerator = Accelerator(cpu=True)
        if is_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing_kwargs)
            original_kwargs = model.model._gradient_checkpointing_func.keywords.copy()

        with (
            pytest.raises(RuntimeError, match="generation failed"),
            unwrap(model, accelerator, gradient_checkpointing_kwargs=checkpointing_kwargs) as unwrapped_model,
        ):
            assert unwrapped_model is model
            assert model.is_gradient_checkpointing is False
            raise RuntimeError("generation failed")

        assert model.is_gradient_checkpointing is is_gradient_checkpointing
        if is_gradient_checkpointing:
            assert model.model._gradient_checkpointing_func.keywords == original_kwargs
