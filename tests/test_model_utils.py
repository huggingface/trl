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
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.utils import is_peft_available

from trl.import_utils import is_deepspeed_available
from trl.models.utils import disable_gradient_checkpointing, freeze_non_language_model_parameters, prepare_deepspeed


if is_peft_available():
    from peft import PromptTuningConfig, TaskType, get_peft_model


def test_freeze_non_language_model_parameters_preserves_language_trainability():
    model = AutoModelForImageTextToText.from_pretrained("trl-internal-testing/tiny-LlavaForConditionalGeneration")
    language_model = model.model.language_model
    frozen_parameter = next(language_model.parameters())
    frozen_parameter.requires_grad_(False)

    freeze_non_language_model_parameters(model)

    assert not any(parameter.requires_grad for parameter in model.model.vision_tower.parameters())
    assert not any(parameter.requires_grad for parameter in model.model.multi_modal_projector.parameters())
    assert not frozen_parameter.requires_grad
    assert all(
        parameter.requires_grad for parameter in language_model.parameters() if parameter is not frozen_parameter
    )
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())


@pytest.mark.skipif(not is_peft_available(), reason="peft is not installed")
def test_freeze_non_language_model_parameters_preserves_prompt_encoder():
    base_model = AutoModelForImageTextToText.from_pretrained("trl-internal-testing/tiny-LlavaForConditionalGeneration")
    model = get_peft_model(
        base_model,
        PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=4),
    )

    freeze_non_language_model_parameters(model)

    assert all(parameter.requires_grad for parameter in model.prompt_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.get_base_model().model.vision_tower.parameters())


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
