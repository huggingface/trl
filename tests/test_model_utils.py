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
import torch.nn as nn
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from trl.import_utils import is_deepspeed_available
from trl.models.utils import disable_gradient_checkpointing, freeze_non_language_model_parameters, prepare_deepspeed


class _TextModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = nn.Linear(4, 4)


class _MultimodalConfig(PretrainedConfig):
    def __init__(self):
        self.text_config = PretrainedConfig()
        super().__init__()

    def get_text_config(self, decoder=False):
        return self.text_config


class _MultimodalModel(PreTrainedModel):
    config_class = _MultimodalConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = nn.Linear(4, 4)
        self.multi_modal_projector = nn.Linear(4, 4)
        self.language_model = _TextModel(config.text_config)
        self.lm_head = nn.Linear(4, 4)

    def get_output_embeddings(self):
        return self.lm_head


class _PromptLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _MultimodalModel(_MultimodalConfig())
        self.prompt_encoder = nn.ModuleDict({"default": nn.Linear(4, 4)})
        self.peft_config = {"default": types.SimpleNamespace(is_prompt_learning=True)}

    def get_base_model(self):
        return self.base_model


def test_freeze_non_language_model_parameters_preserves_language_trainability():
    model = _MultimodalModel(_MultimodalConfig())
    model.language_model.proj.bias.requires_grad_(False)

    freeze_non_language_model_parameters(model)

    assert not any(parameter.requires_grad for parameter in model.vision_tower.parameters())
    assert not any(parameter.requires_grad for parameter in model.multi_modal_projector.parameters())
    assert model.language_model.proj.weight.requires_grad
    assert not model.language_model.proj.bias.requires_grad
    assert all(parameter.requires_grad for parameter in model.lm_head.parameters())


def test_freeze_non_language_model_parameters_preserves_prompt_encoder():
    model = _PromptLearningModel()

    with patch("trl.models.utils.is_peft_model", return_value=True):
        freeze_non_language_model_parameters(model)

    assert all(parameter.requires_grad for parameter in model.prompt_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.base_model.vision_tower.parameters())


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
