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
import torch
import transformers
from accelerate import Accelerator
from packaging.version import Version
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.import_utils import is_deepspeed_available
from trl.models.utils import disable_gradient_checkpointing, prepare_deepspeed, unwrap_model_for_generation


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


# transformers 5.23 renamed `gradient_checkpointing*` to `activation_checkpointing*` (see #8135)
def _is_checkpointing(model):
    if Version(transformers.__version__) >= Version("5.23.0"):
        return model.is_activation_checkpointing
    return model.is_gradient_checkpointing


def _enable_checkpointing(model, checkpointing_kwargs=None):
    if Version(transformers.__version__) >= Version("5.23.0"):
        model.activation_checkpointing_enable(checkpointing_kwargs)
    else:
        model.gradient_checkpointing_enable(checkpointing_kwargs)


class TestDisableGradientCheckpointing:
    def test_when_disabled(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        assert _is_checkpointing(model) is False
        with disable_gradient_checkpointing(model):
            assert _is_checkpointing(model) is False
        assert _is_checkpointing(model) is False

    def test_when_enabled(self):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        _enable_checkpointing(model)
        assert _is_checkpointing(model) is True
        with disable_gradient_checkpointing(model):
            assert _is_checkpointing(model) is False
        assert _is_checkpointing(model) is True


class TestUnwrapModelForGeneration:
    @pytest.mark.parametrize("use_reentrant", [True, False])
    def test_restores_gradient_checkpointing_kwargs(self, use_reentrant):
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5").to(torch_device)
        model.train()  # checkpointing only runs in training mode
        _enable_checkpointing(model, {"use_reentrant": use_reentrant})
        with unwrap_model_for_generation(
            model, Accelerator(), gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
        ):
            pass

        input_ids = torch.randint(0, model.config.vocab_size, (1, 8), device=torch_device)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        # Only non-reentrant checkpointing supports `torch.autograd.grad`, so this tells which one is enabled
        if use_reentrant:
            with pytest.raises(RuntimeError, match=r"\.grad\(\)"):
                torch.autograd.grad(loss, list(model.parameters()))
        else:
            torch.autograd.grad(loss, list(model.parameters()))
