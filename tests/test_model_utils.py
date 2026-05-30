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

import pytest
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM

from trl.models.utils import _override_model_generation_config, disable_gradient_checkpointing


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


class TestOverrideModelGenerationConfig:
    """Tests for _override_model_generation_config (workaround for transformers#42762).

    Qwen2.5 models ship with a model-specific generation_config that sets temperature
    close to 0 (near-greedy). On transformers < 5.0, passing generation_config to
    model.generate() merges model defaults on top, silently overriding training kwargs
    such as temperature=1.0. _override_model_generation_config temporarily replaces the
    model's generation_config with the training kwargs so the merge is a no-op.
    """

    def test_overrides_model_generation_config_during_context(self):
        """Inside the context, model.generation_config reflects the training kwargs.

        Only applies to transformers < 5.0; on v5+ the upstream bug is fixed and
        the context manager is a no-op.
        """
        if Version(transformers.__version__) >= Version("5.0.0"):
            pytest.skip("transformers >= 5.0 fixes the upstream bug; context manager is a no-op")
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        training_kwargs = {"temperature": 1.0, "do_sample": True}
        with _override_model_generation_config(model, generation_kwargs=training_kwargs):
            assert model.generation_config.temperature == 1.0
            assert model.generation_config.do_sample is True

    def test_restores_original_generation_config_after_context(self):
        """After the context, model.generation_config is restored to its original value."""
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        original_temperature = model.generation_config.temperature
        original_config_id = id(model.generation_config)
        training_kwargs = {"temperature": 1.0, "do_sample": True}
        with _override_model_generation_config(model, generation_kwargs=training_kwargs):
            pass
        assert model.generation_config.temperature == original_temperature
        assert id(model.generation_config) == original_config_id

    def test_no_op_when_generation_kwargs_is_none(self):
        """When generation_kwargs is None, the context manager yields without modifying the model."""
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        original_config_id = id(model.generation_config)
        with _override_model_generation_config(model, generation_kwargs=None):
            assert id(model.generation_config) == original_config_id

    def test_no_op_on_transformers_v5(self):
        """On transformers >= 5.0 the upstream bug is fixed; the context manager is a no-op."""
        if Version(transformers.__version__) < Version("5.0.0"):
            return  # only meaningful on transformers v5+
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        original_config_id = id(model.generation_config)
        training_kwargs = {"temperature": 1.0}
        with _override_model_generation_config(model, generation_kwargs=training_kwargs):
            assert id(model.generation_config) == original_config_id
