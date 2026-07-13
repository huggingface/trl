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

"""Tests for the Qwen3.5 / Qwen3-VL weight-name prefix mapping used during vLLM weight sync."""

from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from trl.generation.vllm_generation import VLLMGeneration, get_vllm_param_prefix


def make_model(class_name: str, architectures=None, name_or_path=None, config=None):
    """Build a minimal model stand-in whose class name and config drive the prefix detection.

    Must be a real `nn.Module`: `get_vllm_param_prefix` runs the model through accelerate's
    `is_peft_model`, which walks module internals.
    """
    if config is None:
        config = SimpleNamespace()
        if architectures is not None:
            config.architectures = architectures
    cls = type(class_name, (nn.Module,), {})
    model = cls()
    model.config = config
    if name_or_path is not None:
        model.name_or_path = name_or_path
    return model


class TestGetVllmParamPrefix:
    def test_qwen3_5_moe_text_trainer_gets_language_model_prefix(self):
        # Trainer holds the text-only model; checkpoint architecture is the VLM wrapper.
        model = make_model(
            "Qwen3_5MoeForCausalLM", architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        assert get_vllm_param_prefix(model) == "language_model."

    def test_qwen3_vl_moe_trainer_gets_language_model_prefix(self):
        model = make_model("Qwen3MoeForCausalLM", architectures=["Qwen3VLMoeForConditionalGeneration"])
        assert get_vllm_param_prefix(model) == "language_model."

    def test_trainer_holding_the_wrapper_itself_needs_no_prefix(self):
        model = make_model(
            "Qwen3_5MoeForConditionalGeneration", architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        assert get_vllm_param_prefix(model) == ""

    def test_non_qwen_vlm_passes_through(self):
        # Unknown conditional-generation wrappers are NOT remapped (explicit allowlist only).
        model = make_model("LlamaForCausalLM", architectures=["LlavaForConditionalGeneration"])
        assert get_vllm_param_prefix(model) == ""

    def test_plain_causal_lm_needs_no_prefix(self):
        model = make_model("Qwen2ForCausalLM", architectures=["Qwen2ForCausalLM"])
        assert get_vllm_param_prefix(model) == ""

    def test_missing_architectures_without_name_or_path_is_passthrough(self):
        model = make_model("Qwen3_5MoeForCausalLM", architectures=None)
        assert get_vllm_param_prefix(model) == ""

    def test_missing_config_is_passthrough(self):
        cls = type("SomeModel", (nn.Module,), {})
        model = cls()
        assert get_vllm_param_prefix(model) == ""

    def test_subconfig_without_architectures_refetches_checkpoint_config(self):
        # model.config is a sub-config (e.g. text_config) lacking `architectures`; the checkpoint
        # config must be re-fetched to see the wrapper architecture vLLM will load.
        model = make_model("Qwen3_5MoeForCausalLM", architectures=None, name_or_path="org/qwen3.5-ckpt")
        full_config = SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=full_config) as mock_fetch:
            assert get_vllm_param_prefix(model) == "language_model."
        mock_fetch.assert_called_once_with("org/qwen3.5-ckpt")

    def test_refetch_failure_is_passthrough(self):
        model = make_model("Qwen3_5MoeForCausalLM", architectures=None, name_or_path="org/missing")
        with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("gone")):
            assert get_vllm_param_prefix(model) == ""


class TestFixParamNameToVllm:
    def make_generation(self, prefix: str) -> VLLMGeneration:
        # _fix_param_name_to_vllm only reads `_vllm_param_prefix`; skip the heavy __init__.
        generation = VLLMGeneration.__new__(VLLMGeneration)
        generation._vllm_param_prefix = prefix
        return generation

    def test_model_weights_map_under_language_model(self):
        generation = self.make_generation("language_model.")
        name = generation._fix_param_name_to_vllm("model.layers.0.self_attn.q_proj.weight")
        assert name == "language_model.model.layers.0.self_attn.q_proj.weight"

    def test_lm_head_maps_under_language_model(self):
        generation = self.make_generation("language_model.")
        assert generation._fix_param_name_to_vllm("lm_head.weight") == "language_model.lm_head.weight"

    def test_wrapper_prefixes_are_stripped_before_prefixing(self):
        generation = self.make_generation("language_model.")
        name = generation._fix_param_name_to_vllm(
            "model.layers.0._checkpoint_wrapped_module.mlp.gate_proj.weight",
            extra_prefixes=["_fsdp_wrapped_module."],
        )
        assert name == "language_model.model.layers.0.mlp.gate_proj.weight"

    def test_no_prefix_passthrough(self):
        generation = self.make_generation("")
        name = generation._fix_param_name_to_vllm("model.layers.0.self_attn.q_proj.weight")
        assert name == "model.layers.0.self_attn.q_proj.weight"
