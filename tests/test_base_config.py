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

import dataclasses
from unittest.mock import patch

import pytest

from trl import DPOConfig, GRPOConfig, KTOConfig, RewardConfig, RLOOConfig, SFTConfig


KERNELS_AVAILABLE = "trl.trainer.base_config.is_kernels_available"
CONFIGS = [SFTConfig, DPOConfig, KTOConfig, GRPOConfig, RLOOConfig, RewardConfig]


@pytest.mark.parametrize("config_cls", CONFIGS)
class TestUseKernels:
    def test_default_is_off(self, config_cls, tmp_path):
        config = config_cls(output_dir=str(tmp_path), report_to="none")
        assert config.use_kernels is False
        assert config.model_init_kwargs is None

    def test_sets_hub_kernels_and_flash_attention_kernel(self, config_cls, tmp_path):
        with patch(KERNELS_AVAILABLE, return_value=True):
            config = config_cls(output_dir=str(tmp_path), report_to="none", use_kernels=True)
        assert config.model_init_kwargs == {
            "use_kernels": True,
            "attn_implementation": "kernels-community/flash-attn2",
        }

    def test_explicit_attn_implementation_and_other_kwargs_are_kept(self, config_cls, tmp_path):
        model_init_kwargs = {"dtype": "bfloat16", "attn_implementation": "kernels-community/vllm-flash-attn3"}
        with patch(KERNELS_AVAILABLE, return_value=True):
            config = config_cls(
                output_dir=str(tmp_path), report_to="none", use_kernels=True, model_init_kwargs=model_init_kwargs
            )
        assert config.model_init_kwargs == {**model_init_kwargs, "use_kernels": True}
        # The dict passed by the user is not mutated
        assert model_init_kwargs == {"dtype": "bfloat16", "attn_implementation": "kernels-community/vllm-flash-attn3"}

    def test_conflicting_model_init_kwargs_raises(self, config_cls, tmp_path):
        with patch(KERNELS_AVAILABLE, return_value=True):
            with pytest.raises(ValueError, match="conflicts with"):
                config_cls(
                    output_dir=str(tmp_path),
                    report_to="none",
                    use_kernels=True,
                    model_init_kwargs={"use_kernels": False},
                )

    def test_missing_kernels_library_raises(self, config_cls, tmp_path):
        with patch(KERNELS_AVAILABLE, return_value=False):
            with pytest.raises(ImportError, match="requires the `kernels` library"):
                config_cls(output_dir=str(tmp_path), report_to="none", use_kernels=True)

    def test_replace_is_idempotent(self, config_cls, tmp_path):
        with patch(KERNELS_AVAILABLE, return_value=True):
            config = config_cls(output_dir=str(tmp_path), report_to="none", use_kernels=True)
            replaced = dataclasses.replace(config)
        assert replaced.model_init_kwargs == config.model_init_kwargs

    def test_legacy_model_init_kwargs_spelling_is_untouched(self, config_cls, tmp_path):
        config = config_cls(output_dir=str(tmp_path), report_to="none", model_init_kwargs={"use_kernels": True})
        assert config.model_init_kwargs == {"use_kernels": True}
