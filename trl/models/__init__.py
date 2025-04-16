# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ..import_utils import OptionalDependencyNotAvailable, _LazyModule, is_diffusers_available


_import_structure = {
    "activation_offloading": ["get_act_offloading_ctx_manager"],
    "modeling_base": ["GeometricMixtureWrapper", "PreTrainedModelWrapper", "create_reference_model"],
    "modeling_value_head": ["AutoModelForCausalLMWithValueHead", "AutoModelForSeq2SeqLMWithValueHead"],
    "utils": [
        "SUPPORTED_ARCHITECTURES",
        "prepare_deepspeed",
        "prepare_fsdp",
        "setup_chat_format",
        "unwrap_model_for_generation",
    ],
}

try:
    if not is_diffusers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_sd_base"] = [
        "DDPOPipelineOutput",
        "DDPOSchedulerOutput",
        "DDPOStableDiffusionPipeline",
        "DefaultDDPOStableDiffusionPipeline",
    ]

if TYPE_CHECKING:
    from .activation_offloading import get_act_offloading_ctx_manager
    from .modeling_base import GeometricMixtureWrapper, PreTrainedModelWrapper, create_reference_model
    from .modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
    from .utils import (
        SUPPORTED_ARCHITECTURES,
        prepare_deepspeed,
        prepare_fsdp,
        setup_chat_format,
        unwrap_model_for_generation,
    )

    try:
        if not is_diffusers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_sd_base import (
            DDPOPipelineOutput,
            DDPOSchedulerOutput,
            DDPOStableDiffusionPipeline,
            DefaultDDPOStableDiffusionPipeline,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
