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

from ..import_utils import _LazyModule


_import_structure = {
    "activation_offloading": ["get_act_offloading_ctx_manager"],
    "modeling_base": ["PreTrainedModelWrapper"],
    "modeling_value_head": ["AutoModelForCausalLMWithValueHead", "AutoModelForSeq2SeqLMWithValueHead"],
    "utils": [
        "create_reference_model",
        "prepare_deepspeed",
        "prepare_fsdp",
        "prepare_model_for_kbit_training",
        "prepare_peft_model",
        "unwrap_model_for_generation",
    ],
}


if TYPE_CHECKING:
    from .activation_offloading import get_act_offloading_ctx_manager
    from .modeling_base import PreTrainedModelWrapper
    from .modeling_value_head import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
    from .utils import (
        create_reference_model,
        prepare_deepspeed,
        prepare_fsdp,
        prepare_model_for_kbit_training,
        prepare_peft_model,
        unwrap_model_for_generation,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
