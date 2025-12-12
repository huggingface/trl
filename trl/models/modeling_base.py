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

import warnings

from ..import_utils import suppress_experimental_warning


with suppress_experimental_warning():
    from ..experimental.ppo.modeling_value_head import PreTrainedModelWrapper as _PreTrainedModelWrapper


LAYER_PATTERNS = [
    "transformer.h.{layer}",
    "model.decoder.layers.{layer}",
    "gpt_neox.layers.{layer}",
    "model.layers.{layer}",
]


class PreTrainedModelWrapper(_PreTrainedModelWrapper):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `PreTrainedModelWrapper` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.bco import PreTrainedModelWrapper`. The current import path will be removed and "
            "no longer supported in TRL 0.29. For more information, see "
            "https://github.com/huggingface/trl/issues/4223.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
