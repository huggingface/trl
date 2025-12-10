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

from .import_utils import suppress_experimental_warning


with suppress_experimental_warning():
    from .experimental.merge_model_callback import MergeConfig as _MergeConfig
    from .experimental.merge_model_callback import merge_models as _merge_models
    from .experimental.merge_model_callback import upload_model_to_hf as _upload_model_to_hf


def upload_model_to_hf(*args, **kwargs):
    warnings.warn(
        "`upload_model_to_hf` is now located in `trl.experimental`. Please update your imports to "
        "`from trl.experimental.merge_model_callback import upload_model_to_hf`. The current import path will be "
        "removed and no longer supported in TRL 0.29. For more information, see "
        "https://github.com/huggingface/trl/issues/4223.",
        FutureWarning,
        stacklevel=2,
    )
    return _upload_model_to_hf(*args, **kwargs)


class MergeConfig(_MergeConfig):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`MergeConfig` is now located in `trl.experimental`. Please update your imports to "
            "`from trl.experimental.merge_model_callback import MergeConfig`. The current import path will be "
            "removed and no longer supported in TRL 0.29. For more information, see "
            "https://github.com/huggingface/trl/issues/4223.",
            FutureWarning,
            stacklevel=2,
        )


def merge_models(*args, **kwargs):
    warnings.warn(
        "`merge_models` is now located in `trl.experimental`. Please update your imports to "
        "`from trl.experimental.merge_model_callback import merge_models`. The current import path will be "
        "removed and no longer supported in TRL 0.29. For more information, see "
        "https://github.com/huggingface/trl/issues/4223.",
        FutureWarning,
        stacklevel=2,
    )
    return _merge_models(*args, **kwargs)
