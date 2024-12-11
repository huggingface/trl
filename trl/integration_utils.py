# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional

from transformers import is_comet_available


def get_comet_experiment_url() -> Optional[str]:
    """If Comet integration is enabled returns the URL of the current Comet experiment
    or None if disabled or no Comet experiment is currently running."""
    if not is_comet_available():
        return None

    import comet_ml

    if comet_ml.get_running_experiment() is not None:
        return comet_ml.get_running_experiment().url

    return None
