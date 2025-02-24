# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ..import_utils import is_agents_available

if not is_agents_available():
    raise ImportError(
        "Agents utilities are not available. Please install trl with "
        "`pip install trl[agents]` to use utils"
    )

from .utils import (
    E2BExecutor,
    LocalExecutor,
    generate_agent_responses,
    get_code,
    prepare_data_for_e2b_agent,
    prepare_data_for_local_agent,
    read_script,
)


__all__ = [
    "E2BExecutor",
    "LocalExecutor",
    "prepare_data_for_e2b_agent",
    "prepare_data_for_local_agent",
    "generate_agent_responses",
    "get_code",
    "read_script",
]
