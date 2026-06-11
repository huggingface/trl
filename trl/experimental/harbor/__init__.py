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

"""Harbor × TRL integration (experimental).

Train on Harbor agentic task suites with `GRPOTrainer` via `environment_factory`, with a pluggable
base agent (harness). Requires `harbor` installed in the same interpreter (`pip install trl[harbor]`,
Python >= 3.12); `harbor` is imported lazily so this module imports without it.

```python
from trl.experimental.harbor import HarborSpec
spec = HarborSpec("AdithyaSK/data_agent_rl_environment_train", agent="bash", num_tasks=64)
```
"""

from ._env import AGENTS, HarborBashEnv, HarborEnv
from ._spec import HarborSpec


__all__ = ["AGENTS", "HarborBashEnv", "HarborEnv", "HarborSpec"]
