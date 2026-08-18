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

"""Wordle environment server with higher concurrent session capacity for async GRPO training."""

import os

from openenv.core.env_server.http_server import ConcurrencyConfig, create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.environment import TextArenaEnvironment


# Mark TextArena as safe for concurrent sessions (each session gets its own game instance)
TextArenaEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True

max_sessions = int(os.getenv("MAX_CONCURRENT_SESSIONS", "256"))

app = create_app(
    TextArenaEnvironment,
    TextArenaAction,
    TextArenaObservation,
    concurrency_config=ConcurrencyConfig(max_concurrent_envs=max_sessions),
)
