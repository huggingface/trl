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
