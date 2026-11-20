"""Text-to-SQL env for SkyRL: miles' episode (workloads/text2sql/t2sql_episode.py) as a skyrl-gym env.

The tool, the turn rules and the reward are t2sql_episode.py's own functions (River's `text_to_sql_recipe` underneath):
each turn either calls `execute_sql_query` once or answers; after 5 tool turns the 6th is graded whatever it contains;
an unsafe query ends the episode; observations are user messages (turns-left line, SQL_LIMIT_PROMPT after the 5th
tool turn). SkyRL's generator appends them token-in-token-out with the chat template, like the TRL port.

The SQL work runs as in TRL: one asyncio loop per process (a daemon thread here, since env.step is sync and runs in
SkyRL's env thread pool) holding the recipe's SqlWork(16) and the per-task gold-result cache.
"""

import asyncio
import collections
import json
import os
import sys
import threading
import types

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

# t2sql_episode.py imports TRL's rollout-loop classes for its TRL half; the env only uses its episode/reward half
if "trl.experimental.async_grpo.async_rollout_worker" not in sys.modules:
    for name in ("trl", "trl.experimental", "trl.experimental.async_grpo", "trl.experimental.async_grpo.async_rollout_worker"):
        sys.modules.setdefault(name, types.ModuleType(name))
    stub = sys.modules["trl.experimental.async_grpo.async_rollout_worker"]
    stub.AsyncRolloutLoop, stub.Rollout = object, None
    stub.TurnRecord = collections.namedtuple("TurnRecord", "prompt_ids output_ids logprobs")

import t2sql_episode as ep  # noqa: E402
from text_to_sql_recipe import parsing, prompts, reward  # noqa: E402

_LOOP = None
_LOCK = threading.Lock()
_EPISODES = None


def _run(coro):
    global _LOOP
    with _LOCK:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, daemon=True, name="t2sql-sql").start()
    return asyncio.run_coroutine_threadsafe(coro, _LOOP).result()


def _log(record):
    global _EPISODES
    path = os.environ.get("T2SQL_EPISODES_FILE")
    if not path:
        return
    with _LOCK:
        if _EPISODES is None:
            _EPISODES = open(path, "a", buffering=1)
        _EPISODES.write(json.dumps(record) + "\n")


class Text2SQLEnv(BaseTextEnv):
    def __init__(self, env_config=None, extras: dict = {}):
        super().__init__()
        self.task = json.loads(extras["task"])
        self.turn = 0
        self.episode = {"turns": [], "final_text": "", "unsafe": False, "tool_calls": 0, "tool_protocol_events": []}
        self.metrics = {}

    def _finish(self) -> BaseTextEnvStepOutput:
        scores = _run(ep.reward_func(self.task, self.episode, evaluation=False))
        self.metrics = {k: v for k, v in scores.items() if k != "error_category"}
        self.metrics["turns"] = float(self.turn)
        _log({"id": str(self.task["id"]), "turns": self.turn, **scores})
        return BaseTextEnvStepOutput(observations=[], reward=scores["score"], done=True, metadata={})

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turn += 1
        self.episode["turns"].append(action)
        message = {"role": "assistant", "content": ep._visible_text(action)}
        tools_allowed = self.turn <= ep.MAX_TOOL_TURNS
        reward.observe_tool_calls(self.episode["tool_protocol_events"], message, turn=self.turn, tools_allowed=tools_allowed)
        if not tools_allowed:
            self.episode["final_text"] = message["content"].strip()
            return self._finish()

        call = next((c for c in parsing.parse_tool_calls(message) if c.name == prompts.SQL_TOOL_NAME), None)
        if call is not None:
            self.episode["tool_calls"] += 1
            observation = _run(ep._sql_observation(self.task, call.arguments, turns_left=ep.MAX_TOOL_TURNS - self.turn))
            if observation is None:
                self.episode["unsafe"] = True
                return self._finish()
        elif parsing.has_tool_call_intent(message):
            observation = parsing.MALFORMED_ACTION_PROMPT
        else:
            self.episode["final_text"] = message["content"].strip()
            return self._finish()

        observations = [{"role": "user", "content": observation}]
        if self.turn == ep.MAX_TOOL_TURNS:
            observations.append({"role": "user", "content": prompts.SQL_LIMIT_PROMPT})
        return BaseTextEnvStepOutput(observations=observations, reward=0.0, done=False, metadata={})

    def get_metrics(self) -> dict:
        return self.metrics
