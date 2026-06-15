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

import asyncio
from dataclasses import dataclass

from accelerate.state import PartialState

from trl.experimental.async_grpo.async_rollout_worker import RolloutGroup, _AsyncRolloutLoop


@dataclass
class EnvSnapshot:
    reward: float


def test_score_group_passes_environments():
    captured = {}

    def reward_func(completions, environments, **kwargs):
        captured["environments"] = environments
        return [env.reward for env in environments]

    rollout_loop = _AsyncRolloutLoop.__new__(_AsyncRolloutLoop)
    rollout_loop.reward_funcs = [reward_func]
    rollout_loop.reward_func_names = ["reward_func"]
    PartialState()

    group = RolloutGroup(
        prompt=[{"role": "user", "content": "hi"}],
        prompt_ids=[1, 2],
        reward_kwargs={},
        environments=[EnvSnapshot(0.25), EnvSnapshot(0.75)],
        completions=[
            [{"role": "assistant", "content": "left"}],
            [{"role": "assistant", "content": "right"}],
        ],
        completions_ids=[[3], [4]],
        completions_logprobs=[[-0.1], [-0.2]],
        tool_mask=[[1], [1]],
        tool_call_counts=[0, 0],
        tool_failure_counts=[0, 0],
        model_version=7,
    )

    samples = asyncio.run(rollout_loop._score_group(group))

    assert [env.reward for env in captured["environments"]] == [0.25, 0.75]
    assert [sample.metrics["reward"] for sample in samples] == [0.25, 0.75]
