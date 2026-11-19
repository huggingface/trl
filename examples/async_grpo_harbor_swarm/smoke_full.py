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

"""Synthetic-reward engineering smoke: real sampled tokens, full gradients, and weight transfer.

This deliberately does not report benchmark rewards. Native Harbor runs are a separate gate.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from training import SwarmRolloutLoop, SwarmTrainer

from trl.trainer.async_grpo_trainer import Rollout, TurnRecord


class SyntheticLoop(SwarmRolloutLoop):
    async def setup(self):
        self.runner = SimpleNamespace(root=Path(self.config.swarm_settings["output_dir"]))
        self.calls = 0

    async def close(self):
        pass

    async def generate_one(self, prompt, tools, group_id=0):
        self.calls += 1
        index = self.calls
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Give one short fact about prime numbers."}],
            "max_tokens": 32,
            "temperature": 1.0,
            "top_p": 1.0,
            "logprobs": True,
            "top_logprobs": 0,
            "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with httpx.AsyncClient(timeout=600) as http:
            response = await http.post(f"{self.config.swarm_settings['gateway_url']}/v1/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            version = int(response.headers["x-swarm-policy-version"])
        choice = data["choices"][0]
        turn = TurnRecord(
            data["prompt_token_ids"],
            choice["token_ids"],
            [token["logprob"] for token in choice["logprobs"]["content"]],
        )
        return Rollout(
            completion=[
                {
                    "role": "assistant",
                    "content": json.dumps({"team_id": f"synthetic-{index}", "policy_version": version}),
                }
            ],
            completion_ids=choice["token_ids"],
            turns=[turn],
            reward=float(index % 2),
        )


class SyntheticTrainer(SwarmTrainer):
    loop_cls = SyntheticLoop
