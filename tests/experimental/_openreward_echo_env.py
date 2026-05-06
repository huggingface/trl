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

"""Spawn target for `tests/experimental/test_openreward.py`.

Implements a minimal, deterministic Open Reward Standard environment so the adapter can be exercised end-to-end without
hitting the network. The same definition is also published as a Hugging Face Space at
`trl-internal-testing/openreward-echo-env` for the optional integration test.

The model is given a target string and must call `echo(text=...)` with exactly that string. Reward is `1.0` on match,
`0.0` otherwise; the episode finishes on a correct echo.
"""

from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    TextBlock,
    ToolOutput,
    tool,
)
from pydantic import BaseModel


TRAIN_TASKS: list[JSONObject] = [
    {"id": "echo-0", "target": "hello"},
    {"id": "echo-1", "target": "world"},
    {"id": "echo-2", "target": "trl"},
    {"id": "echo-3", "target": "openreward"},
]


class EchoTaskSpec(BaseModel):
    id: str
    target: str


class EchoParams(BaseModel):
    text: str


class EchoEnvironment(Environment):
    """Tiny deterministic ORS env: echo the target string to win."""

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):  # noqa: B006 - signature dictated by openreward.environments.Environment
        super().__init__(task_spec)
        self.config = EchoTaskSpec.model_validate(task_spec)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split != "train":
            raise ValueError(f"unknown split: {split}")
        return TRAIN_TASKS

    def get_prompt(self) -> list[TextBlock]:
        return [
            TextBlock(
                type="text",
                text=f"Call the `echo` tool with text='{self.config.target}' to win.",
            )
        ]

    @tool
    async def echo(self, params: EchoParams) -> ToolOutput:
        """Submit a string. Reward 1.0 + finished if it matches the target.

        Args:
            text: The string to echo back.
        """
        correct = params.text == self.config.target
        return ToolOutput(
            blocks=[
                TextBlock(
                    type="text",
                    text="match" if correct else f"no match (got {params.text!r})",
                )
            ],
            reward=1.0 if correct else 0.0,
            finished=correct,
        )


if __name__ == "__main__":
    import os

    Server([EchoEnvironment]).run(host="127.0.0.1", port=int(os.environ["PORT"]))
