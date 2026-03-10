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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trl.experimental.openenv.utils import generate_rollout_completions
from trl.generation.backend import RolloutCompletion


class TestOpenEnvUtils:
    def test_generate_rollout_completions_dispatch_and_schema(self):
        backend = MagicMock()
        backend.generate_rollout_completions.return_value = [
            RolloutCompletion(prompt_ids=[1, 2], completion_ids=[10, 11], logprobs=[-0.1, -0.2], text="hello")
        ]
        trainer = SimpleNamespace(generation_backend=backend, processing_class=MagicMock())

        result = generate_rollout_completions(
            trainer,
            prompts=["hi"],
            generation_overrides={"temperature": 0.7},
            as_chat=False,
        )

        assert result == [
            {
                "prompt_ids": [1, 2],
                "completion_ids": [10, 11],
                "logprobs": [-0.1, -0.2],
                "text": "hello",
            }
        ]
        backend.generate_rollout_completions.assert_called_once_with(
            prompts=["hi"],
            processing_class=trainer.processing_class,
            generation_overrides={"temperature": 0.7},
            as_chat=False,
        )

    def test_generate_rollout_completions_forwards_as_chat_none(self):
        backend = MagicMock()
        backend.generate_rollout_completions.return_value = []
        trainer = SimpleNamespace(generation_backend=backend, processing_class=MagicMock())

        generate_rollout_completions(trainer, prompts=[{"role": "user", "content": "hi"}], as_chat=None)

        backend.generate_rollout_completions.assert_called_once_with(
            prompts=[{"role": "user", "content": "hi"}],
            processing_class=trainer.processing_class,
            generation_overrides=None,
            as_chat=None,
        )

    def test_generate_rollout_completions_empty_prompts(self):
        backend = MagicMock()
        trainer = SimpleNamespace(generation_backend=backend, processing_class=MagicMock())

        result = generate_rollout_completions(trainer, prompts=[])

        assert result == []
        backend.generate_rollout_completions.assert_not_called()

    def test_generate_rollout_completions_surfaces_capability_error(self):
        backend = MagicMock()
        backend.generate_rollout_completions.side_effect = RuntimeError("does not support rollout completions")
        trainer = SimpleNamespace(generation_backend=backend, processing_class=MagicMock())

        with pytest.raises(RuntimeError, match="does not support rollout completions"):
            generate_rollout_completions(trainer, prompts=["x"])
