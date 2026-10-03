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

from unittest.mock import call

import pytest


def test_grpo_rollout_loads_weights_once_and_sleeps_before_scoring(colocate_grpo_trainer):
    trainer = colocate_grpo_trainer
    llm = trainer.vllm_generation.llm

    def two_turn_rollout(prompts, trainer):
        for _ in range(2):
            trainer.vllm_generation.generate([[2]], images=None, num_generations=1)
        llm.sleep.assert_not_called()
        return {"prompt_ids": [[2], [2]], "completion_ids": [[2], [2]], "logprobs": [[0.0], [0.0]]}

    def stop_before_scoring(*args, **kwargs):
        llm.sleep.assert_called_once_with(level=2)
        raise RuntimeError("stop before scoring")

    trainer.rollout_func = two_turn_rollout
    trainer._get_per_token_logps_and_entropies = stop_before_scoring
    with pytest.raises(RuntimeError, match="stop before scoring"):
        trainer._generate_and_score_completions([{"prompt": "a"}, {"prompt": "a"}])

    assert llm.wake_up.call_args_list == [call(tags=["weights"]), call(tags=["kv_cache"])]


def test_sleep_does_not_sleep_an_asleep_engine(vllm_generation):
    vllm_generation.enable_sleep_mode = True
    vllm_generation._kv_cache_sleeping = True
    vllm_generation.sleep()

    vllm_generation.llm.sleep.assert_not_called()
