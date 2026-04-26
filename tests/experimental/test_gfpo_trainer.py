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

import pytest
from datasets import load_dataset

from trl.experimental.gfpo import GFPOConfig, GFPOTrainer

from ..testing_utils import TrlTestCase


@pytest.mark.low_priority
class TestGFPOTrainer(TrlTestCase):
    def test_reward_metric_reflects_reward_weights_scheduler(self):
        """Test that GFPOTrainer uses scheduled reward weights in its GRPO override."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def constant_reward_1(completions, **kwargs):
            return [1.0] * len(completions)

        def constant_reward_0(completions, **kwargs):
            return [0.0] * len(completions)

        training_args = GFPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        trainer = GFPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=[constant_reward_1, constant_reward_0],
            reward_weights_scheduler=lambda state: [0.7, 0.3],
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        log = trainer.state.log_history[-1]
        assert abs(log["reward"] - 0.7) < 1e-5, (
            f"Expected logged reward to be ~0.7 (weighted), got {log['reward']}. "
            "The reward metric should reflect reward_weights_scheduler."
        )
        assert abs(log["reward_weights/constant_reward_1"] - 0.7) < 1e-5
        assert abs(log["reward_weights/constant_reward_0"] - 0.3) < 1e-5
