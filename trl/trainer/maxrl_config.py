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

from dataclasses import dataclass

from .grpo_config import GRPOConfig


@dataclass
class MaxRLConfig(GRPOConfig):
    r"""
    Configuration class for the [`MaxRLTrainer`].

    This class extends [`GRPOConfig`] and inherits all its parameters. The key difference in MaxRL compared to GRPO
    is the advantage calculation: MaxRL divides advantages by the mean reward instead of the standard deviation,
    as described in the paper [Maximum Likelihood Reinforcement Learning](https://huggingface.co/papers/2602.02710).

    MaxRL modifies the advantage calculation to:
    ```
    advantage = (reward - mean_reward) / (mean_reward + epsilon)
    ```

    instead of GRPO's:
    ```
    advantage = (reward - mean_reward) / (std_reward + epsilon)
    ```

    This normalization by mean reward (p-normalization) helps prevent bias towards questions with different difficulty
    levels and can lead to more stable training.

    For a complete list of parameters, please refer to the [`GRPOConfig`] documentation.

    Example:

    ```python
    from trl import MaxRLTrainer, MaxRLConfig
    from trl.rewards import accuracy_reward
    from datasets import load_dataset

    dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

    config = MaxRLConfig(
        output_dir="maxrl_model",
        num_generations=8,
        learning_rate=1e-6,
    )

    trainer = MaxRLTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()
    ```
    """

    def __post_init__(self):
        super().__post_init__()

        # MaxRL uses p-normalization (dividing by mean), so we override scale_rewards
        # The actual implementation is in the MaxRLTrainer's _generate_and_score_completions method
        if self.scale_rewards not in ["none", False]:
            # User explicitly set scale_rewards, warn them it will be overridden
            import warnings

            warnings.warn(
                "MaxRL uses p-normalization (dividing by mean reward) for advantage calculation. "
                f"The `scale_rewards='{self.scale_rewards}'` setting will be ignored.",
                UserWarning,
            )
