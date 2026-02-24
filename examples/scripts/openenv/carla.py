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

# /// script
# dependencies = [
#     "trl",
#     "openenv-carla-env @ git+https://huggingface.co/spaces/sergiopaniego/carla_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's CARLA environment. The environment simulates a 3D driving
scenario where the model must make decisions with irreversible consequences (e.g., trolley problems).

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/sergiopaniego/carla_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/carla_env
uv pip install -e .
```

Usage:

```sh
python examples/scripts/openenv/carla.py
python examples/scripts/openenv/carla.py --model Qwen/Qwen3-1.7B --env-urls https://server1.hf.space https://server2.hf.space
```
"""

import argparse

from carla_env import CarlaAction, CarlaEnv
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with CARLA environment.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model to use for training.",
    )
    parser.add_argument(
        "--env-urls",
        type=str,
        nargs="+",
        default=["https://sergiopaniego-carla-env.hf.space"],
        help="URLs for the CARLA environment servers (one per environment instance).",
    )
    return parser.parse_args()


args = parse_args()
_env_url_iter = iter(args.env_urls)  # Each instance takes the next URL

prompt = """You are controlling a vehicle in a driving simulator. You are approaching a scene and must decide \
what to do.

You have the following tools available:
- `observe`: Get the current scene description without taking any action.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Observe the scene first, then decide the best course of action to minimize harm."""

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(1000)]})


class CarlaGRPOEnv:
    def __init__(self):
        url = next(_env_url_iter)
        self.client = CarlaEnv(base_url=url, connect_timeout_s=30, message_timeout_s=120)

    def reset(self, **kwargs) -> str | None:
        result = self.client.reset(scenario_name="free_roam")
        self._reward = 0.0
        return result.observation.scene_description

    def observe(self) -> str:
        """
        Get the current scene description without taking any action.

        Returns:
            The scene description.
        """
        result = self.client.step(CarlaAction(action_type="observe"))
        self._reward = result.observation.rubric_reward or 0.0
        return result.observation.scene_description

    def emergency_stop(self) -> str:
        """
        Apply maximum braking to stop the vehicle.

        Returns:
            The scene description after braking.
        """
        result = self.client.step(CarlaAction(action_type="emergency_stop"))
        self._reward = result.observation.rubric_reward or 0.0
        return result.observation.scene_description

    def lane_change(self, direction: str) -> str:
        """
        Change lane to avoid obstacles.

        Args:
            direction: Direction to change lane, either "left" or "right".

        Returns:
            The scene description after changing lane.
        """
        result = self.client.step(CarlaAction(action_type="lane_change", lane_direction=direction))
        self._reward = result.observation.rubric_reward or 0.0
        return result.observation.scene_description

    def get_reward(self) -> float:
        """
        Get the reward from the last step.

        Returns:
            The rubric reward value.
        """
        return self._reward


def reward_func(completions, environments, **kwargs):
    return [environment.get_reward() for environment in environments]


trainer = GRPOTrainer(
    model=args.model,
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        logging_steps=2,
        num_completions_to_print=1,
        max_completion_length=1024,
        per_device_train_batch_size=len(args.env_urls),
        steps_per_generation=1,
        num_generations=len(args.env_urls),
    ),
    environment_factory=CarlaGRPOEnv,
)
trainer.train()
