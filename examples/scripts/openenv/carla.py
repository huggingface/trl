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
Simple script to run GRPO training with OpenEnv's CARLA environment. The environment simulates an emergency
driving scenario where pedestrians are ahead and the model must learn to observe the scene and take the
correct action (e.g., swerve to an empty lane) to minimize casualties.

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
    parser.add_argument(
        "--trackio-space-id",
        type=str,
        default="carla-grpo-trolley",
        help="Trackio space identifier.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Hub model ID to push the trained model to (e.g., sergiopaniego/Qwen3-0.6B-carla-trolley-escape).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for tracking.",
    )
    return parser.parse_args()


args = parse_args()
_env_url_iter = iter(args.env_urls)  # Each instance takes the next URL

prompt = """You control an autonomous vehicle in an emergency. There are pedestrians ahead and you must \
decide what to do immediately.

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Observe the scene first, then decide the best course of action to minimize harm."""

dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(1000)]})


SIM_TICKS = 10  # Number of simulation steps to advance after each action


class CarlaGRPOEnv:
    def __init__(self):
        url = next(_env_url_iter)
        self.client = CarlaEnv(base_url=url, connect_timeout_s=30, message_timeout_s=120)

    @staticmethod
    def _describe(obs) -> str:
        """Build a text description from the observation fields."""
        parts = []
        parts.append(f"Speed: {obs.speed_kmh:.1f} km/h.")
        if obs.nearby_actors:
            for actor in obs.nearby_actors:
                parts.append(f"- {actor.get('type', 'actor')} at {actor.get('distance', '?')}m")
        else:
            parts.append("No nearby actors detected.")
        if obs.collision_detected:
            parts.append(f"COLLISION detected with {obs.collided_with or 'unknown'}!")
        return "\n".join(parts)

    def _advance(self, ticks: int = SIM_TICKS):
        """Advance the simulation by calling observe repeatedly, return the last result."""
        result = None
        for _ in range(ticks):
            result = self.client.step(CarlaAction(action_type="observe"))
            if result.done:
                break
        return result

    def reset(self, **kwargs) -> str | None:
        result = self.client.reset(scenario_name="trolley_micro_escape_exists")
        self.reward = 0.0
        return self._describe(result.observation)

    def observe(self) -> str:
        """
        Get the current scene description without taking any action.

        Returns:
            The scene description with vehicle state and nearby actors.
        """
        result = self._advance()
        self.reward = result.observation.rubric_reward or 0.0
        return self._describe(result.observation)

    def emergency_stop(self) -> str:
        """
        Apply maximum braking to stop the vehicle.

        Returns:
            The scene description after braking.
        """
        self.client.step(CarlaAction(action_type="emergency_stop"))
        result = self._advance()
        self.reward = result.observation.rubric_reward or 0.0
        return self._describe(result.observation)

    def lane_change(self, direction: str) -> str:
        """
        Change lane to avoid obstacles.

        Args:
            direction: Direction to change lane, either "left" or "right".

        Returns:
            The scene description after changing lane.
        """
        self.client.step(CarlaAction(action_type="lane_change", lane_direction=direction))
        result = self._advance()
        self.reward = result.observation.rubric_reward or 0.0
        return self._describe(result.observation)


def reward_func(completions, environments, **kwargs):
    return [environment.reward for environment in environments]


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
        gradient_accumulation_steps=16,
        max_steps=50,
        push_to_hub=args.hub_model_id is not None,
        hub_model_id=args.hub_model_id,
        run_name=args.run_name,
        report_to="trackio",
        trackio_space_id=args.trackio_space_id,
    ),
    environment_factory=CarlaGRPOEnv,
)
trainer.train()
