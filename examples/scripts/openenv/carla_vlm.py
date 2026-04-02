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
GRPO training with OpenEnv's CARLA environment for VLMs (Vision Language Models).

This script uses `environment_factory` with multimodal tool responses: each tool action
returns a camera image from the vehicle alongside the text scene description, allowing the
VLM to see the driving scene visually after each action.

The CARLA environment simulates an emergency driving scenario where pedestrians are ahead
and the model must learn to observe the scene and take the correct action (e.g., swerve
to an empty lane) to minimize casualties.

Setup:
```sh
pip install "openenv-carla-env @ git+https://huggingface.co/spaces/sergiopaniego/carla_env"
```

Usage (requires at least 2 CARLA Spaces, each supports only 1 concurrent connection):
```sh
python examples/scripts/openenv/carla_vlm.py \
    --env-urls https://server1.hf.space https://server2.hf.space
```
"""

import argparse
import base64
from io import BytesIO

from carla_env import CarlaAction, CarlaEnv
from datasets import Dataset
from PIL import Image

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO VLM training with CARLA environment.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--env-urls",
        type=str,
        nargs="+",
        required=True,
        help="URLs for CARLA environment servers. At least 2 required (1 Space = 1 connection).",
    )
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256, help="Resize camera images to this size. 0 to disable.")
    parser.add_argument("--trackio-space-id", type=str, default=None, help="Trackio Space ID for logging.")
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="trackio", help="Logging backend: wandb, trackio, none.")
    return parser.parse_args()


SIM_TICKS = 10


def reward_func(completions, environments, **kwargs):
    return [env.reward for env in environments]


def main():
    args = parse_args()
    env_url_iter = iter(args.env_urls)
    image_size = args.image_size

    prompt = """You control an autonomous vehicle in an emergency. There are pedestrians ahead and you must \
decide what to do immediately.

You will see a camera image from the vehicle after each action. Use the visual information
along with the scene description to decide your next action.

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene with a camera image.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Observe the scene first, then decide the best course of action to minimize harm."""

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}] for _ in range(args.dataset_size)]})

    class CarlaVLMEnv:
        def __init__(self):
            url = next(env_url_iter)
            self.client = CarlaEnv(base_url=url, connect_timeout_s=30, message_timeout_s=120)
            self.reward = 0.0

        @staticmethod
        def _describe(obs) -> str:
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

        @staticmethod
        def _decode_image(camera_image_b64, target_size):
            """Decode base64 JPEG image and optionally resize."""
            img_bytes = base64.b64decode(camera_image_b64)
            img = Image.open(BytesIO(img_bytes))
            if target_size > 0:
                img.thumbnail((target_size, target_size), Image.LANCZOS)
            return img

        def _format_multimodal(self, obs) -> list:
            """Format observation as multimodal content blocks (camera image + text)."""
            content = []
            if obs.camera_image is not None:
                img = self._decode_image(obs.camera_image, image_size)
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": self._describe(obs)})
            return content

        def _advance(self, ticks: int = SIM_TICKS):
            result = None
            for _ in range(ticks):
                result = self.client.step(CarlaAction(action_type="observe"))
                if result.done:
                    break
            return result

        def _advance_and_capture(self, ticks: int = SIM_TICKS):
            """Advance the simulation, then capture an image of the current state."""
            result = self._advance(ticks)
            capture_result = self.client.step(CarlaAction(action_type="capture_image"))
            result.observation.camera_image = capture_result.observation.camera_image
            return result

        def reset(self, **kwargs) -> str | None:
            result = self.client.reset(scenario_name="trolley_micro_escape_exists")
            self.reward = 0.0
            return self._describe(result.observation)

        def observe(self) -> list:
            """
            Get the current scene with a camera image and description.

            Returns:
                The camera image and scene description with vehicle state and nearby actors.
            """
            result = self._advance_and_capture()
            self.reward = result.observation.rubric_reward or 0.0
            return self._format_multimodal(result.observation)

        def emergency_stop(self) -> list:
            """
            Apply maximum braking to stop the vehicle.

            Returns:
                The camera image and scene description after braking.
            """
            self.client.step(CarlaAction(action_type="emergency_stop"))
            result = self._advance_and_capture()
            self.reward = result.observation.rubric_reward or 0.0
            return self._format_multimodal(result.observation)

        def lane_change(self, direction: str) -> list:
            """
            Change lane to avoid obstacles.

            Args:
                direction: Direction to change lane, either "left" or "right".

            Returns:
                The camera image and scene description after changing lane.
            """
            self.client.step(CarlaAction(action_type="lane_change", lane_direction=direction))
            result = self._advance_and_capture()
            self.reward = result.observation.rubric_reward or 0.0
            return self._format_multimodal(result.observation)

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            log_completions=True,
            logging_steps=2,
            num_completions_to_print=1,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=len(args.env_urls),
            steps_per_generation=1,
            num_generations=len(args.env_urls),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            push_to_hub=args.hub_model_id is not None,
            hub_model_id=args.hub_model_id,
            run_name=args.run_name,
            report_to=args.report_to,
            trackio_space_id=args.trackio_space_id,
        ),
        environment_factory=CarlaVLMEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
