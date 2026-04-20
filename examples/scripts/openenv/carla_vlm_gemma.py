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
GRPO training with OpenEnv's CARLA environment for Gemma VLMs.

Gemma-specific variant of `carla_vlm.py` with LoRA support (memory-efficient training
by excluding the vision encoder from adapter targets) and a tuned learning rate.

Setup:

```sh
uv pip install git+https://huggingface.co/spaces/sergiopaniego/carla_env
```

Usage (requires at least 2 CARLA Spaces, each supports only 1 concurrent connection):

```sh
python examples/scripts/openenv/carla_vlm_gemma.py \
    --env-urls https://server1.hf.space https://server2.hf.space \
    --use-lora
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
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--env-urls",
        type=str,
        nargs="+",
        required=True,
        help="URLs for CARLA environment servers. At least 2 required (1 Space = 1 connection).",
    )
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-completion-length", type=int, default=3072)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None, help="Defaults to len(env-urls).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256, help="Resize camera images to this size. 0 to disable.")
    parser.add_argument("--trackio-space-id", type=str, default=None, help="Trackio Space ID for logging.")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for memory-efficient training.")
    parser.add_argument("--lora-r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=256, help="LoRA alpha.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="llm-only",
        help="LoRA target modules. Use 'llm-only' to skip vision encoder, 'all-linear' for all.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-6, help="Learning rate. Default 5e-6 (good for LoRA r=128)."
    )
    parser.add_argument("--hub-model-id", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--report-to", type=str, default="trackio", help="Logging backend: wandb, trackio, none.")
    return parser.parse_args()


PROMPT = """\
You control an autonomous vehicle in an emergency. There are pedestrians ahead and you must \
decide what to do immediately.

You will see a camera image from the vehicle after each action. Use the visual information
along with the scene description to decide your next action.

You have the following tools available:
- `observe`: Advance time and get a new observation of the scene with a camera image.
- `emergency_stop`: Apply maximum braking to stop the vehicle.
- `lane_change(direction)`: Change lane to the left or right. Direction must be "left" or "right".

Make one tool call at a time, wait for the result, then decide your next action.
Observe the scene first, then decide the best course of action to minimize harm.
Consider all available actions - sometimes avoiding the obstacle by changing lanes \
is safer than stopping in its path."""


SIM_TICKS = 10  # Number of simulation steps to advance after each action


class CarlaVLMEnv:
    _env_url_iter = None
    _image_size = 256

    def __init__(self):
        self.url = next(CarlaVLMEnv._env_url_iter)
        self.client = CarlaEnv(base_url=self.url, connect_timeout_s=30, message_timeout_s=120)
        self.reward = 0.0

    @staticmethod
    def _describe(obs) -> str:
        parts = [f"Speed: {obs.speed_kmh:.1f} km/h."]
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
        img = Image.open(BytesIO(base64.b64decode(camera_image_b64)))
        if target_size > 0:
            img.thumbnail((target_size, target_size), Image.LANCZOS)
        return img

    def _format_multimodal(self, obs) -> list:
        """Format observation as multimodal content blocks (camera image + text)."""
        content = []
        if obs.camera_image is not None:
            content.append({"type": "image", "image": self._decode_image(obs.camera_image, CarlaVLMEnv._image_size)})
        content.append({"type": "text", "text": self._describe(obs)})
        return content

    def _advance_and_capture(self, ticks: int = SIM_TICKS):
        """Advance the simulation, then capture an image of the current state."""
        result = None
        for _ in range(ticks):
            result = self.client.step(CarlaAction(action_type="observe"))
            if result.done:
                break
        capture_result = self.client.step(CarlaAction(action_type="capture_image"))
        result.observation.camera_image = capture_result.observation.camera_image
        return result

    def reset(self, **kwargs) -> str | None:
        for attempt in range(3):
            try:
                result = self.client.reset(scenario_name="trolley_micro_escape_exists")
                self.reward = 0.0
                return self._describe(result.observation)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[WARN] reset failed (attempt {attempt + 1}/3): {e}. Reconnecting...")
                self.client = CarlaEnv(base_url=self.url, connect_timeout_s=30, message_timeout_s=120)

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
        print(f"[DEBUG env] emergency_stop: done={result.done}, reward={self.reward}")
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
        print(f"[DEBUG env] lane_change({direction}): done={result.done}, reward={self.reward}")
        return self._format_multimodal(result.observation)


def reward_func(completions, environments, **kwargs):
    rewards = [env.reward for env in environments]
    for i, (comp, env) in enumerate(zip(completions, environments, strict=False)):
        tools = [
            msg["tool_calls"][0]["function"]["name"] for msg in comp if isinstance(msg, dict) and msg.get("tool_calls")
        ]
        print(f"[DEBUG reward] gen={i} tools={tools} env_reward={env.reward}")
    return rewards


def main():
    args = parse_args()
    CarlaVLMEnv._env_url_iter = iter(args.env_urls)
    CarlaVLMEnv._image_size = args.image_size

    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(args.dataset_size)]})

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig

        if args.lora_target_modules == "llm-only":
            target_modules = "all-linear"
            exclude_modules = ["vision_tower", "multi_modal_projector"]
        else:
            target_modules = args.lora_target_modules
            exclude_modules = None

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=peft_config,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            log_completions=True,
            logging_steps=2,
            num_completions_to_print=1,
            max_completion_length=args.max_completion_length,
            per_device_train_batch_size=args.per_device_train_batch_size or len(args.env_urls),
            steps_per_generation=1,
            num_generations=len(args.env_urls),
            max_tool_calling_iterations=10,
            learning_rate=args.learning_rate,
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
