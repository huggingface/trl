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
#     "trl[vllm,peft]",
#     "trackio",
#     "kernels",
#     "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env",
# ]
# ///

"""
GRPO training with OpenEnv's BrowserGym environment for VLMs (Vision Language Models).

This script uses `environment_factory` with multimodal tool responses: each tool action
returns a screenshot (PIL Image) alongside the accessibility tree text, allowing the VLM
to see the page visually after each action.

Setup:
```sh
pip install "openenv-browsergym @ git+https://huggingface.co/spaces/openenv/browsergym_env"
```

Usage:
```sh
# Without vLLM (default, 1 GPU)
python examples/scripts/openenv/browsergym.py

# With vLLM colocate (1 GPU, requires vLLM support for the model)
python examples/scripts/openenv/browsergym.py --use-vllm

# With vLLM server (2 GPUs)
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3.5-2B --host 0.0.0.0 --port 8000
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/browsergym.py --use-vllm --vllm-mode server
```
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from browsergym_env import BrowserGymAction, BrowserGymEnv
from datasets import Dataset
from PIL import Image

from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training with BrowserGym VLM environment.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--space-url", default="https://openenv-browsergym-env.hf.space")
    parser.add_argument("--dataset-prompt", default="Complete the web task successfully.")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per episode.")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=512, help="Resize screenshots to this size. 0 to disable.")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-vllm", action="store_true", default=False, help="Enable vLLM for generation.")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-url", default="http://localhost:8000")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


SYSTEM_PROMPT = """You are interacting with a web page. Use the available tools to complete the task."""


def reward_completion(completions, environments, **kwargs) -> list[float]:
    """Reward for task completion."""
    return [env.reward for env in environments]


def reward_efficiency(completions, **kwargs) -> list[float]:
    """Penalize extra tool calls beyond the first one."""
    rewards = []
    for comp in completions:
        n_tool_calls = sum(1 for m in comp if isinstance(m, dict) and m.get("tool_calls"))
        extra_calls = max(0, n_tool_calls - 1)
        rewards.append(-0.1 * extra_calls)
    return rewards


def main() -> None:
    args = parse_args()

    space_url = args.space_url
    max_steps = args.max_steps
    image_size = args.image_size

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": args.dataset_prompt},
                ]
            ]
            * args.dataset_size
        }
    )

    class BrowserGymVLMEnv:
        def __init__(self):
            self.client = BrowserGymEnv(base_url=space_url)
            self.reward = 0.0
            self.done = False
            self._step_count = 0

        def _ensure_large_max_size(self):
            """Raise WebSocket max message size for large observations (screenshots + axtree).

            openenv-core<=0.2.1 does not pass max_size to ws_connect, so the websockets library
            defaults to 1MB. We force a connection and patch it to 100MB before any messages are sent.
            """
            import websockets

            self.client.connect()
            ws = self.client._ws
            if ws is not None and ws.protocol is not None:
                proto = ws.protocol
                # websockets renamed max_size to max_message_size in version 16
                if int(websockets.__version__.split(".")[0]) >= 16:
                    if proto.max_message_size == 2**20:
                        proto.max_message_size = 100 * 1024 * 1024
                else:
                    if proto.max_size == 2**20:
                        proto.max_size = 100 * 1024 * 1024

        def reset(self, **kwargs) -> str | None:
            self.reward = 0.0
            self.done = False
            self._step_count = 0
            self._ensure_large_max_size()
            result = self.client.reset()
            self.done = result.done
            return self._format_observation(result.observation)

        @staticmethod
        def _normalize_bid(bid) -> str:
            """Coerce bid to a clean string.

            Qwen-style XML tool calls may parse the bid as int (e.g. 13) or as
            a stringified list ("[13]") copied verbatim from the axtree. BrowserGym
            expects a plain string id, so strip brackets and unwrap 1-element lists.
            """
            if isinstance(bid, list) and len(bid) == 1:
                bid = bid[0]
            return str(bid).strip().strip("[]")

        def click(self, bid: str) -> list | str:
            """Click an element on the page.

            Args:
                bid: The BrowserGym ID of the element to click.

            Returns:
                The updated page observation with screenshot.
            """
            bid = self._normalize_bid(bid)
            return self._do_action(f"click({bid!r})")

        def fill(self, bid: str, text: str) -> list | str:
            """Fill an input field with text.

            Args:
                bid: The BrowserGym ID of the input field.
                text: The text to type into the field.

            Returns:
                The updated page observation with screenshot.
            """
            bid = self._normalize_bid(bid)
            return self._do_action(f"fill({bid!r}, {text!r})")

        def send_keys(self, text: str) -> list | str:
            """Send keyboard input to the page.

            Args:
                text: The keyboard input to send.

            Returns:
                The updated page observation with screenshot.
            """
            return self._do_action(f"send_keys({text!r})")

        def scroll(self, direction: str) -> list | str:
            """Scroll the page.

            Args:
                direction: Direction to scroll, either 'up' or 'down'.

            Returns:
                The updated page observation with screenshot.
            """
            return self._do_action(f"scroll({direction!r})")

        def noop(self) -> list | str:
            """Do nothing and observe the current page state.

            Returns:
                The current page observation with screenshot.
            """
            return self._do_action("noop()")

        def _do_action(self, action_str: str) -> list | str:
            if self.done:
                return "Episode finished successfully. No further actions needed."

            self._step_count += 1
            result = self.client.step(BrowserGymAction(action_str=action_str))
            observation = result.observation
            step_reward = float(result.reward or 0.0)
            self.done = result.done

            if self.done and step_reward > 0:
                self.reward = 1.0
            elif self.done:
                self.reward = 0.0
            else:
                self.reward = step_reward

            if self._step_count >= max_steps:
                self.done = True

            return self._format_observation_multimodal(observation)

        def _format_observation(self, observation) -> str:
            """Format initial observation as text (for reset, appended to prompt)."""
            parts = []
            if observation.goal:
                parts.append(f"Goal: {observation.goal}")
            if observation.axtree_txt:
                axtree = observation.axtree_txt
                if len(axtree) > 2000:
                    axtree = axtree[:2000] + "..."
                parts.append(f"Page structure:\n{axtree}")
            return "\n\n".join(parts) if parts else "No observation available."

        def _format_observation_multimodal(self, observation) -> list:
            """Format observation as multimodal content blocks (screenshot + text)."""
            content = []

            if observation.screenshot is not None:
                screenshot_array = np.array(observation.screenshot, dtype=np.uint8)
                screenshot_image = Image.fromarray(screenshot_array)
                if image_size > 0:
                    screenshot_image.thumbnail((image_size, image_size), Image.LANCZOS)
                content.append({"type": "image", "image": screenshot_image})

            parts = []
            if observation.goal:
                parts.append(f"Goal: {observation.goal}")
            if observation.last_action_error and observation.error:
                parts.append(f"Error: {observation.error}")
            if observation.axtree_txt:
                axtree = observation.axtree_txt
                if len(axtree) > 2000:
                    axtree = axtree[:2000] + "..."
                parts.append(f"Page structure:\n{axtree}")
            text = "\n\n".join(parts) if parts else "No observation available."
            content.append({"type": "text", "text": text})

            return content

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"browsergym-vlm-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_completion, reward_efficiency],
        train_dataset=dataset,
        args=GRPOConfig(
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode if args.use_vllm else "colocate",
            vllm_server_base_url=args.vllm_server_url if args.use_vllm and args.vllm_mode == "server" else None,
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            logging_steps=args.logging_steps,
            log_completions=True,
            report_to="trackio",
            trackio_space_id=f"browsergym-vlm-grpo-{sanitize_name(args.model_id)}",
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=BrowserGymVLMEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
