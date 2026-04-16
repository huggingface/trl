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
Simple script to run GRPO training with OpenEnv's BrowserGym environment and vLLM for LLMs.

This script is optimized for text-only Language Models (LLMs). It uses the accessibility
tree text from BrowserGym, making it memory-efficient.

The environment runs on a Hugging Face Space by default.

Setup:

```sh
uv pip install git+https://huggingface.co/spaces/openenv/browsergym_env
```

Usage:

```sh
# HF Spaces + Colocated vLLM (1 GPU required)
python examples/scripts/openenv/browsergym_llm.py --vllm-mode colocate

# HF Spaces + Separate vLLM server (2 GPUs required)
# Terminal 1:
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8001
# Terminal 2:
CUDA_VISIBLE_DEVICES=1 python examples/scripts/openenv/browsergym_llm.py --vllm-mode server --vllm-server-url http://localhost:8001
```
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from browsergym_env import BrowserGymAction, BrowserGymEnv
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO training for BrowserGym MiniWoB using OpenEnv environment.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--space-url", default="https://openenv-browsergym-env.hf.space")
    parser.add_argument("--dataset-prompt", default="Complete the web task successfully.")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per episode.")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-url", default="http://localhost:8001")
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


SYSTEM_PROMPT = """You are interacting with a web page. Elements have numeric IDs shown in brackets like [13]. Use the available tools to complete the task."""


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

    class BrowserGymLLMEnv:
        def __init__(self):
            self.client = BrowserGymEnv(base_url=space_url)
            self.reward = 0.0
            self._done = False
            self._step_count = 0

        def _ensure_large_max_size(self):
            """Raise WebSocket max message size for large observations (e.g. accessibility trees).

            openenv-core<=0.2.1 does not pass max_size to ws_connect, so the websockets library
            defaults to 1MB. We force a connection and patch it to 100MB before any messages are sent.
            """
            self.client.connect()
            ws = self.client._ws
            if ws is not None and hasattr(ws, "protocol"):
                proto = ws.protocol
                # websockets <16: max_size; websockets >=16: max_message_size
                attr = "max_size" if hasattr(proto, "max_size") else "max_message_size"
                if getattr(proto, attr) == 2**20:
                    setattr(proto, attr, 100 * 1024 * 1024)

        def reset(self, **kwargs) -> str:
            self.reward = 0.0
            self._done = False
            self._step_count = 0
            self._ensure_large_max_size()
            result = self.client.reset()
            self._done = result.done
            return self._format_observation(result.observation)

        def click(self, bid: str) -> str:
            """Click an element on the page.

            Args:
                bid: The BrowserGym ID of the element to click.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"click({bid!r})")

        def fill(self, bid: str, text: str) -> str:
            """Fill an input field with text.

            Args:
                bid: The BrowserGym ID of the input field.
                text: The text to type into the field.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"fill({bid!r}, {text!r})")

        def send_keys(self, text: str) -> str:
            """Send keyboard input to the page.

            Args:
                text: The keyboard input to send.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"send_keys({text!r})")

        def scroll(self, direction: str) -> str:
            """Scroll the page.

            Args:
                direction: Direction to scroll, either 'up' or 'down'.

            Returns:
                The updated page observation.
            """
            return self._do_action(f"scroll({direction!r})")

        def noop(self) -> str:
            """Do nothing and observe the current page state.

            Returns:
                The current page observation.
            """
            return self._do_action("noop()")

        def _do_action(self, action_str: str) -> str:
            if self._done:
                return "Episode finished successfully. No further actions needed."

            self._step_count += 1
            result = self.client.step(BrowserGymAction(action_str=action_str))
            observation = result.observation
            step_reward = float(result.reward or 0.0)
            self._done = result.done

            if self._done and step_reward > 0:
                self.reward = 1.0
            elif self._done:
                self.reward = 0.0
            else:
                self.reward = step_reward

            if self._step_count >= max_steps:
                self._done = True

            return self._format_observation(observation)

        def _format_observation(self, observation) -> str:
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
            return "\n\n".join(parts) if parts else "No observation available."

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path("outputs") / f"browsergym-grpo-{sanitize_name(args.model_id)}-{timestamp}"
    output_dir = Path(args.output_dir or default_output_dir)

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_completion, reward_efficiency],
        train_dataset=dataset,
        args=GRPOConfig(
            use_vllm=True,
            vllm_mode=args.vllm_mode,
            vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
            vllm_gpu_memory_utilization=0.4,
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            logging_steps=args.logging_steps,
            log_completions=True,
            report_to="trackio",
            trackio_space_id=f"browsergym-llm-grpo-{sanitize_name(args.model_id)}",
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=BrowserGymLLMEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
