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

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/openenv/browsergym_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/browsergym_env
uv pip install -e .
```

# Option 1: HF Spaces + Colocated vLLM (1 GPU required)
```sh
python examples/scripts/openenv/browsergym_llm.py --vllm-mode colocate
```

# Option 2: HF Spaces + Separate vLLM server (2 GPUs required)

# Spin up vLLM server (Terminal 1)
```sh
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8001
```

# Run training (Terminal 2)
```sh
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
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-0.6B",
        help="Model identifier passed to GRPOTrainer for fine-tuning.",
    )
    parser.add_argument(
        "--space-url",
        type=str,
        default="https://openenv-browsergym-env.hf.space",
        help="URL for the Hugging Face Space running the BrowserGym environment.",
    )
    parser.add_argument(
        "--benchmark",
        default="miniwob",
        help="BrowserGym benchmark to use (miniwob, webarena, etc.).",
    )
    parser.add_argument(
        "--task-name",
        default="click-test",
        help="Specific task within the benchmark (e.g., click-test, click-button).",
    )
    parser.add_argument(
        "--dataset-prompt",
        default="Complete the web task successfully.",
        help="Prompt text used to seed the training dataset.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1000,
        help="Number of entries to include in the synthetic training dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="Maximum completion length in tokens for tool-calling generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature used during rollout generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter forwarded to vLLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top-p sampling parameter forwarded to vLLM.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate for GRPO training.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied during optimization.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation steps for GRPO training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Warmup steps for the scheduler.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of rollout generations per dataset prompt.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Interval (in steps) between checkpoint saves.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where training outputs and checkpoints are stored.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name for logging systems.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional project identifier for logging systems.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help="vLLM execution mode: 'colocate' or 'server'.",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8001",
        help="URL for the vLLM server (only used when --vllm-mode=server).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Frequency of logging steps for GRPO training.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You control a web browser to complete tasks.

The page structure shows elements as: [bid] element_type 'element_text'
For example: [13] button 'Click Me!' means the element has bid='13'.

Use the available tools to interact with the page:
- click: Click an element by its bid
- fill: Fill an input field with text
- send_keys: Send keyboard input
- scroll: Scroll the page
- noop: Do nothing

Complete the given task as efficiently as possible."""


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def reward_completion(environments, **kwargs) -> list[float]:
    """Reward for task completion."""
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


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
                raise ValueError("Episode is done.")

            self._step_count += 1
            result = self.client.step(BrowserGymAction(action_str=action_str))
            observation = result.observation
            step_reward = float(result.reward or 0.0)
            self._done = result.done

            # Reward shaping: binary success/failure on completion
            if self._done and step_reward > 0:
                self.reward = 1.0
            elif self._done:
                self.reward = 0.0
            else:
                self.reward = step_reward

            # Enforce max steps
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

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=0.4,
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        generation_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=args.logging_steps,
        report_to="trackio",
        trackio_space_id=f"browsergym-grpo-{sanitize_name(args.model_id)}-{timestamp}",
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=args.save_total_limit,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        chat_template_kwargs={"enable_thinking": False},
    )

    grpo_config.run_name = args.run_name or f"run-{timestamp}"
    grpo_config.project = args.project or f"group-{sanitize_name(args.model_id)}"

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=[reward_completion],
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=BrowserGymLLMEnv,
    )

    print("=" * 80)
    print("Starting GRPO training with BrowserGym environment (LLM mode)")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model_id}")
    print("Mode: LLM (text-only, using accessibility tree)")
    print(f"Using {args.num_generations} rollouts per dataset prompt")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    trainer.train()


if __name__ == "__main__":
    main()
