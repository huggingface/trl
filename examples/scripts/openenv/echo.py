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
#     "openenv-echo-env @ git+https://huggingface.co/spaces/qgallouedec/echo_env",
# ]
# ///


"""
Simple script to run GRPO training with OpenEnv's Echo environment. The environment echoes back the message
sent to it and rewards longer completions.

Setup (Option A - Install from HF Space, recommended):

```sh
uv pip install git+https://huggingface.co/spaces/qgallouedec/echo_env
```

Setup (Option B - Clone OpenEnv repo, for development):

```sh
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv/envs/echo_env
uv pip install -e .
```

Usage:

```sh
python examples/scripts/openenv/echo.py
python examples/scripts/openenv/echo.py --model Qwen/Qwen2.5-0.5B-Instruct --env-host https://qgallouedec-echo-env.hf.space
```
"""

import argparse

from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with Echo environment.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model to use for training.",
    )
    parser.add_argument(
        "--env-host",
        type=str,
        default="https://qgallouedec-echo-env.hf.space",
        help="URL for the Echo environment HF Space.",
    )
    return parser.parse_args()


def reward_func(environments, **kwargs):
    return [env.reward for env in environments]


def main():
    args = parse_args()

    dataset = Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "Try to echo 'Hello World!' in the environment."}],
                [{"role": "user", "content": "Make the environment echo 'Goodbye World!'"}],
                [{"role": "user", "content": "Can you ask the environment to echo 'TRL is great!'?"}],
                [{"role": "user", "content": "What happens if you ask the environment to echo 'I love RLHF!'?"}],
                [{"role": "user", "content": "Try to make the environment echo 'OpenEnv is awesome!'"}],
            ],
        }
    )

    class EchoToolEnv:
        def __init__(self):
            self.env = EchoEnv(base_url=args.env_host)
            self.reward = 0.0

        def reset(self, **kwargs) -> None | str:
            self.reward = 0.0
            return None

        def echo(self, message: str) -> str:
            """
            Echo the message back from the environment.

            Args:
                message: The message to echo

            Returns:
                The echoed message.
            """
            observation = self.env.step(EchoAction(message=message))
            self.reward = observation.observation.reward
            return observation.observation.echoed_message

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": False},
            log_completions=True,
            logging_steps=2,
            num_completions_to_print=1,
        ),
        environment_factory=EchoToolEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
