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

# pip install git+https://github.com/huggingface/transformers.git@main

from datasets import Dataset
from echo_env import EchoEnv
from echo_env.models import EchoAction

from trl import GRPOConfig, GRPOTrainer


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


def reward_func(completions, environments, **kwargs):
    return [environment.get_reward() for environment in environments]


class MyEchoEnv:
    def __init__(self):
        self.env = EchoEnv(base_url="https://qgallouedec-echo-env.hf.space")

    def reset(self, **kwargs) -> None:
        self._reward = None

    def step(self, message: str) -> str:
        """
        Echo the message back from the environment.

        Args:
            message: The message to echo

        Returns:
            The echoed message.
        """
        observation = self.env.step(EchoAction(message=message))
        self._reward = observation.observation.reward
        return observation.observation.echoed_message

    def get_reward(self) -> float:
        """
        Get the reward from the last step.

        Returns:
            The reward value.
        """
        return self._reward


trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        logging_steps=2,
        num_completions_to_print=1,
    ),
    environment_factory=MyEchoEnv,
)
trainer.train()
