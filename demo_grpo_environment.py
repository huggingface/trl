# pip install a specific transformers branch: pip install git+https://github.com/huggingface/transformers.git@json-schema-support-methods

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset


dataset = Dataset.from_dict(
    {
        "prompt": [
            [{"role": "user", "content": "Increment by one."}],
            [{"role": "user", "content": "Increment by two."}],
            [{"role": "user", "content": "Increment by three."}],
            [{"role": "user", "content": "Increment by four."}],
            [{"role": "user", "content": "Increment by five."}],
            [{"role": "user", "content": "Increment by six."}],
            [{"role": "user", "content": "Increment by seven."}],
            [{"role": "user", "content": "Increment by eight."}],
            [{"role": "user", "content": "Increment by nine."}],
            [{"role": "user", "content": "Increment by ten."}],
            [{"role": "user", "content": "Increment by eleven."}],
        ],
    }
)


def reward_func(completions, **kwargs):
    return [0.0 for _ in completions]


class IncrementEnv:
    def __init__(self):
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def increment(self, a: int) -> int:
        """
        Increment the integer by a specified value

        Args:
            a: The value to increment by

        Returns:
            The incremented integer.
        """
        self._counter += a
        return self._counter


trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    reward_funcs=reward_func,
    args=GRPOConfig(chat_template_kwargs={"enable_thinking": False}),
    environment_factory=IncrementEnv,
)
trainer.train()
