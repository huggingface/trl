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

from dataclasses import dataclass, field

import numpy as np
from datasets import Audio, Dataset, Features, Value
from transformers import HfArgumentParser


Message = [{"content": Value("string"), "role": Value("string")}]

SAMPLING_RATE = 16000


def _make_audios(n: int) -> list[dict]:
    """Build n short random-noise audio clips (0.5-1.5s at 16 kHz)."""
    lengths = np.random.randint(SAMPLING_RATE // 2, SAMPLING_RATE * 3 // 2, size=n)
    return [
        {"array": np.random.uniform(-1.0, 1.0, size=length).astype(np.float32), "sampling_rate": SAMPLING_RATE}
        for length in lengths
    ]


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        test_size (`float`, *optional*, defaults to `0.1`):
            Fraction of the dataset to include in the test split.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-internal-testing/zen-audio"`):
            Hugging Face repository ID to push the dataset to.
    """

    test_size: float = field(
        default=0.1,
        metadata={"help": "Fraction of the dataset to include in the test split."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the dataset to the Hugging Face Hub."},
    )
    repo_id: str = field(
        default="trl-internal-testing/zen-audio",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )


def main(test_size, push_to_hub, repo_id):
    # fmt: off
    n = 19
    data = {
        "messages": [
            [{"role": "user", "content": "What is better than ugly?"}, {"role": "assistant", "content": "Beautiful."}],
            [{"role": "user", "content": "What is better than implicit?"}, {"role": "assistant", "content": "Explicit."}],
            [{"role": "user", "content": "What is better than complex?"}, {"role": "assistant", "content": "Simple."}],
            [{"role": "user", "content": "What is better than complicated?"}, {"role": "assistant", "content": "Complex."}],
            [{"role": "user", "content": "What is better than nested?"}, {"role": "assistant", "content": "Flat."}],
            [{"role": "user", "content": "What is better than dense?"}, {"role": "assistant", "content": "Sparse."}],
            [{"role": "user", "content": "What counts?"}, {"role": "assistant", "content": "Readability."}],
            [{"role": "user", "content": "Are special cases enough to break the rules?"}, {"role": "assistant", "content": "No, special cases aren't special enough to break the rules."}],
            [{"role": "user", "content": "What beats purity?"}, {"role": "assistant", "content": "Practicality."}],
            [{"role": "user", "content": "What should never pass silently?"}, {"role": "assistant", "content": "Errors."}],
            [{"role": "user", "content": "When can errors pass silently?"}, {"role": "assistant", "content": "When explicitly silenced."}],
            [{"role": "user", "content": "What should you do in the face of ambiguity?"}, {"role": "assistant", "content": "Refuse the temptation to guess."}],
            [{"role": "user", "content": "How many ways should there be to do it?"}, {"role": "assistant", "content": "One, and preferably only one."}],
            [{"role": "user", "content": "For whom may the way not be obvious at first?"}, {"role": "assistant", "content": "Dutch."}],
            [{"role": "user", "content": "What is better than never?"}, {"role": "assistant", "content": "Now is better than never."}],
            [{"role": "user", "content": "Is never better than *right* now?"}, {"role": "assistant", "content": "Yes, often."}],
            [{"role": "user", "content": "What does it mean if the implementation is hard to explain?"}, {"role": "assistant", "content": "It means it's a bad idea."}],
            [{"role": "user", "content": "What does it mean if the implementation is easy to explain?"}, {"role": "assistant", "content": "It means it may be a good idea."}],
            [{"role": "user", "content": "Any great ideas?"}, {"role": "assistant", "content": "Namespaces are one honking great idea."}],
        ],
        "audio": _make_audios(n),
    }
    conversational_language_modeling_dataset = Dataset.from_dict(data, features=Features(messages=Message, audio=Audio(sampling_rate=SAMPLING_RATE)))
    conversational_language_modeling_dataset = conversational_language_modeling_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_language_modeling_dataset.push_to_hub(repo_id, config_name="conversational_language_modeling")

    data = {
        "prompt": [
            [{"role": "user", "content": "What is better than ugly?"}],
            [{"role": "user", "content": "What is better than implicit?"}],
            [{"role": "user", "content": "What is better than complex?"}],
            [{"role": "user", "content": "What is better than complicated?"}],
            [{"role": "user", "content": "What is better than nested?"}],
            [{"role": "user", "content": "What is better than dense?"}],
            [{"role": "user", "content": "What counts?"}],
            [{"role": "user", "content": "Are special cases enough to break the rules?"}],
            [{"role": "user", "content": "What beats purity?"}],
            [{"role": "user", "content": "What should never pass silently?"}],
            [{"role": "user", "content": "When can errors pass silently?"}],
            [{"role": "user", "content": "What should you do in the face of ambiguity?"}],
            [{"role": "user", "content": "How many ways should there be to do it?"}],
            [{"role": "user", "content": "For whom may the way not be obvious at first?"}],
            [{"role": "user", "content": "What is better than never?"}],
            [{"role": "user", "content": "Is never better than *right* now?"}],
            [{"role": "user", "content": "What does it mean if the implementation is hard to explain?"}],
            [{"role": "user", "content": "What does it mean if the implementation is easy to explain?"}],
            [{"role": "user", "content": "Any great ideas?"}],
        ],
        "audio": _make_audios(n),
    }
    conversational_prompt_only_dataset = Dataset.from_dict(data, features=Features(prompt=Message, audio=Audio(sampling_rate=SAMPLING_RATE)))
    conversational_prompt_only_dataset = conversational_prompt_only_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_only_dataset.push_to_hub(repo_id, config_name="conversational_prompt_only")

    data = {
        "prompt": [
            [{"role": "user", "content": "What is better than ugly?"}],
            [{"role": "user", "content": "What is better than implicit?"}],
            [{"role": "user", "content": "What is better than complex?"}],
            [{"role": "user", "content": "What is better than complicated?"}],
            [{"role": "user", "content": "What is better than nested?"}],
            [{"role": "user", "content": "What is better than dense?"}],
            [{"role": "user", "content": "What counts?"}],
            [{"role": "user", "content": "Are special cases enough to break the rules?"}],
            [{"role": "user", "content": "What beats purity?"}],
            [{"role": "user", "content": "What should never pass silently?"}],
            [{"role": "user", "content": "When can errors pass silently?"}],
            [{"role": "user", "content": "What should you do in the face of ambiguity?"}],
            [{"role": "user", "content": "How many ways should there be to do it?"}],
            [{"role": "user", "content": "For whom may the way not be obvious at first?"}],
            [{"role": "user", "content": "What is better than never?"}],
            [{"role": "user", "content": "Is never better than *right* now?"}],
            [{"role": "user", "content": "What does it mean if the implementation is hard to explain?"}],
            [{"role": "user", "content": "What does it mean if the implementation is easy to explain?"}],
            [{"role": "user", "content": "Any great ideas?"}],
        ],
        "completion": [
            [{"role": "assistant", "content": "Beautiful."}],
            [{"role": "assistant", "content": "Explicit."}],
            [{"role": "assistant", "content": "Simple."}],
            [{"role": "assistant", "content": "Complex."}],
            [{"role": "assistant", "content": "Flat."}],
            [{"role": "assistant", "content": "Sparse."}],
            [{"role": "assistant", "content": "Readability."}],
            [{"role": "assistant", "content": "No, special cases aren't special enough to break the rules."}],
            [{"role": "assistant", "content": "Practicality."}],
            [{"role": "assistant", "content": "Errors."}],
            [{"role": "assistant", "content": "When explicitly silenced."}],
            [{"role": "assistant", "content": "Refuse the temptation to guess."}],
            [{"role": "assistant", "content": "One, and preferably only one."}],
            [{"role": "assistant", "content": "Dutch."}],
            [{"role": "assistant", "content": "Now is better than never."}],
            [{"role": "assistant", "content": "Yes, often."}],
            [{"role": "assistant", "content": "It means it's a bad idea."}],
            [{"role": "assistant", "content": "It means it may be a good idea."}],
            [{"role": "assistant", "content": "Namespaces are one honking great idea."}],
        ],
        "audio": _make_audios(n),
    }
    conversational_prompt_completion_dataset = Dataset.from_dict(data, features=Features(prompt=Message, completion=Message, audio=Audio(sampling_rate=SAMPLING_RATE)))
    conversational_prompt_completion_dataset = conversational_prompt_completion_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_completion_dataset.push_to_hub(repo_id, config_name="conversational_prompt_completion")
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
