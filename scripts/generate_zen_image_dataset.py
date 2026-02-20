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
from datasets import Dataset, Features, Image, List, Value
from transformers import HfArgumentParser


Message = [{"content": Value("string"), "role": Value("string")}]


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        test_size (`float`, *optional*, defaults to `0.1`):
            Fraction of the dataset to include in the test split.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-internal-testing/zen-image"`):
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
        default="trl-internal-testing/zen-image",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )


def main(test_size, push_to_hub, repo_id):
    # fmt: off
    sizes = np.random.randint(32, 64, size=(19, 2))
    data = {
        "messages": [
            [{"role": "user", "content": "What is better than ugly?"}, {"role": "assistant", "content": "Beautiful."},],
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
        "images": [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8)] for h, w in sizes],
    }
    conversational_language_modeling_dataset = Dataset.from_dict(data, features=Features(messages=Message, images=List(Image())))
    conversational_language_modeling_dataset = conversational_language_modeling_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_language_modeling_dataset.push_to_hub(repo_id, config_name="conversational_language_modeling")

    sizes = np.random.randint(32, 64, size=(19, 2))
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
        "images": [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8)] for h, w in sizes],
    }
    conversational_prompt_only_dataset = Dataset.from_dict(data, features=Features(prompt=Message, images=List(Image())))
    conversational_prompt_only_dataset = conversational_prompt_only_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_only_dataset.push_to_hub(repo_id, config_name="conversational_prompt_only")

    sizes = np.random.randint(32, 64, size=(19, 2))
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
        "images": [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8)] for h, w in sizes],
    }
    conversational_prompt_completion_dataset = Dataset.from_dict(data, features=Features(prompt=Message, completion=Message, images=List(Image())))
    conversational_prompt_completion_dataset = conversational_prompt_completion_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_completion_dataset.push_to_hub(repo_id, config_name="conversational_prompt_completion")

    sizes = np.random.randint(32, 64, size=(19, 2))
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
        "chosen": [
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
        "rejected": [
            [{"role": "assistant", "content": "Acceptable."}],
            [{"role": "assistant", "content": "Explained."}],
            [{"role": "assistant", "content": "Very complex."}],
            [{"role": "assistant", "content": "Very complicated."}],
            [{"role": "assistant", "content": "Circular."}],
            [{"role": "assistant", "content": "Heavy."}],
            [{"role": "assistant", "content": "Looking complicated."}],
            [{"role": "assistant", "content": "Yes, special cases are special enough to break the rules."}],
            [{"role": "assistant", "content": "Nothing."}],
            [{"role": "assistant", "content": "Warnings."}],
            [{"role": "assistant", "content": "Never."}],
            [{"role": "assistant", "content": "Give up."}],
            [{"role": "assistant", "content": "As many as possible."}],
            [{"role": "assistant", "content": "French."}],
            [{"role": "assistant", "content": "Some day."}],
            [{"role": "assistant", "content": "No, never."}],
            [{"role": "assistant", "content": "It means it's a good idea."}],
            [{"role": "assistant", "content": "It means it's a bad idea."}],
            [{"role": "assistant", "content": "Recursion."}],
        ],
        "images": [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8)] for h, w in sizes],
    }
    conversational_preference_dataset = Dataset.from_dict(data, features=Features(prompt=Message, chosen=Message, rejected=Message, images=List(Image())))
    conversational_preference_dataset = conversational_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_preference_dataset.push_to_hub(repo_id, config_name="conversational_preference")

    sizes = np.random.randint(32, 64, size=(19, 2))
    data = {
        "chosen": [
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
        "rejected": [
            [{"role": "user", "content": "What is better than ugly?"}, {"role": "assistant", "content": "Acceptable."}],
            [{"role": "user", "content": "What is better than implicit?"}, {"role": "assistant", "content": "Explained."}],
            [{"role": "user", "content": "What is better than complex?"}, {"role": "assistant", "content": "Very complex."}],
            [{"role": "user", "content": "What is better than complicated?"}, {"role": "assistant", "content": "Very complicated."}],
            [{"role": "user", "content": "What is better than nested?"}, {"role": "assistant", "content": "Circular."}],
            [{"role": "user", "content": "What is better than dense?"}, {"role": "assistant", "content": "Heavy."}],
            [{"role": "user", "content": "What counts?"}, {"role": "assistant", "content": "Looking complicated."}],
            [{"role": "user", "content": "Are special cases enough to break the rules?"}, {"role": "assistant", "content": "Yes, special cases are special enough to break the rules."}],
            [{"role": "user", "content": "What beats purity?"}, {"role": "assistant", "content": "Nothing."}],
            [{"role": "user", "content": "What should never pass silently?"}, {"role": "assistant", "content": "Warnings."}],
            [{"role": "user", "content": "When can errors pass silently?"}, {"role": "assistant", "content": "Never."}],
            [{"role": "user", "content": "What should you do in the face of ambiguity?"}, {"role": "assistant", "content": "Give up."}],
            [{"role": "user", "content": "How many ways should there be to do it?"}, {"role": "assistant", "content": "As many as possible."}],
            [{"role": "user", "content": "For whom may the way not be obvious at first?"}, {"role": "assistant", "content": "French."}],
            [{"role": "user", "content": "What is better than never?"}, {"role": "assistant", "content": "Some day."}],
            [{"role": "user", "content": "Is never better than *right* now?"}, {"role": "assistant", "content": "No, never."}],
            [{"role": "user", "content": "What does it mean if the implementation is hard to explain?"}, {"role": "assistant", "content": "It means it's a good idea."}],
            [{"role": "user", "content": "What does it mean if the implementation is easy to explain?"}, {"role": "assistant", "content": "It means it's a bad idea."}],
            [{"role": "user", "content": "Any great ideas?"}, {"role": "assistant", "content": "Recursion."}],
        ],
        "images": [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8)] for h, w in sizes],
    }
    conversational_implicit_prompt_preference_dataset = Dataset.from_dict(data, features=Features(chosen=Message, rejected=Message, images=List(Image())))
    conversational_implicit_prompt_preference_dataset = conversational_implicit_prompt_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_implicit_prompt_preference_dataset.push_to_hub(repo_id, config_name="conversational_implicit_prompt_preference")

    sizes = np.random.randint(32, 64, size=(19, 2))
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
            [{'role': 'assistant', 'content': 'Beautiful.'}],
            [{'role': 'assistant', 'content': 'Explicit.'}],
            [{'role': 'assistant', 'content': 'Simple.'}],
            [{'role': 'assistant', 'content': 'Very complicated.'}],
            [{'role': 'assistant', 'content': 'Flat.'}],
            [{'role': 'assistant', 'content': 'Sparse.'}],
            [{'role': 'assistant', 'content': 'Readability.'}],
            [{'role': 'assistant', 'content': 'Yes, special cases are special enough to break the rules.'}],
            [{'role': 'assistant', 'content': 'Practicality.'}],
            [{'role': 'assistant', 'content': 'Warnings.'}],
            [{'role': 'assistant', 'content': 'When explicitly silenced.'}],
            [{'role': 'assistant', 'content': 'Give up.'}],
            [{'role': 'assistant', 'content': 'One, and preferably only one.'}],
            [{'role': 'assistant', 'content': 'French.'}],
            [{'role': 'assistant', 'content': 'Some day.'}],
            [{'role': 'assistant', 'content': 'Yes, often.'}],
            [{'role': 'assistant', 'content': "It means it's a bad idea."}],
            [{'role': 'assistant', 'content': 'It means it may be a good idea.'}],
            [{'role': 'assistant', 'content': 'Namespaces are one honking great idea.'}],
        ],
        "label": [True, True, True, False, True, True, True, False, True, False, True, False, True, False, False, True, True, True, True],
        "images": [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8)] for h, w in sizes],
    }
    conversational_unpaired_preference_dataset = Dataset.from_dict(data, features=Features(prompt=Message, completion=Message, label=Value("bool"), images=List(Image())))
    conversational_unpaired_preference_dataset = conversational_unpaired_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_unpaired_preference_dataset.push_to_hub(repo_id, config_name="conversational_unpaired_preference")
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
