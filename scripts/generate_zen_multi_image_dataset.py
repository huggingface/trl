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


Message = List({"content": List({"text": Value("string"), "type": Value("string")}), "role": Value("string")})


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        test_size (`float`, *optional*, defaults to `0.1`):
            Fraction of the dataset to include in the test split.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-internal-testing/zen-multi-image"`):
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
        default="trl-internal-testing/zen-multi-image",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )


def main(test_size, push_to_hub, repo_id):
    # fmt: off
    messages = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than ugly?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Beautiful."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than implicit?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Explicit."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":  "What is better than complex?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Simple."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What is better than complicated?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Complex."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than nested?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Flat."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than dense?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Sparse."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What counts?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Readability."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Are special cases enough to break the rules?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "No, special cases aren't special enough to break the rules."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What beats purity?"}, {"type": "image"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Practicality."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What should never pass silently?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Errors."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "When can errors pass silently?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "When explicitly silenced."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What should you do in the face of ambiguity?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Refuse the temptation to guess."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many ways should there be to do it?"}, {"type": "image"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "One, and preferably only one."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "For whom may the way not be obvious at first?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Dutch."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than never?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Now is better than never."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Is"}, {"type": "image"}, {"type": "text", "text": " never better than *right* now?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Yes, often."}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does it mean if the implementation is hard to explain?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "It means it's a bad idea."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What does it mean if the implementation is easy to explain?"}, {"type": "image"}]}, {"role": "assistant", "content": [{"type": "text", "text": "It means it may be a good idea."}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Any great ideas?"}]}, {"role": "assistant", "content": [{"type": "text", "text": "Namespaces are one honking great idea."}]}],
    ]
    # Create the images
    number_of_images = [sum(1 for part in row[0]["content"] if part.get("type") == "image") for row in messages]
    sizes = [np.random.randint(32, 64, size=(num_images, 2)) for num_images in number_of_images]
    images = [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8) for h, w in s] for s in sizes]
    conversational_language_modeling_dataset = Dataset.from_dict({"messages": messages, "images": images}, features=Features(messages=Message, images=List(Image())))
    conversational_language_modeling_dataset = conversational_language_modeling_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_language_modeling_dataset.push_to_hub(repo_id, config_name="conversational_language_modeling")

    prompt = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than ugly?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than implicit?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":  "What is better than complex?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What is better than complicated?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than nested?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than dense?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What counts?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Are special cases enough to break the rules?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What beats purity?"}, {"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What should never pass silently?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "When can errors pass silently?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What should you do in the face of ambiguity?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many ways should there be to do it?"}, {"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "For whom may the way not be obvious at first?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than never?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Is"}, {"type": "image"}, {"type": "text", "text": " never better than *right* now?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does it mean if the implementation is hard to explain?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What does it mean if the implementation is easy to explain?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Any great ideas?"}]}],
    ]
    # Create the images
    number_of_images = [sum(1 for part in row[0]["content"] if part.get("type") == "image") for row in prompt]
    sizes = [np.random.randint(32, 64, size=(num_images, 2)) for num_images in number_of_images]
    images = [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8) for h, w in s] for s in sizes]
    conversational_prompt_only_dataset = Dataset.from_dict({"prompt": prompt, "images": images}, features=Features(prompt=Message, images=List(Image())))
    conversational_prompt_only_dataset = conversational_prompt_only_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_only_dataset.push_to_hub(repo_id, config_name="conversational_prompt_only")

    prompt = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than ugly?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than implicit?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":  "What is better than complex?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What is better than complicated?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than nested?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than dense?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What counts?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Are special cases enough to break the rules?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What beats purity?"}, {"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What should never pass silently?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "When can errors pass silently?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What should you do in the face of ambiguity?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many ways should there be to do it?"}, {"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "For whom may the way not be obvious at first?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than never?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Is"}, {"type": "image"}, {"type": "text", "text": " never better than *right* now?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does it mean if the implementation is hard to explain?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What does it mean if the implementation is easy to explain?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Any great ideas?"}]}],
    ]
    completion = [
        [{"role": "assistant", "content": [{"type": "text", "text": "Beautiful."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Explicit."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Simple."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Complex."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Flat."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Sparse."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Readability."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "No, special cases aren't special enough to break the rules."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Practicality."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Errors."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "When explicitly silenced."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Refuse the temptation to guess."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "One, and preferably only one."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Dutch."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Now is better than never."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Yes, often."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "It means it's a bad idea."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "It means it may be a good idea."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Namespaces are one honking great idea."}]}],
    ]
    # Create the images
    number_of_images = [sum(1 for part in row[0]["content"] if part.get("type") == "image") for row in prompt]
    sizes = [np.random.randint(32, 64, size=(num_images, 2)) for num_images in number_of_images]
    images = [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8) for h, w in s] for s in sizes]
    conversational_prompt_completion_dataset = Dataset.from_dict({"prompt": prompt, "completion": completion, "images": images}, features=Features(prompt=Message, completion=Message, images=List(Image())))
    conversational_prompt_completion_dataset = conversational_prompt_completion_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_completion_dataset.push_to_hub(repo_id, config_name="conversational_prompt_completion")

    prompt = [
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than ugly?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than implicit?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":  "What is better than complex?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What is better than complicated?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is better than nested?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than dense?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What counts?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Are special cases enough to break the rules?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What beats purity?"}, {"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "What should never pass silently?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "When can errors pass silently?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What should you do in the face of ambiguity?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "How many ways should there be to do it?"}, {"type": "image"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "For whom may the way not be obvious at first?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What is better than never?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Is"}, {"type": "image"}, {"type": "text", "text": " never better than *right* now?"}]}],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What does it mean if the implementation is hard to explain?"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "What does it mean if the implementation is easy to explain?"}, {"type": "image"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "Any great ideas?"}]}],
    ]
    chosen = [
        [{"role": "assistant", "content": [{"type": "text", "text": "Beautiful."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Explicit."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Simple."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Complex."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Flat."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Sparse."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Readability."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "No, special cases aren't special enough to break the rules."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Practicality."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Errors."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "When explicitly silenced."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Refuse the temptation to guess."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "One, and preferably only one."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Dutch."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Now is better than never."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Yes, often."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "It means it's a bad idea."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "It means it may be a good idea."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Namespaces are one honking great idea."}]}],
    ]
    rejected = [
        [{"role": "assistant", "content": [{"type": "text", "text": "Acceptable."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Explained."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Very complex."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Very complicated."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Circular."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Heavy."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Looking complicated."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Yes, special cases are special enough to break the rules."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Nothing."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Warnings."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Never."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Give up."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "As many as possible."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "French."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Some day."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "No, never."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "It means it's a good idea."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "It means it's a bad idea."}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": "Recursion."}]}],
    ],
    # Create the images
    number_of_images = [sum(1 for part in row[0]["content"] if part.get("type") == "image") for row in prompt]
    sizes = [np.random.randint(32, 64, size=(num_images, 2)) for num_images in number_of_images]
    images = [[np.random.uniform(low=0.0, high=255.0, size=(h, w, 3)).astype(np.uint8) for h, w in s] for s in sizes]
    conversational_preference_dataset = Dataset.from_dict({"prompt": prompt, "chosen": chosen, "rejected": rejected, "images": images}, features=Features(prompt=Message, chosen=Message, rejected=Message, images=List(Image())))
    conversational_preference_dataset = conversational_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_preference_dataset.push_to_hub(repo_id, config_name="conversational_preference")
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
