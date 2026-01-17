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

from datasets import Dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        test_size (`float`, *optional*, defaults to `0.1`):
            Fraction of the dataset to include in the test split.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the dataset to the Hugging Face Hub.
        repo_id (`str`, *optional*, defaults to `"trl-internal-testing/harmony"`):
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
        default="trl-internal-testing/harmony",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )


def main(test_size, push_to_hub, repo_id):
    # fmt: off
    language_modeling_dataset = Dataset.from_dict({
        "messages": [
            [{"role": "user", "content": "What is better than ugly?"}, {"role": "assistant", "thinking": "Beauty improves clarity and appeal.", "content": "Beautiful."}],
            [{"role": "user", "content": "What is better than implicit?"}, {"role": "assistant", "thinking": "Clarity avoids confusion.", "content": "Explicit."}],
            [{"role": "user", "content": "What is better than complex?"}, {"role": "assistant", "thinking": "Simplicity is easier to manage.", "content": "Simple."}],
            [{"role": "user", "content": "What is better than complicated?"}, {"role": "assistant", "thinking": "Complexity has structure; complication doesn’t.", "content": "Complex."}],
            [{"role": "user", "content": "What is better than nested?"}, {"role": "assistant", "thinking": "Flat structures are easier to read.", "content": "Flat."}],
            [{"role": "user", "content": "What is better than dense?"}, {"role": "assistant", "thinking": "Spacing aids understanding.", "content": "Sparse."}],
            [{"role": "user", "content": "What counts?"}, {"role": "assistant", "thinking": "Readable code lasts longer.", "content": "Readability."}],
            [{"role": "user", "content": "Are special cases enough to break the rules?"}, {"role": "assistant", "thinking": "Consistency is more valuable.", "content": "No, special cases aren't special enough to break the rules."}],
            [{"role": "user", "content": "What beats purity?"}, {"role": "assistant", "thinking": "Real-world needs outweigh ideals.", "content": "Practicality."}],
            [{"role": "user", "content": "What should never pass silently?"}, {"role": "assistant", "thinking": "Silent errors cause hidden issues.", "content": "Errors."}],
            [{"role": "user", "content": "When can errors pass silently?"}, {"role": "assistant", "thinking": "Silence must be intentional.", "content": "When explicitly silenced."}],
            [{"role": "user", "content": "What should you do in the face of ambiguity?"}, {"role": "assistant", "thinking": "Guessing leads to wrong choices.", "content": "Refuse the temptation to guess."}],
            [{"role": "user", "content": "How many ways should there be to do it?"}, {"role": "assistant", "thinking": "One way reduces confusion.", "content": "One, and preferably only one."}],
            [{"role": "user", "content": "For whom may the way not be obvious at first?"}, {"role": "assistant", "thinking": "A playful nod to Python’s creator.", "content": "Dutch."}],
            [{"role": "user", "content": "What is better than never?"}, {"role": "assistant", "thinking": "Action is better than endless delay.", "content": "Now is better than never."}],
            [{"role": "user", "content": "Is never better than *right* now?"}, {"role": "assistant", "thinking": "Rushed action can be worse.", "content": "Yes, often."}],
            [{"role": "user", "content": "What does it mean if the implementation is hard to explain?"}, {"role": "assistant", "thinking": "Good ideas should be clear.", "content": "It means it's a bad idea."}],
            [{"role": "user", "content": "What does it mean if the implementation is easy to explain?"}, {"role": "assistant", "thinking": "Clarity suggests soundness.", "content": "It means it may be a good idea."}],
            [{"role": "user", "content": "Any great ideas?"}, {"role": "assistant", "thinking": "Namespaces prevent conflicts.", "content": "Namespaces are one honking great idea."}]
        ],
        "chat_template_kwargs": [
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
        ]
    })
    language_modeling_dataset = language_modeling_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        language_modeling_dataset.push_to_hub(repo_id, config_name="language_modeling")
    language_modeling_dataset.save_to_disk(repo_id + "/language_modeling")
    prompt_completion_dataset = Dataset.from_dict({
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
            [{"role": "assistant", "thinking": "Beauty improves clarity and appeal.", "content": "Beautiful."}],
            [{"role": "assistant", "thinking": "Clarity avoids confusion.", "content": "Explicit."}],
            [{"role": "assistant", "thinking": "Simplicity is easier to manage.", "content": "Simple."}],
            [{"role": "assistant", "thinking": "Complexity has structure; complication doesn’t.", "content": "Complex."}],
            [{"role": "assistant", "thinking": "Flat structures are easier to read.", "content": "Flat."}],
            [{"role": "assistant", "thinking": "Spacing aids understanding.", "content": "Sparse."}],
            [{"role": "assistant", "thinking": "Readable code lasts longer.", "content": "Readability."}],
            [{"role": "assistant", "thinking": "Consistency is more valuable.", "content": "No, special cases aren't special enough to break the rules."}],
            [{"role": "assistant", "thinking": "Real-world needs outweigh ideals.", "content": "Practicality."}],
            [{"role": "assistant", "thinking": "Silent errors cause hidden issues.", "content": "Errors."}],
            [{"role": "assistant", "thinking": "Silence must be intentional.", "content": "When explicitly silenced."}],
            [{"role": "assistant", "thinking": "Guessing leads to wrong choices.", "content": "Refuse the temptation to guess."}],
            [{"role": "assistant", "thinking": "One way reduces confusion.", "content": "One, and preferably only one."}],
            [{"role": "assistant", "thinking": "A playful nod to Python’s creator.", "content": "Dutch."}],
            [{"role": "assistant", "thinking": "Action is better than endless delay.", "content": "Now is better than never."}],
            [{"role": "assistant", "thinking": "Rushed action can be worse.", "content": "Yes, often."}],
            [{"role": "assistant", "thinking": "Good ideas should be clear.", "content": "It means it's a bad idea."}],
            [{"role": "assistant", "thinking": "Clarity suggests soundness.", "content": "It means it may be a good idea."}],
            [{"role": "assistant", "thinking": "Namespaces prevent conflicts.", "content": "Namespaces are one honking great idea."}],
        ],
        "chat_template_kwargs": [
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "medium", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "high", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
            {"reasoning_effort": "low", "model_identity": "You are Tiny ChatGPT, a tiny language model."},
        ]
    })
    prompt_completion_dataset = prompt_completion_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        prompt_completion_dataset.push_to_hub(repo_id, config_name="prompt_completion")
    prompt_completion_dataset.save_to_disk(repo_id + "/prompt_completion")
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
