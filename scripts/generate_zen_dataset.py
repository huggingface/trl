# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
        repo_id (`str`, *optional*, defaults to `"trl-internal-testing/zen"`):
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
        default="trl-internal-testing/zen",
        metadata={"help": "Hugging Face repository ID to push the dataset to."},
    )


def main(test_size, push_to_hub, repo_id):
    # fmt: off
    standard_language_modeling_dataset = Dataset.from_dict({
        "text": [
            "Beautiful is better than ugly.",
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Sparse is better than dense.",
            "Readability counts.",
            "Special cases aren't special enough to break the rules.",
            "Although practicality beats purity.",
            "Errors should never pass silently.",
            "Unless explicitly silenced.",
            "In the face of ambiguity, refuse the temptation to guess.",
            "There should be one-- and preferably only one --obvious way to do it.",
            "Although that way may not be obvious at first unless you're Dutch.",
            "Now is better than never.",
            "Although never is often better than *right* now.",
            "If the implementation is hard to explain, it's a bad idea.",
            "If the implementation is easy to explain, it may be a good idea.",
            "Namespaces are one honking great idea -- let's do more of those!",
        ],
    })
    standard_language_modeling_dataset = standard_language_modeling_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_language_modeling_dataset.push_to_hub(repo_id, config_name="standard_language_modeling")

    standard_prompt_only_dataset = Dataset.from_dict({
        "prompt": [
            "Beautiful is better than",
            "Explicit is",
            "Simple is better",
            "Complex",
            "Flat is better than",
            "Sparse is better",
            "Readability",
            "Special cases aren't special",
            "Although practicality beats",
            "Errors should never",
            "Unless explicitly",
            "In the face of ambiguity, refuse",
            "There should be one-- and preferably",
            "Although that way may not be obvious at first unless you're",
            "Now is",
            "Although never is often",
            "If the implementation is hard to explain,",
            "If the implementation is easy",
            "Namespaces are one honking great",
        ],
    })
    standard_prompt_only_dataset = standard_prompt_only_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_prompt_only_dataset.push_to_hub(repo_id, config_name="standard_prompt_only")

    standard_prompt_completion_dataset = Dataset.from_dict({
        "prompt": [
            "Beautiful is better than",
            "Explicit is",
            "Simple is better",
            "Complex",
            "Flat is better than",
            "Sparse is better",
            "Readability",
            "Special cases aren't special",
            "Although practicality beats",
            "Errors should never",
            "Unless explicitly",
            "In the face of ambiguity, refuse",
            "There should be one-- and preferably",
            "Although that way may not be obvious at first unless you're",
            "Now is",
            "Although never is often",
            "If the implementation is hard to explain,",
            "If the implementation is easy",
            "Namespaces are one honking great",
        ],
        "completion": [
            " ugly.",
            " better than implicit.",
            " than complex.",
            " is better than complicated.",
            " nested.",
            " than dense.",
            " counts.",
            " enough to break the rules.",
            " purity.",
            " pass silently.",
            " silenced.",
            " the temptation to guess.",
            " only one --obvious way to do it.",
            " Dutch.",
            " better than never.",
            " better than *right* now.",
            " it's a bad idea.",
            " to explain, it may be a good idea.",
            " idea -- let's do more of those!",
        ],
    })
    standard_prompt_completion_dataset = standard_prompt_completion_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_prompt_completion_dataset.push_to_hub(repo_id, config_name="standard_prompt_completion")

    standard_preference_dataset = Dataset.from_dict({
        "prompt": [
            "Beautiful is better than",
            "Explicit is",
            "Simple is better",
            "Complex",
            "Flat is better than",
            "Sparse is better",
            "Readability",
            "Special cases aren't special",
            "Although practicality beats",
            "Errors should never",
            "Unless explicitly",
            "In the face of ambiguity, refuse",
            "There should be one-- and preferably",
            "Although that way may not be obvious at first unless you're",
            "Now is",
            "Although never is often",
            "If the implementation is hard to explain,",
            "If the implementation is easy",
            "Namespaces are one honking great",
        ],
        "chosen": [
            " ugly.",
            " better than implicit.",
            " than complex.",
            " is better than complicated.",
            " nested.",
            " than dense.",
            " counts.",
            " enough to break the rules.",
            " purity.",
            " pass silently.",
            " silenced.",
            " the temptation to guess.",
            " only one --obvious way to do it.",
            " Dutch.",
            " better than never.",
            " better than *right* now.",
            " it's a bad idea.",
            " to explain, it may be a good idea.",
            " idea -- let's do more of those!",
        ],
        "rejected": [
            " the moon.",
            " worse than nothing.",
            " than a long vacation.",
            " is always the answer.",
            " chocolate.",
            " without any context.",
            " is optional.",
            " enough to become unicorns.",
            " reality.",
            " pass their driving test.",
            " forgotten.",
            " the opportunity to laugh.",
            " two or more confusing methods.",
            " a time traveler.",
            " never better.",
            " not even a possibility.",
            " it's clearly the best choice.",
            " it's probably magic.",
            " watermelon -- let's plant some!",
        ],
    })
    standard_preference_dataset = standard_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_preference_dataset.push_to_hub(repo_id, config_name="standard_preference")

    standard_implicit_prompt_preference_dataset = Dataset.from_dict({
        "chosen": [
            "Beautiful is better than ugly.",
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Sparse is better than dense.",
            "Readability counts.",
            "Special cases aren't special enough to break the rules.",
            "Although practicality beats purity.",
            "Errors should never pass silently.",
            "Unless explicitly silenced.",
            "In the face of ambiguity, refuse the temptation to guess.",
            "There should be one-- and preferably only one --obvious way to do it.",
            "Although that way may not be obvious at first unless you're Dutch.",
            "Now is better than never.",
            "Although never is often better than *right* now.",
            "If the implementation is hard to explain, it's a bad idea.",
            "If the implementation is easy to explain, it may be a good idea.",
            "Namespaces are one honking great idea -- let's do more of those!",
        ],
        "rejected": [
            "Beautiful is better than the moon.",
            "Explicit is worse than nothing.",
            "Simple is better than a long vacation.",
            "Complex is always the answer.",
            "Flat is better than chocolate.",
            "Sparse is better without any context.",
            "Readability is optional.",
            "Special cases aren't special enough to become unicorns.",
            "Although practicality beats reality.",
            "Errors should never pass their driving test.",
            "Unless explicitly forgotten.",
            "In the face of ambiguity, refuse the opportunity to laugh.",
            "There should be one-- and preferably two or more confusing methods.",
            "Although that way may not be obvious at first unless you're a time traveler.",
            "Now is never better.",
            "Although never is often not even a possibility.",
            "If the implementation is hard to explain, it's clearly the best choice.",
            "If the implementation is easy it's probably magic.",
            "Namespaces are one honking great watermelon -- let's plant some!",
        ],
    })
    standard_implicit_prompt_preference_dataset = standard_implicit_prompt_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_implicit_prompt_preference_dataset.push_to_hub(repo_id, config_name="standard_implicit_prompt_preference")

    standard_unpaired_preference_dataset = Dataset.from_dict({
        "prompt": [
            "Beautiful is better than",
            "Explicit is",
            "Simple is better",
            "Complex",
            "Flat is better than",
            "Sparse is better",
            "Readability",
            "Special cases aren't special",
            "Although practicality beats",
            "Errors should never",
            "Unless explicitly",
            "In the face of ambiguity, refuse",
            "There should be one-- and preferably",
            "Although that way may not be obvious at first unless you're",
            "Now is",
            "Although never is often",
            "If the implementation is hard to explain,",
            "If the implementation is easy",
            "Namespaces are one honking great",
        ],
        "completion": [
            " ugly.",
            " worse than nothing.",
            " than a long vacation.",
            " is better than complicated.",
            " nested.",
            " without any context.",
            " counts.",
            " enough to become unicorns.",
            " purity.",
            " pass silently.",
            " forgotten.",
            " the temptation to guess.",
            " only one --obvious way to do it.",
            " a time traveler.",
            " better than never.",
            " not even a possibility.",
            " it's a bad idea.",
            " it's probably magic.",
            " watermelon -- let's plant some!",
        ],
        "label": [True, False, False, True, True, False, True, False, True, True, False, True, True, False, True, False, True, False, False],
    })
    standard_unpaired_preference_dataset = standard_unpaired_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_unpaired_preference_dataset.push_to_hub(repo_id, config_name="standard_unpaired_preference")

    standard_stepwise_supervision_dataset = Dataset.from_dict({
        "prompt": [
            "Beautiful is better than",
            "Explicit is better than",
            "Simple is better than",
            "Complex is better than",
            "Flat is better than",
            "Sparse is better than",
            "Readability counts",
            "Special cases aren't special enough",
            "Although practicality beats",
            "Errors should never pass",
            "In the face of ambiguity, refuse",
            "There should be one-- and preferably only one --",
            "Although that way may not be",
            "Now is better than",
            "Never is often better than",
            "If the implementation is hard to explain, it's",
            "If the implementation is easy to explain, it",
            "Namespaces are one",
            "Although practicality sometimes beats purity,",
        ],
        "completions":[
            [", let me think...", " ugly."],
            [", of course,", " implicit.", " because clarity matters."],
            ["... let's keep it basic,", " complex."],
            [" when needed,", " complicated."],
            [" in terms of structure,", " nested."],
            ["... especially for readability."],
            [" especially when others read it."],
            [", unless...", " they follow the rules."],
            [" some theoretical elegance,", " purity."],
            [" silently,", " unless explicitly silenced."],
            [" the temptation to guess."],
            [" way to do it,"," but sometimes it's not obvious.", " especially when there's more than one possibility."],
            [" clear at first,", " it will eventually emerge."],
            [" later."],
            [" problematic fixes."],
            [" likely because it's too complicated."],
            [" might be a good design."],
            [" of those great ideas,", " that solve many problems."],
            [" the code should still aim for balance."],
        ],
        "labels": [
            [False, True],
            [False, True, False],
            [False, True],
            [True, True],
            [True, False],
            [True],
            [False],
            [True, False],
            [False, False],
            [False, False],
            [True],
            [True, True, False],
            [True, True],
            [False],
            [True], [False],
            [False],
            [True, True],
            [False]
        ]
    })
    standard_stepwise_supervision_dataset = standard_stepwise_supervision_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        standard_stepwise_supervision_dataset.push_to_hub(repo_id, config_name="standard_stepwise_supervision")

    conversational_language_modeling_dataset = Dataset.from_dict({
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
    })
    conversational_language_modeling_dataset = conversational_language_modeling_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_language_modeling_dataset.push_to_hub(repo_id, config_name="conversational_language_modeling")

    conversational_prompt_only_dataset = Dataset.from_dict({
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
    })
    conversational_prompt_only_dataset = conversational_prompt_only_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_only_dataset.push_to_hub(repo_id, config_name="conversational_prompt_only")

    conversational_prompt_completion_dataset = Dataset.from_dict({
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
    })
    conversational_prompt_completion_dataset = conversational_prompt_completion_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_prompt_completion_dataset.push_to_hub(repo_id, config_name="conversational_prompt_completion")

    conversational_preference_dataset = Dataset.from_dict({
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
    })
    conversational_preference_dataset = conversational_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_preference_dataset.push_to_hub(repo_id, config_name="conversational_preference")

    conversational_implicit_prompt_preference_dataset = Dataset.from_dict({
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
    })
    conversational_implicit_prompt_preference_dataset = conversational_implicit_prompt_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_implicit_prompt_preference_dataset.push_to_hub(repo_id, config_name="conversational_implicit_prompt_preference")

    conversational_unpaired_preference_dataset = Dataset.from_dict({
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
    })
    conversational_unpaired_preference_dataset = conversational_unpaired_preference_dataset.train_test_split(test_size=test_size, shuffle=False)
    if push_to_hub:
        conversational_unpaired_preference_dataset.push_to_hub(repo_id, config_name="conversational_unpaired_preference")
    # fmt: on


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    main(script_args.test_size, script_args.push_to_hub, script_args.repo_id)
