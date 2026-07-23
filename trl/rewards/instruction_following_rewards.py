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

# The constraint checkers below are adapted from the IFEval instruction registry
# (https://github.com/google-research/google-research/tree/master/instruction_following_eval, Apache-2.0).
# We vendor a curated subset that is verifiable with the Python standard library only (no nltk/langdetect).

import collections
import json
import re


_CONSTRAINED_RESPONSE_OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


def _keyword_existence(response, keywords, **_):
    return all(re.search(keyword, response, flags=re.IGNORECASE) for keyword in keywords)


def _keyword_frequency(response, keyword, frequency, relation="at least", **_):
    count = len(re.findall(keyword, response, flags=re.IGNORECASE))
    return count < frequency if relation == "less than" else count >= frequency


def _forbidden_words(response, forbidden_words, **_):
    return not any(re.search(r"\b" + word + r"\b", response, flags=re.IGNORECASE) for word in forbidden_words)


def _letter_frequency(response, letter, let_frequency, let_relation="at least", **_):
    count = collections.Counter(response.lower())[letter.lower()]
    return count < let_frequency if let_relation == "less than" else count >= let_frequency


def _number_of_words(response, num_words, relation="at least", **_):
    count = len(re.findall(r"\w+", response))
    return count < num_words if relation == "less than" else count >= num_words


def _number_of_paragraphs(response, num_paragraphs, **_):
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                count -= 1
            else:
                return False
    return count == num_paragraphs


def _number_of_placeholders(response, num_placeholders, **_):
    return len(re.findall(r"\[.*?\]", response)) >= num_placeholders


def _postscript(response, postscript_marker, **_):
    response = response.lower()
    if postscript_marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif postscript_marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + postscript_marker.lower() + r".*$"
    return bool(re.findall(pattern, response, flags=re.MULTILINE))


def _number_of_bullets(response, num_bullets, **_):
    bullets = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(bullets) == num_bullets


def _number_of_highlights(response, num_highlights, **_):
    count = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", response):
        if highlight.strip("*").strip():
            count += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if highlight.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def _multiple_sections(response, section_spliter, num_sections, **_):
    pattern = r"\s?" + section_spliter + r"\s?\d+\s?"
    return len(re.split(pattern, response)) - 1 >= num_sections


def _title(response, **_):
    return any(title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", response))


def _json_format(response, **_):
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _constrained_response(response, **_):
    response = response.strip()
    return any(option in response for option in _CONSTRAINED_RESPONSE_OPTIONS)


def _end_checker(response, end_phrase, **_):
    return response.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def _quotation(response, **_):
    response = response.strip()
    return len(response) > 1 and response[0] == '"' and response[-1] == '"'


def _no_comma(response, **_):
    return not re.search(r"\,", response)


# Maps IFEval instruction ids to their (stdlib-only) checker. Each checker takes the response text plus the
# instruction's keyword arguments and returns whether the constraint is satisfied.
INSTRUCTION_CHECKERS = {
    "keywords:existence": _keyword_existence,
    "keywords:frequency": _keyword_frequency,
    "keywords:forbidden_words": _forbidden_words,
    "keywords:letter_frequency": _letter_frequency,
    "length_constraints:number_words": _number_of_words,
    "length_constraints:number_paragraphs": _number_of_paragraphs,
    "detectable_content:number_placeholders": _number_of_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _number_of_bullets,
    "detectable_format:number_highlighted_sections": _number_of_highlights,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:title": _title,
    "detectable_format:json_format": _json_format,
    "detectable_format:constrained_response": _constrained_response,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "punctuation:no_comma": _no_comma,
}


def ifeval_reward(
    completions: list[list[dict[str, str]]],
    instruction_id_list: list[list[str]],
    instruction_kwargs: list[list[dict]],
    **kwargs,
) -> list[float]:
    r"""
    Reward function that checks how many verifiable instruction-following constraints a completion satisfies. This is
    the rule-based ("verifiable") reward used to extend RLVR beyond math. Reference: Tülu 3
    (https://huggingface.co/papers/2411.15124), using the IFEval instruction set
    (https://huggingface.co/papers/2311.07911).

    Each completion comes with a list of constraints (e.g. "include keyword X", "answer with at least N words"). Each
    constraint is identified by an IFEval instruction id and a dictionary of arguments. The reward is the fraction of
    constraints satisfied, in `[0.0, 1.0]`.

    Only a curated, standard-library-only subset of the IFEval constraints is supported (no `nltk`/`langdetect`
    dependency). The supported instruction ids are the keys of `INSTRUCTION_CHECKERS` (importable from `trl.rewards`).
    An unsupported id raises a `ValueError`, so datasets should be filtered to the supported set.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        instruction_id_list (`list[list[str]]`):
            For each completion, the list of IFEval instruction ids to check (e.g.
            `["keywords:existence", "length_constraints:number_words"]`).
        instruction_kwargs (`list[list[dict]]`):
            For each completion, the list of keyword-argument dictionaries, aligned with `instruction_id_list` (e.g.
            `[{"keywords": ["AI"]}, {"num_words": 100, "relation": "at least"}]`).
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float]`:
            A list of rewards, where each reward is the fraction of satisfied constraints (`0.0` if a completion has no
            constraints).

    Example:
    ```python
    >>> from trl.rewards import ifeval_reward

    >>> completions = [[{"content": 'I think "AI" is great. AI AI'}], [{"content": "No keyword here."}]]
    >>> instruction_id_list = [["keywords:existence", "keywords:frequency"], ["keywords:existence"]]
    >>> instruction_kwargs = [
    ...     [{"keywords": ["AI"]}, {"keyword": "AI", "frequency": 3, "relation": "at least"}],
    ...     [{"keywords": ["AI"]}],
    ... ]
    >>> ifeval_reward(completions, instruction_id_list, instruction_kwargs)
    [1.0, 0.0]
    ```
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, ids, kwargs_list in zip(contents, instruction_id_list, instruction_kwargs, strict=True):
        if not ids:
            rewards.append(0.0)
            continue
        satisfied = 0
        for instruction_id, args in zip(ids, kwargs_list, strict=True):
            if instruction_id not in INSTRUCTION_CHECKERS:
                raise ValueError(
                    f"Unsupported IFEval instruction id: '{instruction_id}'. Supported ids: "
                    f"{sorted(INSTRUCTION_CHECKERS)}"
                )
            args = {key: value for key, value in (args or {}).items() if value is not None}
            satisfied += bool(INSTRUCTION_CHECKERS[instruction_id](content, **args))
        rewards.append(satisfied / len(ids))
    return rewards
