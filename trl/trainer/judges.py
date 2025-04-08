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

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from accelerate import Accelerator
from huggingface_hub import InferenceClient
from transformers.utils import is_openai_available

from ..import_utils import is_llm_blender_available


if is_llm_blender_available():
    import llm_blender

if is_openai_available():
    from openai import OpenAI


DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


class BaseJudge(ABC):
    """
    Base class for judges. The subclasses of this class should implement the `judge` method.
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[str], shuffle_order: bool = True) -> list:
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseRankJudge(ABC):
    """
    Base class for LLM ranking judges.

    **Example**:
    ```python
    class MyRankJudge(BaseRankJudge):
        def judge(self, prompts, completions, shuffle_order=True):
            return ...  # Your ranking logic here

    judge = MyRankJudge()
    judge.judge(
        prompts=["The capital of France is", "The capital of Germany is"],
        completions=[[" Paris", " Marseille", "Lyon"], [" Munich", " Berlin"]]
    )  # [[0, 1, 2], [1, 0]]
    ```
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[list[int]]:
        """
        Judge the completion for the given prompts and return the ranks of each completion.

        Args:
            prompts (`list[str]`):
                List of prompts.
            completions (`list[list[str]]`):
                List of completions list, where each element is a list of completions for the corresponding prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[list[int]]`:
                List of lists of idxs, where each list contains the ranks of the completions for the corresponding
                prompt. E.g., `[1, 2, 0]` means that the second completion (`idx=1`) is the best, followed by the
                third, and then the first.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BasePairwiseJudge(BaseJudge):
    """
    Base class for pairwise judges.
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        """
        Judge the completion pairs for the given prompts.

        Args:
            prompts (`list[str]`):
                List of prompts.
            completions (`list[list[str]]`):
                List of completions pairs, where each element is a pair of completions for the corresponding prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[int]`:
                List of idxs, where each idx is the rank of the best completion for the corresponding prompt.
                E.g., `1` means that the second completion (`idx=1`) is the best.

        Note:
            If the judge returns `-1` for any prompt, it indicates that the inner process used to compute the
            preference has failed. For instance, this could occur if the underlying language model returned an invalid
            answer. In such cases, the caller should handle these invalid indices appropriately, possibly by
            implementing fallback logic or error handling.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseBinaryJudge(BaseJudge):
    """
    Base class for binary judges.
    """

    @abstractmethod
    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        """
        Judge the completion for a given prompt. Used to assess if a completion satisfies a constraint.

        This base class should be used to implement binary evaluations as done in section 4.1.4 of the
        [CGPO paper](https://huggingface.co/papers/2409.20370).
        It is relevant for assessing whether a prompt completion pair satisfies a specific contraint.

        Args:
            prompts (`list[str]`): List of prompts.
            completions (`list[str]`): List of completions.
            gold_completions (`list[str]`, `optional`): List of gold completions if it exists.
            shuffle_order (`bool`): Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            list[int]: A list of binary labels:
                - 1 indicates that the completion satisfies the evaluated constraint.
                - 0 indicates that the completion does not satisfy the evaluated constraint.

        Note:
            If the judge returns -1 for any prompt, it indicates that the inner process used to compute the preference has failed.
            For instance, this could occur if the underlying language model or rule based contraint returned an invalid answer.
            In such cases, the caller should handle these invalid indices appropriately, possibly by implementing fallback logic or error handling.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class PairRMJudge(BasePairwiseJudge):
    """
    LLM judge based on the PairRM model from AllenAI.

    This judge uses the PairRM model to rank pairs of completions for given prompts. It's designed for pairwise
    comparison of language model outputs. The PairRM model is loaded using the llm-blender library and runs on the
    default Accelerator device.

    **Attributes**:

        blender (`llm_blender.Blender`):
            An instance of the Blender class from llm-blender.

    **Example**:
    ```python
    >>> pairrm_judge = PairRMJudge()
    >>> prompts = ["Translate 'hello' to French", "What's the capital of Japan?"]
    >>> completions = [["Bonjour", "Salut"], ["Kyoto", "Tokyo"]]
    >>> results = pairrm_judge.judge(prompts, completions)
    >>> print(results)  # [0, 1] (indicating the first completion is preferred for the first prompt and the second)
    ```

    <Tip>

    This class requires the llm-blender library to be installed. Install it with: `pip install llm-blender`.

    </Tip>
    """

    def __init__(self):
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Please install it with `pip install llm-blender`.")
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ) -> list[Union[int, float]]:
        """
        Judge the completion pairs for the given prompts using the PairRM model.

        Args:
            prompts (`list[str]`):
                List of prompts to judge.
            completions (`list[list[str]]`):
                List of completion pairs for each prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.
            return_scores (`bool`, *optional*, defaults to `False`):
                If `True`, return probability scores of the first completion instead of ranks (i.e. a *soft-judge*).
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature for scaling logits if `return_scores` is True.

        Returns:
            `Union[list[int, float]]`:
                If `return_scores` is `False`, returns a list of ranks (`0` or `1`) for each prompt, indicating which
                completion is preferred.
                If `return_scores` is `True`, returns softmax probabilities for the first completion.

        Raises:
            `ValueError`:
                If the number of completions per prompt is not exactly 2.

        Note:
            Unlike llm-blender, ranks are 0-indexed (`0` means the first completion is preferred).
        """

        if len(completions[0]) != 2:
            raise ValueError("PairRM judge requires exactly 2 completions per prompt.")

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Rank the completions
        ranks = self.blender.rank(prompts, completions, return_scores=return_scores, disable_tqdm=True)
        if not return_scores:
            ranks -= 1  # PairRM rank is 1-indexed, so we subtract 1 to make it 0-indexed
        else:
            # scale the logits by temperature
            ranks /= temperature

        # Flip back the ranks or scores to the original order if needed
        if shuffle_order:
            ranks[flip_mask] = ranks[flip_mask][:, ::-1]

        # Return the ranks or score probability
        if return_scores:
            logit_max = np.amax(ranks, axis=-1, keepdims=True)
            exp_logit_shifted = np.exp(ranks - logit_max)
            probs = exp_logit_shifted / np.sum(exp_logit_shifted, axis=-1, keepdims=True)
            return probs[:, 0].tolist()
        else:
            return ranks[:, 0].tolist()


class HfPairwiseJudge(BasePairwiseJudge):
    """
    Pairwise judge based on the Hugging Face API with chat completion.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3-70B-Instruct"`):
            Model to use for the judge.
        token (`str`, *optional*):
            Hugging Face API token to use for the [`huggingface_hub.InferenceClient`].
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            The system prompt to be used for the judge. If not provided, a default prompt is used. Note that the system
            prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`. Also, the
            inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token
            response.
    """

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        token: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = InferenceClient(model=model, token=token)
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            completion = self.client.chat_completion(messages=[{"role": "user", "content": content}], max_tokens=1)
            response = completion.choices[0].message.content
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.debug(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        # Return the ranks
        return ranks


class OpenAIPairwiseJudge(BasePairwiseJudge):
    """
    Judge based on the OpenAI API.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*, defaults to `"gpt-4-turbo-preview"`):
            Model to use for the judge.
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            System prompt to be used for the judge. If not provided, a default prompt is used. Note that the system
            prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`. Also, the
            inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token
            response.
        max_requests (`int` or `None`, *optional*, defaults to `1000`):
            Maximum number of requests to make to the OpenAI API. If set to `None`, there is no limit.
    """

    def __init__(
        self, model="gpt-4-turbo-preview", system_prompt: Optional[str] = None, max_requests: Union[int, None] = 1_000
    ):
        if not is_openai_available():
            raise ValueError("OpenAI client is not installed. Please install it with 'pip install openai'.")
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT
        self.max_requests = max_requests
        self.num_requests = 0
        self._warned = False

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        # Check if the limit of requests is reached, if so, use random choice instead
        if self.max_requests is not None and self.num_requests >= self.max_requests:
            if not self._warned:  # Print the warning only once
                logging.warning(
                    f"Reached the maximum number of requests ({self.max_requests}). From now on, returning -1 instead. "
                    " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
                )
                self._warned = True
            return [-1] * len(prompts)

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            messages = [{"role": "user", "content": content}]
            completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=1)
            response = completion.choices[0].message.content
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.debug(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        # Update the number of requests
        self.num_requests += len(prompts)

        # Return the ranks
        return ranks


class AllTrueJudge(BaseBinaryJudge):
    """
    Unify the decision of multiple [`BaseBinaryJudge`] instances.

    Returns `1` only if all inner binary judges return `1`. If any judge returns `0`, it returns `0`.
    If any judge returns `-1`, indicating a failure in its process, this judge will also return `-1`.

    Implements the Mixture of Judges as described in the [CGPO paper](https://huggingface.co/papers/2409.20370).

    Args:
    judges (`list[BaseBinaryJudge]`): A list of [`BaseBinaryJudge`] instances whose decisions will be unified.
    """

    def __init__(self, judges: list[BaseBinaryJudge]):
        self.judges = judges

    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        all_binary_judgments = [
            judge.judge(prompts, completions, gold_completions, shuffle_order) for judge in self.judges
        ]
        output = []
        for binary_judgments in zip(*all_binary_judgments):
            # Check that all values are in {0, 1, -1}
            if any(binary_judgment not in {0, 1, -1} for binary_judgment in binary_judgments):
                raise ValueError(
                    f"Invalid binary judgment: {binary_judgments}, expected list of values in {{0, 1, -1}}."
                )

            # Unify the decision
            if -1 in binary_judgments:
                output.append(-1)
            elif all(binary_judgment == 1 for binary_judgment in binary_judgments):
                output.append(1)
            else:
                output.append(0)
        return output
