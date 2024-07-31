import concurrent.futures
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from accelerate import Accelerator
from huggingface_hub import InferenceClient

from ..import_utils import is_llmblender_available, is_openai_available


if is_llmblender_available():
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
    def judge(self, prompts: List[str], completions: List[str], shuffle_order: bool = True) -> List:
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseRankJudge(ABC):
    """
    Base class for LLM ranking judges.

    Example:
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
    def judge(self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True) -> List[List[int]]:
        """
        Judge the completion for the given prompts and return the ranks of each completion.

        Args:
            prompts (`List[str]`): List of prompts.
            completions (`List[List[str]]`): List of completions list, where each element is a list of completions for the corresponding prompt.
            shuffle_order (`bool`): Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            List of lists of idxs, where each list contains the ranks of the completions for the corresponding prompt.
            E.g., [1, 2, 0] means that the second completion (idx=1) is the best, followed by the third, and then the first.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BasePairwiseJudge(BaseJudge):
    """
    Base class for pairwise judges.
    """

    @abstractmethod
    def judge(self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True) -> List[int]:
        """
        Judge the completion pairs for the given prompts.

        Args:
            prompts (`List[str]`): List of prompts.
            completions (`List[List[str]]`): List of completions pairs, where each element is a pair of completions for the corresponding prompt.
            shuffle_order (`bool`): Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            List of idxs, where each idx is the rank of the best completion for the corresponding prompt.
            E.g., 1 means that the second completion (idx=1) is the best.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class RandomRankJudge(BaseRankJudge):
    """
    Random rank, for testing purposes.
    """

    def judge(self, prompts, completions, shuffle_order=True):
        num_completions = [len(completions[i]) for i in range(len(prompts))]
        return [random.sample(range(n), n) for n in num_completions]


class RandomPairwiseJudge(BasePairwiseJudge):
    """
    Random pairwise judge, for testing purposes.
    """

    def judge(self, prompts, completions, shuffle_order=True):
        return [random.randint(0, len(completion) - 1) for completion in completions]


class PairRMJudge(BasePairwiseJudge):
    """
    LLM judge based on the PairRM model from AllenAI.

    See: https://huggingface.co/llm-blender/PairRM
    """

    def __init__(self):
        if not is_llmblender_available():
            raise ValueError("llm-blender is not installed. Please install it with 'pip install llm-blender'.")
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)

    def judge(self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True) -> List[int]:
        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Rank the completions
        ranks = self.blender.rank(prompts, completions)
        ranks -= 1  # PairRM is 1-indexed, so we subtract 1 to make it 0-indexed

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks[flip_mask] = ranks[flip_mask][:, ::-1]

        # Return the ranks
        return ranks[:, 0].tolist()


class HfPairwiseJudge(BasePairwiseJudge):
    """
    Pairwise judge based on the Hugging Face API with chat completion.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*): The model to use for the judge. Defaults to "meta-llama/Meta-Llama-3-70B-Instruct".
        token (`str`, *optional*): The Hugging Face API token to use for the InferenceClient.
        system_prompt (`str`, *optional*): The system prompt to be used for the judge. If not provided, a default prompt is used.
            Note that the system prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`.
            Also, the inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token response.
    """

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        token: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = InferenceClient(model=model, token=token)
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def judge(self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True) -> List[int]:
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
                logging.warning(f"Invalid response from the model: {response}, using random choice instead.")
                return random.choice([0, 1])

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
        model (`str`, *optional*): The model to use for the judge. Defaults to `"gpt-4-turbo-preview"`.
        system_prompt (`str`, *optional*): The system prompt to be used for the judge. If not provided, a default prompt is used.
            Note that the system prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`.
            Also, the inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token response.
        max_requests (`int`, *optional*): The maximum number of requests to make to the OpenAI API. Defaults to 1000. If set to `None`, there is no limit.

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

    def judge(self, prompts: List[str], completions: List[List[str]], shuffle_order: bool = True) -> List[int]:
        # Check if the limit of requests is reached, if so, use random choice instead
        if self.max_requests is not None and self.num_requests >= self.max_requests:
            if not self._warned:  # Print the warning only once
                logging.warning(
                    f"Reached the maximum number of requests ({self.max_requests}). From now on, using random choice instead. "
                    " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
                )
                self._warned = True
            return [random.choice([0, 1]) for _ in prompts]

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
                logging.warning(f"Invalid response from the model: {response}, using random choice instead.")
                return random.choice([0, 1])

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
