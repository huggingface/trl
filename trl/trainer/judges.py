import logging
import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from accelerate import Accelerator
from huggingface_hub import InferenceClient
from requests import HTTPError

from ..import_utils import is_llmblender_available, is_openai_available


if is_llmblender_available():
    import llm_blender

if is_openai_available():
    from openai import BadRequestError, OpenAI


DEFAULT_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response1}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response2}"""
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): 0 or 1.

## Best Model Identifier'''


class BaseJudge(ABC):
    """
    Base class for LLM judges.

    Example:
    ```python
    class MockJudge(BaseJudge):
        def judge(self, prompts, completion_pairs, shuffle_order=True):
            return [random.choice([0, 1]) for _ in range(len(prompts))]

    judge = MockJudge()
    judge.judge(
        prompts=["What is the capital of France?", "What is the capital of Germany?"],
        completion_pairs=[["Paris", "Marseille"], ["Munich", "Berlin"]]
    )  # [0, 0]
    ```
    """

    @abstractmethod
    def judge(self, prompts: List[str], completion_pairs: List[List[str]], shuffle_order: bool = True) -> List[int]:
        """
        Judge the completion pairs for the given prompts.

        Args:
            prompts (`List[str]`): List of prompts.
            completion_pairs (`List[List[str]]`): List of completion pairs, where each pair is a list of two strings.
            shuffle_order (`bool`): Whether to shuffle the order of the completion pairs, to avoid positional bias.

        Returns:
            List of integers, where each integer is the index of the completion pair that is preferred.
        """
        raise NotImplementedError("Judge subclasses must implement this method.")


class BaseAPIJudge(BaseJudge):
    """
    Base class for LLM judges reached via an API.

    The subclasses of this class should implement the `get_response` method to interact with the API.

    Args:
        system_prompt (`str`, *optional*): The system prompt to be used for the judge. If not provided, a default prompt is used.
        max_tries (`int`, *optional*): The maximum number of retries for a request. Defaults to 5.
        max_workers (`int`, *optional*): The maximum number of parallel requests. Defaults to 8.

    Example:
    ```python
    class MockAPIJudge(BaseAPIJudge):
        def get_response(self, content):
            return random.choice(["0", "1"])

    judge = MockAPIJudge()
    judge.judge(
        prompts=["What is the capital of France?", "What is the capital of Germany?"],
        completion_pairs=[["Paris", "Marseille"], ["Munich", "Berlin"]]
    )  # [1, 1]
    ```
    """

    # TODO: add max_requests parameter to limit the number of requests made
    def __init__(self, system_prompt: Optional[str] = None, max_tries: int = 5, max_workers: int = 8):
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt = system_prompt
        self.max_tries = max_tries
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self) -> None:
        self.thread_pool_executor.shutdown()

    @abstractmethod
    def get_response(self, content: str) -> str:
        """
        Get the response from the API for the given content.

        Args:
            content (`str`): The string content.

        Returns:
            The response from the API as a string.
        """

        raise NotImplementedError("Judge subclasses must implement this method.")

    def judge_single(self, prompt: str, completion_pair: List[str], shuffle_order: bool = True) -> int:
        flipped = random.choice([True, False]) if shuffle_order else False
        completion_pair = completion_pair[::-1] if flipped else completion_pair

        retry = 0
        while retry < self.max_tries:
            content = self.system_prompt.format(
                prompt=prompt, response1=completion_pair[0], response2=completion_pair[1]
            )
            reply = self.get_response(content)
            reply = reply.strip()

            if reply in ["0"]:
                return 0 if not flipped else 1
            elif reply in ["1"]:
                return 1 if not flipped else 0
            else:
                logging.info(f"Judge gave response `{reply}` instead of the expected 0 or 1. Retrying.")
                retry += 1

        logging.info(
            f"Max retries reached for prompt:\n\n{prompt}\nand completion pair:\n\n{completion_pair}\n\nReturning random choice."
        )
        return random.choice([0, 1])

    def judge(self, prompts: List[str], completion_pairs: List[List[str]], shuffle_order: bool = True) -> List[int]:
        futures = []
        for prompt, completion_pair in zip(prompts, completion_pairs):
            future = self.thread_pool_executor.submit(self.judge_single, prompt, completion_pair, shuffle_order)
            futures.append(future)

        return [f.result() for f in futures]


class PairRMJudge(BaseJudge):
    """
    LLM judge based on the PairRM model from AllenAI.

    See: https://huggingface.co/llm-blender/PairRM
    """

    def __init__(self):
        if not is_llmblender_available():
            raise ValueError("llm-blender is not installed. Please install it with 'pip install llm-blender'.")
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)

    def judge(self, prompts: List[str], completion_pairs: List[List[str]], shuffle_order: bool = True) -> List[int]:
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completion_pairs = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completion_pairs)]
        ranks = self.blender.rank(prompts, completion_pairs)
        ranks -= 1  # PairRM is 1-indexed, so we subtract 1 to make it 0-indexed
        if shuffle_order:
            # Flip back the ranks to the original order
            ranks[flip_mask] = ranks[flip_mask][:, ::-1]
        return ranks[:, 0].tolist()


class MockJudge(BaseJudge):
    """
    Mock judge that randomly selects a model for each completion pair.
    """

    def judge(self, prompts: List[str], completion_pairs: List[List[str]]) -> List[int]:
        return [random.choice([0, 1]) for _ in range(len(prompts))]


class MockAPIJudge(BaseAPIJudge):
    """
    Mock judge that returns a random choice instead of interacting with an API.
    """

    def get_response(self, content: str) -> str:
        return random.choice(["0", "1"])


class HuggingFaceJudge(BaseAPIJudge):
    """
    Judge based on the Hugging Face API.

    Args:
        model (`str`, *optional*): The model to use for the judge. Defaults to "meta-llama/Meta-Llama-3-70B-Instruct".
        system_prompt (`str`, *optional*): The system prompt to be used for the judge. If not provided, a default prompt is used.
        max_tries (`int`, *optional*): The maximum number of retries for a request. Defaults to 5.
        max_workers (`int`, *optional*): The maximum number of parallel requests. Defaults to 8.
        token (`str`, *optional*): The Hugging Face API token to use for the InferenceClient.
    """

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        system_prompt: Optional[str] = None,
        max_tries: int = 5,
        max_workers: int = 8,
        token: Optional[str] = None,
    ):
        super().__init__(system_prompt=system_prompt, max_tries=max_tries, max_workers=max_workers)
        self.client = InferenceClient(model=model, token=token)

    def get_response(self, content: str) -> str:
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": content}],
                max_tokens=1,
                stop=["<|eot_id|>"],  # For llama-3 models
            )
            return response.choices[0].message.content
        except HTTPError as e:
            logging.info(f"Unable to reach the Hugging Face API due to error: {e}\nReturning random choice (0,1)")
            return random.choice(["0", "1"])


class OpenAIJudge(BaseAPIJudge):
    """
    Judge based on the OpenAI API.

    Args:
        model (`str`, *optional*): The model to use for the judge. Defaults to "gpt-4-turbo-preview".
        system_prompt (`str`, *optional*): The system prompt to be used for the judge. If not provided, a default prompt is used.
        max_tries (`int`, *optional*): The maximum number of retries for a request. Defaults to 5.
        max_workers (`int`, *optional*): The maximum number of parallel requests. Defaults to 8.
    """

    def __init__(
        self,
        model="gpt-4-turbo-preview",
        system_prompt: Optional[str] = None,
        max_tries: int = 5,
        max_workers: int = 8,
    ):
        if not is_openai_available():
            raise ValueError("OpenAI client is not installed. Please install it with 'pip install openai'.")
        super().__init__(system_prompt=system_prompt, max_tries=max_tries, max_workers=max_workers)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def get_response(self, content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1,  # TODO: let users configure these variables
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            logging.warn(f"Unable to reach to OpenAI API due to error: {e}\nReturning random choice (0, 1)")
            return random.choice(["0", "1"])
