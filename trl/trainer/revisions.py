import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from huggingface_hub import InferenceClient

from ..import_utils import is_openai_available


if is_openai_available():
    from openai import OpenAI


DEFAULT_REVISION_PROMPT_TEMPLATE = """You are a teacher and your task is to minimally improve a student's answer. I will give you a {{task}} and a {{student_solution}}. Your job is to revise the {{student_solution}} such that it is clearer, more correct, and more engaging. Copy all non-corrected parts of the student's answer. Do not allude to the {{corrected_student_solution}} being a revision or a correction in your final solution. You must keep the length of the {{corrected_student_solution}} roughly equal to the length of the {{student_solution}}.

{{task}}: {prompt}

{{student_solution}}: {response}

-----------------

Let's first think step by step with a {{teacher_reasoning}} to decide how to improve the {{student_solution}}, then give the {{corrected_student_solution}}. Mention the {{teacher_reasoning}} and {{corrected_student_solution}} identifiers to structure your answer.

"""


def _parse_default_revision(content: str) -> Tuple[str, str]:
    if "**Corrected Student Solution:**" in content:
        splits = content.split("**Corrected Student Solution:**")
    elif "{corrected_student_solution}:" in content:
        splits = content.split("{corrected_student_solution}:")
    elif "{corrected_student_solution}" in content:
        splits = content.split("{corrected_student_solution}")

    if len(splits) >= 2:
        edit = splits[1]
        edit = edit.removesuffix("\n\n").strip()

        rational = splits[0]
        if "{teacher_reasoning}" in rational:
            rational = rational.split("{teacher_reasoning}")[1].removesuffix(":").strip()
        rational = rational.removesuffix("\n\n").strip()
    else:
        raise RuntimeError("Bad parsing")

    return edit, rational


class BaseReviser(ABC):
    """
    Base class for revisers. The subclasses of this class should implement the `revise` method.

    Example:
    ```python
    class MyReviser(BaseReviser):
        def revise(self, prompts, completions):
            return ...  # Your revision logic here

    reviser = MyReviser()
    reviser.revise(
        prompts=["The capital of France is", "Berlin is"],
        completions=[" Marseille", " the second largest city in Germany"]
    )  # [" Paris", " the largest city in Germany"]
    ```
    """

    @abstractmethod
    def revise(self, prompts: List[str], completions: List[str]) -> List[str]:
        raise NotImplementedError("Reviser subclasses must implement the `revise` method.")


class IdentityReviser(BaseReviser):
    """
    Identity revisions, for testing purposes.
    """

    def revise(self, prompts: List[str], completions: List[str]) -> List[str]:
        return completions


class HfReviser(BaseReviser):
    """
    Reviser based on the Hugging Face API with chat completion.

    Args:
        model (`str`, *optional*): The model to use for the judge. Defaults to "meta-llama/Meta-Llama-3-70B-Instruct".
        token (`str`, *optional*): The Hugging Face API token to use for the InferenceClient.
        prompt_template (`str`, *optional*): The prompt template to be used for the reviser. If not provided, a default prompt is used.
            Note that the prompt template should contain the following placeholders: `{prompt}` and `{response}`.
        max_tokens (`int`, *optional*): Maximum ammount of tokens to be generated for the revision and revision rational.
    """

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        token: Optional[str] = None,
        prompt_template: Optional[str] = None,
        max_tokens: int = 2048,
    ):
        self.client = InferenceClient(model=model, token=token)
        self.prompt_template = prompt_template or DEFAULT_REVISION_PROMPT_TEMPLATE
        self.max_tokens = max_tokens

    def revise(self, prompts: List[str], completions: List[str]) -> List[str]:
        # Define a function to get the revision for a single prompt, will be called concurrently
        def get_revision(prompt, completion):
            content = self.prompt_template.format(prompt=prompt, response=completion)
            completion = self.client.chat_completion(
                messages=[{"role": "user", "content": content}], max_tokens=self.max_tokens
            )

            # Parse response
            response = completion.choices[0].message.content
            edit, rational = _parse_default_revision(response)

            return edit, rational

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outputs = list(executor.map(get_revision, prompts, completions))

        edits = [o[0] for o in outputs]
        # rationals = [o[1] for o in outputs]

        return edits


class OpenAIReviser(BaseReviser):
    """
    Reviser based on the OpenAI API.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*): The model to use for the judge. Defaults to `"gpt-4-turbo-preview"`.
        prompt_template (`str`, *optional*): The prompt template to be used for the reviser. If not provided, a default prompt is used.
            Note that the prompt template should contain the following placeholders: `{prompt}` and `{response}`.
        max_requests (`int`, *optional*): The maximum number of requests to make to the OpenAI API. Defaults to 1000. If set to `None`, there is no limit.
        max_tokens (`int`, *optional*): Maximum ammount of tokens to be generated for the revision and revision rational.

    """

    def __init__(
        self,
        model="gpt-4-turbo-preview",
        prompt_template: Optional[str] = None,
        max_requests: Union[int, None] = 1_000,
        max_tokens: int = 2048,
    ):
        if not is_openai_available():
            raise ValueError("OpenAI client is not installed. Please install it with 'pip install openai'.")
        self.client = OpenAI()
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_REVISION_PROMPT_TEMPLATE
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.num_requests = 0
        self._warned = False

    def revise(self, prompts: List[str], completions: List[str]) -> List[str]:
        # Check if the limit of requests is reached, if so, use random choice instead
        # if self.max_requests is not None and self.num_requests >= self.max_requests:
        #     if not self._warned:  # Print the warning only once
        #         logging.warning(
        #             f"Reached the maximum number of requests ({self.max_requests}). From now on, using random choice instead. "
        #             " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
        #         )
        #         self._warned = True
        #     return [random.choice([0, 1]) for _ in prompts]

        # Shuffle the order of the completions to avoid positional bias
        # if shuffle_order:
        #     flip_mask = np.random.choice([True, False], size=len(prompts))
        #     completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the revision for a single prompt, will be called concurrently

        def get_revision(prompt, completion):
            content = self.prompt_template.format(prompt=prompt, response=completion)
            completion = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": content}], max_tokens=self.max_tokens
            )

            # Parse response
            response = completion.choices[0].message.content
            edit, rational = _parse_default_revision(response)

            return edit, rational

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outputs = list(executor.map(get_revision, prompts, completions))

        edits = [o[0] for o in outputs]
        # rationals = [o[1] for o in outputs]

        return edits
