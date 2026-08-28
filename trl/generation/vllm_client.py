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

import atexit
import base64
import copy
import logging
import math
import socket
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from urllib.parse import urlparse

import torch
from requests.adapters import HTTPAdapter
from torch import nn
from transformers.utils import get_json_schema
from urllib3.util.retry import Retry

from ..import_utils import is_requests_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
    from vllm.utils.network_utils import get_ip


# `/start_weight_update` and `/finish_weight_update` were introduced in vLLM 0.21.0. Before that, `/update_weights`
# ran the whole weight update lifecycle (layerwise reload init and finalize) on its own.
_HAS_WEIGHT_UPDATE_LIFECYCLE = is_vllm_available(min_version="0.21.0")


logger = logging.getLogger(__name__)


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """
    Convert TRL-style messages to the OpenAI format understood by the vLLM server, by inlining images as data URLs.

    Args:
        messages (`list[dict]`):
            Messages whose content parts may be `{"type": "image", "image": <PIL.Image>}` or `{"type": "image_pil",
            "image_pil": <PIL.Image>}`.

    Returns:
        `list[dict]`: Messages with image parts replaced by `{"type": "image_url", "image_url": {"url": "data:..."}}`.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message["content"], list):
            for idx, part in enumerate(message["content"]):
                key = "image" if part["type"] == "image" else "image_pil" if part["type"] == "image_pil" else None
                if key is not None:
                    url = f"data:image/png;base64,{pil_to_base64(part[key])}"
                    message["content"][idx] = {"type": "image_url", "image_url": {"url": url}}
    return messages


def parse_logprobs(choices_logprobs: list[dict | None]) -> tuple[list | None, list | None]:
    """
    Parse the logprobs of a batch of choices into two lists of shape (num_sequences, seq_len, num_logprobs).

    The server is asked for tokens as token IDs (`return_tokens_as_token_ids`), so every token is a `"token_id:<id>"`
    string. Entries are sorted by descending probability, which matches the rank ordering used by vLLM.

    Args:
        choices_logprobs (`list[dict]`):
            Logprobs of each choice, either `{"tokens": ..., "top_logprobs": ...}` (completions) or `{"content":
            [{"token": ..., "logprob": ..., "top_logprobs": ...}]}` (chat completions and token IDs).

    Returns:
        Tuple of (logprobs, logprob_token_ids), or `(None, None)` when the server returned no logprob.
    """
    if choices_logprobs[0] is None:
        return None, None

    all_logprobs = []
    all_token_ids = []
    for logprobs in choices_logprobs:
        if "content" in logprobs:  # chat completions
            positions = [
                (entry["token"], entry["logprob"], {top["token"]: top["logprob"] for top in entry["top_logprobs"]})
                for entry in logprobs["content"]
            ]
        else:  # completions
            positions = list(
                zip(logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"], strict=True)
            )

        seq_logprobs = []
        seq_token_ids = []
        for token, logprob, top_logprobs in positions:
            # The sampled token is always included, even when it falls outside the top-N.
            top_logprobs = {token: logprob, **(top_logprobs or {})}
            items = sorted(top_logprobs.items(), key=lambda item: -item[1])
            seq_token_ids.append([int(token.removeprefix("token_id:")) for token, _ in items])
            seq_logprobs.append([None if math.isnan(logprob) else logprob for _, logprob in items])
        all_logprobs.append(seq_logprobs)
        all_token_ids.append(seq_token_ids)
    return all_logprobs, all_token_ids


class VLLMClient:
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start a vLLM server with `vllm serve`, see the vLLM integration
    guide for the required flags.

    Args:
        base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `host` and `server_port` are
            ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server. Ignored if `base_url` is provided.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen2.5-7B --weight-transfer-config '{"backend": "nccl"}' \
              --logprobs-mode processed_logprobs --max-logprobs -1
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.generation.vllm_client import VLLMClient

        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        {'prompt_ids': [[9707, 11, 15235, 0],
                        [40451, 752, 264, 21646]],
         'completion_ids': [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
                            [911, 98072, 2142, 624, 45, 51426, 2142, 374, 279, 16396, 429, 4302, 702, 36988, 7290, 476]],
         'logprobs': [[[-1.6612], [-0.0081], [-1.5189], [-0.0123], [-1.2045], [-0.6227], [-2.9791], [-2.8387], [-0.1267], [-0.0366], [-2.6528], [-0.3197], [-0.0001], [-1.8174], [-0.0251], [-1.473]],
                      [[-0.018], [-10.7331], [-0.1605], [-0.891], [-3.7945], [-0.0127], [-0.3073], [-1.1648], [-1.8025], [-0.409], [-0.0256], [-1.6127], [-2.2935], [-4.1785], [-0.6531], [-0.2629]]],
         'logprob_token_ids': [[[2980], [498], [1492], [752], [448], [264], [13027], [8645], [30], [358], [2776], [4460], [311], [3270], [264], [2025]],
                               [[911], [98072], [2142], [624], [45], [51426], [2142], [374], [279], [16396], [429], [4302], [702], [36988], [7290], [476]]]}

        >>> from transformers import AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator(device="cuda")
        >>> client.update_model_params(model)
        ```

        There are several ways to initialize the client:

        ```python
        >>> VLLMClient(base_url="http://localhost:8000")
        >>> VLLMClient(base_url="http://192.168.1.100:8000")
        >>> VLLMClient(host="localhost", server_port=8000)
        >>> VLLMClient(host="192.168.1.100", server_port=8000)
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install trl[vllm]`.")

        self.session = requests.Session()

        # Configure retries for HTTP requests made through this session.
        # This is not strictly required for correctness, but it helps make training more robust to rare, transient
        # failures (network hiccups, temporary 5xx errors, overloaded servers). Without this, such failures could cause
        # an otherwise healthy training run to fail.
        retry_strategy = Retry(
            total=5,  # global cap on the total number of retries across all failure types
            connect=5,  # retry connection-level failures (DNS issues, refused connections, etc)
            read=5,  # retry failures while reading the response after the connection was successfully established
            status=3,  # retry a limited number of times when we receive certain HTTP error responses from the server
            status_forcelist=[500, 502, 503],  # only retry on server-side errors that are usually temporary
            backoff_factor=2,  # exponential backoff between retries (2s, 4s, 8s, ...)
            allowed_methods=["POST", "GET"],  # allow POST as well, even though we're not sure it's safe here
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port
        self.communicator = None
        self._updating_weights = False  # set while inside `weight_update`
        self.check_server(connection_timeout)  # check server and fail after timeout
        self.model = self._get(f"{self.base_url}/v1/models")["data"][0]["id"]

    def _get(self, url: str, **kwargs) -> dict:
        response = self.session.get(url, **kwargs)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    def _post(self, url: str, **kwargs) -> dict:
        response = self.session.post(url, **kwargs)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"{self.base_url}/health"
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        "sure the server is running by running `vllm serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    logger.info("Server is up!")
                    return

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def get_world_size(self) -> int:
        """
        Returns the number of workers of the vLLM server, i.e. `tensor_parallel_size * data_parallel_size`.
        """
        return self._get(f"{self.base_url}/get_world_size")["world_size"]

    def image_features(self, images: list[list | None], max_concurrent_requests: int = 64) -> list[dict | None]:
        """
        Processes images server-side into the features that pair with token IDs in
        [`~generation.vllm_client.VLLMClient.generate`].

        The server generates from either token IDs or images, never both, so images are sent on their own first. The
        features it returns carry the processed image data and its hashes, which depend on the image alone, and stay
        valid for any token sequence they are paired with.

        Args:
            images (`list[list[PIL.Image] | None]`):
                List of image lists for VLM support. Each element is a list of PIL images for the corresponding prompt,
                or `None` if no images for that prompt.
            max_concurrent_requests (`int`, *optional*, defaults to `64`):
                Maximum number of prompts processed at the same time, as the endpoint takes one at a time.

        Returns:
            `list[dict]`: Processed image data for each prompt, `None` where the prompt had no image.
        """

        def send(images_for_prompt):
            if not images_for_prompt:
                return None
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image} for image in images_for_prompt]}
            ]
            rendered = self._post(
                f"{self.base_url}/v1/chat/completions/render",
                json={"model": self.model, "messages": to_openai_messages(messages), "max_tokens": 1},
            )
            return rendered["features"]

        with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(images))) as executor:
            return list(executor.map(send, images))

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        features: list[dict | None] | None = None,
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        logprobs: int | None = 0,
        structured_outputs_regex: str | None = None,
        generation_kwargs: dict | None = None,
    ) -> dict[str, list[list[int]]]:
        """
        Generates model completions for the provided prompts.

        Args:
            prompts (`list[str]` or `list[list[int]]`):
                List of text prompts or list of token ID lists for which the model will generate completions.
            features (`list[dict]`, *optional*):
                Processed image data from [`~generation.vllm_client.VLLMClient.image_features`], one per prompt (`None`
                for prompts without images). When provided, prompts must be token IDs.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `0`):
                Top-k sampling parameter. `0` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            logprobs (`int` or `None`, *optional*, defaults to `0`):
                Number of top logprobs to return per token. When 0, only the sampled token's logprob is returned. When
                N>0, returns up to N+1 logprobs sorted by descending probability, because vLLM always includes the
                sampled token's logprob (which may fall outside the top-N). When `None`, no logprob is returned.
            structured_outputs_regex (`str`, *optional*):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters, passed as-is in the request body. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.

        Returns:
            `dict` with keys:
                - `prompt_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the tokenized input prompts.
                - `completion_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the model-generated completions for each prompt.
                - `logprobs` (`list[list[list[float]]]`):
                    Per-token logprobs of shape (num_sequences, seq_len, num_logprobs), sorted by descending
                    probability.
                - `logprob_token_ids` (`list[list[list[int]]]`):
                    Token IDs corresponding to each logprob, same shape as `logprobs`.
        """
        sampling_params = {
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        }
        if structured_outputs_regex is not None:
            sampling_params["structured_outputs"] = {"regex": structured_outputs_regex}
        sampling_params.update(generation_kwargs or {})

        if features is not None:
            return self._generate_from_features(prompts, features, sampling_params)

        payload = {
            "model": self.model,
            "prompt": prompts,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            **sampling_params,
        }
        choices = self._post(f"{self.base_url}/v1/completions", json=payload)["choices"]

        # Choices are returned prompt-major: the n completions of a prompt come before those of the next prompt.
        prompt_ids = [choice["prompt_token_ids"] for choice in choices[::n]]
        completion_ids = [choice["token_ids"] for choice in choices]
        logprobs, logprob_token_ids = parse_logprobs([choice["logprobs"] for choice in choices])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
        }

    def _generate_from_features(
        self,
        prompts: list[list[int]],
        features: list[dict | None],
        sampling_params: dict,
        max_concurrent_requests: int = 64,
    ) -> dict[str, list[list[int]]]:
        """Generate from token IDs paired with multimodal features, one request per prompt."""
        # The server leaves the default output kind on non-streaming requests, under which it returns only the
        # sequences that finished last, silently dropping the others when n > 1. See vLLM PR #52399.
        sampling_params = {**sampling_params, "output_kind": 2}  # FINAL_ONLY

        def send(index):
            return self._post(
                f"{self.base_url}/inference/v1/generate",
                json={
                    "request_id": f"trl-{uuid.uuid4().hex}",
                    "model": self.model,
                    "token_ids": prompts[index],
                    "features": features[index],
                    "sampling_params": sampling_params,
                },
            )

        with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(prompts))) as executor:
            responses = list(executor.map(send, range(len(prompts))))

        choices = [choice for response in responses for choice in response["choices"]]
        logprobs, logprob_token_ids = parse_logprobs([choice["logprobs"] for choice in choices])
        return {
            "prompt_ids": list(prompts),  # the server generates from what it was given, so it echoes back unchanged
            "completion_ids": [choice["token_ids"] for choice in choices],
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
        }

    def chat(
        self,
        messages: list[list[dict]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_tokens: int = 16,
        logprobs: int | None = 0,
        structured_outputs_regex: str | None = None,
        generation_kwargs: dict | None = None,
        chat_template_kwargs: dict | None = None,
        tools: list | None = None,
        chat_template: str | None = None,
        max_concurrent_requests: int = 64,
    ) -> dict[str, list[list[int]]]:
        """
        Generates model completions for the provided chat messages.

        The server renders the chat template, so this is the path to use for multimodal prompts: images travel as part
        of the messages, while [`~generation.vllm_client.VLLMClient.generate`] only accepts text or token IDs.

        Args:
            messages (`list[list[dict]]`):
                List of message lists for which the model will generate completions. Each message is a dictionary with
                keys like "role" and "content".
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each message list.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. 1.0 means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter.`1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `0`):
                Top-k sampling parameter. `0` means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each message list.
            logprobs (`int` or `None`, *optional*, defaults to `0`):
                Number of top logprobs to return per token. When 0, only the sampled token's logprob is returned. When
                N>0, returns up to N+1 logprobs sorted by descending probability, because vLLM always includes the
                sampled token's logprob (which may fall outside the top-N). When `None`, no logprob is returned.
            structured_outputs_regex (`str`, *optional*):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters, passed as-is in the request body. This can include parameters like
                `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they
                will override them.
            chat_template_kwargs (`dict`, *optional*):
                Additional keyword arguments to customize the chat template used by the model.
            tools (`list[dict | Callable]`, *optional*):
                List of tool functions available for tool calling during chat generation.
            chat_template (`str`, *optional*):
                Template to use for structuring the chat. If not provided, the model's default chat template will be
                used.
            max_concurrent_requests (`int`, *optional*, defaults to `64`):
                Maximum number of conversations sent to the server at the same time. The chat completions endpoint
                takes a single conversation per request, so they are dispatched concurrently.

        Returns:
            `dict` with keys:
                - `prompt_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the tokenized input messages.
                - `completion_ids` (`list[list[int]]`):
                    List of lists of token IDs representing the model-generated completions for each message list.
                - `logprobs` (`list[list[list[float]]]`):
                    Per-token logprobs of shape (num_sequences, seq_len, num_logprobs), sorted by descending
                    probability.
                - `logprob_token_ids` (`list[list[list[int]]]`):
                    Token IDs corresponding to each logprob, same shape as `logprobs`.
        """
        payload = {
            "model": self.model,
            "n": n,
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "logprobs": logprobs is not None,
            "top_logprobs": logprobs,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        }
        if tools:  # the server rejects an empty list, so only send it when there is something to send
            payload["tools"] = [get_json_schema(tool) if callable(tool) else tool for tool in tools]
            # Tool calls are parsed from the completion by the caller, so the server only has to render the tools into
            # the prompt. Without this, it defaults to `"auto"` and refuses the request unless it was started with
            # `--enable-auto-tool-choice --tool-call-parser`.
            payload["tool_choice"] = "none"
        if chat_template is not None:
            payload["chat_template"] = chat_template
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if structured_outputs_regex is not None:
            payload["structured_outputs"] = {"regex": structured_outputs_regex}
        payload.update(generation_kwargs or {})

        def send(conversation):
            return self._post(
                f"{self.base_url}/v1/chat/completions", json={**payload, "messages": to_openai_messages(conversation)}
            )

        with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(messages))) as executor:
            responses = list(executor.map(send, messages))

        prompt_ids = [response["prompt_token_ids"] for response in responses]
        choices = [choice for response in responses for choice in response["choices"]]
        completion_ids = [choice["token_ids"] for choice in choices]
        logprobs, logprob_token_ids = parse_logprobs([choice["logprobs"] for choice in choices])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
        }

    def get_sequence_logprobs(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        top_logprobs: int = 100,
        temperature: float = 1.0,
        chunk_size: int = 0,
        max_concurrent_requests: int = 4,
    ) -> dict[str, list]:
        """
        Computes teacher logprobs for existing token sequences without generating new tokens.

        Sends full sequences (prompt + completion) to the vLLM server and retrieves per-token top-k logprobs for the
        completion region only. This is used for knowledge distillation where the teacher model evaluates existing
        sequences rather than generating new ones.

        When `chunk_size > 0`, splits the batch into chunks and dispatches them concurrently, so that the server can
        start working before the whole batch has been sent.

        Args:
            sequences (`list[list[int]]`):
                List of full token ID sequences (prompt + completion).
            prompt_lengths (`list[int]`):
                Number of prompt tokens in each sequence. Logprobs are returned starting from this position.
            top_logprobs (`int`, *optional*, defaults to `100`):
                Number of top logprobs to return per token position.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature used when scoring the teacher distribution.
            chunk_size (`int`, *optional*, defaults to `0`):
                If > 0, split batch into chunks of this size and dispatch concurrently. If 0, send the entire batch in
                a single request.
            max_concurrent_requests (`int`, *optional*, defaults to `4`):
                Maximum number of concurrent requests when using chunked dispatch.

        Returns:
            `dict` with keys:
                - `logprobs` (`list[list[list[float]]]`):
                    Teacher's top-k logprobs per completion token, of shape (batch, completion_len, top_logprobs),
                    sorted by descending probability and padded with `-inf`. Used for the forward KL term.
                - `logprob_token_ids` (`list[list[list[int]]]`):
                    Token IDs corresponding to each logprob, same shape as `logprobs`.
                - `actual_logprobs` (`list[list[list[float]]]`):
                    Teacher logprob of the actual token at each position, of shape (batch, completion_len, 1), or
                    `-inf` when the token falls outside the top-k. Used for the reverse KL term.
                - `actual_token_ids` (`list[list[list[int]]]`):
                    Actual token IDs, same shape as `actual_logprobs`.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        chunk_size = chunk_size if chunk_size > 0 else len(sequences)
        chunks = [
            (sequences[idx : idx + chunk_size], prompt_lengths[idx : idx + chunk_size])
            for idx in range(0, len(sequences), chunk_size)
        ]

        def send(chunk):
            chunk_sequences, chunk_prompt_lengths = chunk
            payload = {
                "model": self.model,
                "prompt": chunk_sequences,
                "max_tokens": 1,
                "temperature": temperature,
                "prompt_logprobs": top_logprobs,
            }
            choices = self._post(f"{self.base_url}/v1/completions", json=payload)["choices"]
            return [
                self._format_sequence_logprobs(choice, sequence, prompt_length, top_logprobs)
                for choice, sequence, prompt_length in zip(choices, chunk_sequences, chunk_prompt_lengths, strict=True)
            ]

        with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(chunks))) as executor:
            results = [result for chunk_results in executor.map(send, chunks) for result in chunk_results]

        return {key: [result[key] for result in results] for key in results[0]}

    @staticmethod
    def _format_sequence_logprobs(choice: dict, sequence: list[int], prompt_length: int, top_logprobs: int) -> dict:
        """Slice the completion region out of a choice's prompt logprobs, sort it by rank and pad it to `top_logprobs`."""
        logprobs, logprob_token_ids, actual_logprobs, actual_token_ids = [], [], [], []
        for position in range(prompt_length, len(choice["prompt_logprobs"])):
            position_logprobs = choice["prompt_logprobs"][position] or {}
            items = sorted(position_logprobs.items(), key=lambda item: item[1]["rank"])[:top_logprobs]
            values = [-math.inf if math.isnan(item["logprob"]) else item["logprob"] for _, item in items]
            token_ids = [int(token_id) for token_id, _ in items]
            # Pad so that every position has exactly `top_logprobs` entries, as callers index it as an array.
            logprobs.append(values + [-math.inf] * (top_logprobs - len(values)))
            logprob_token_ids.append(token_ids + [0] * (top_logprobs - len(token_ids)))
            actual = position_logprobs.get(str(sequence[position]))
            actual_logprobs.append(
                [-math.inf if actual is None or math.isnan(actual["logprob"]) else actual["logprob"]]
            )
            actual_token_ids.append([sequence[position]])
        return {
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
            "actual_logprobs": actual_logprobs,
            "actual_token_ids": actual_token_ids,
        }

    def init_communicator(self, device: torch.device | str | int = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process. It's the device that will be used for the weights synchronization. Can
                be a `torch.device` object, a string like `'cuda:0'`, or an integer device index.
        """
        # The trainer joins the vLLM workers as an extra rank; it is rank 0, so the workers are offset by one.
        world_size = self.get_world_size() + 1
        init_info = {
            "master_address": get_ip(),
            "master_port": self.group_port,
            "rank_offset": 1,
            "world_size": world_size,
        }

        # vLLM builds the trainer side of the group on the current device, so point it at the requested one. A device
        # without an index (e.g. `torch.device("cuda")`) means the current device, which is already the right one.
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        if device.type != "cuda":
            raise NotImplementedError(
                f"Weight synchronization runs on vLLM's NCCL weight-transfer engine, which needs a CUDA device, got "
                f"'{device}'. Generation through the server is unaffected."
            )
        if device.index is not None:
            torch.cuda.set_device(device)

        # The server blocks in the NCCL rendezvous while handling the request, so it must run concurrently with the
        # trainer joining the same rendezvous.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._post, f"{self.base_url}/init_weight_transfer_engine", json={"init_info": init_info}
            )
            self.communicator = NCCLWeightTransferEngine.trainer_init(init_info)
            future.result()

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    @contextmanager
    def weight_update(self):
        """
        Groups several [`~generation.vllm_client.VLLMClient.update_named_param`] calls into a single weight update.

        The server prepares the model once on entry and finalizes it once on exit, instead of doing it per tensor.

        Examples:

        ```python
        >>> with client.weight_update():
        ...     for name, param in model.named_parameters():
        ...         client.update_named_param(name, param.data)
        ```
        """
        self._start_weight_update()
        self._updating_weights = True
        try:
            yield
        finally:
            self._updating_weights = False
            self._finish_weight_update()

    def _start_weight_update(self):
        if _HAS_WEIGHT_UPDATE_LIFECYCLE:
            self._post(f"{self.base_url}/start_weight_update", json={})

    def _finish_weight_update(self):
        if _HAS_WEIGHT_UPDATE_LIFECYCLE:
            self._post(f"{self.base_url}/finish_weight_update", json={})

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        metadata = [(name, str(weights.dtype).removeprefix("torch."), list(weights.shape))]
        self.update_named_params(metadata, iter([(name, weights)]))

    def update_named_params(
        self, metadata: list[tuple[str, str, list[int]]], named_params: Iterator[tuple[str, torch.Tensor]]
    ):
        """
        Updates the model weights of the server, by streaming them over the weight update group.

        The server needs to know what it is about to receive, so the tensors are announced first, then broadcast in the
        same order.

        Args:
            metadata (`list[tuple[str, str, list[int]]]`):
                One `(name, dtype, shape)` triplet per tensor, in the order they are streamed. Dtypes are vLLM style,
                i.e. `"bfloat16"` rather than `"torch.bfloat16"`.
            named_params (`Iterator[tuple[str, torch.Tensor]]`):
                Iterator yielding the `(name, tensor)` pairs described by `metadata`.
        """
        names, dtype_names, shapes = (list(field) for field in zip(*metadata, strict=True))
        update_info = {"names": names, "dtype_names": dtype_names, "shapes": shapes, "packed": True}

        if not self._updating_weights:
            self._start_weight_update()
        # The workers block in the NCCL receive while handling the request, so it must run concurrently with the
        # trainer-side broadcast.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._post, f"{self.base_url}/update_weights", json={"update_info": update_info})
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=named_params,
                trainer_args=NCCLTrainerSendWeightsArgs(group=self.communicator, packed=True),
            )
            future.result()
        if not self._updating_weights:
            self._finish_weight_update()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        metadata = [
            (name, str(param.dtype).removeprefix("torch."), list(param.shape))
            for name, param in model.named_parameters()
        ]
        self.update_named_params(metadata, ((name, param.data) for name, param in model.named_parameters()))

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        self._post(f"{self.base_url}/reset_prefix_cache")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        if self.communicator is not None:
            # The group holds a socket to the trainer, which the server would keep waiting on.
            self.communicator.group.store = None
            self.communicator.group.socket = None
            self.communicator = None


# Example usage
if __name__ == "__main__":
    client = VLLMClient()
    client.init_communicator(device="cuda")

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32)
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)
