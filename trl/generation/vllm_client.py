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
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from urllib.parse import urlparse

import torch
import torch.distributed.distributed_c10d as c10d
from requests.adapters import HTTPAdapter
from torch import nn
from transformers import is_torch_xpu_available
from transformers.utils import get_json_schema
from urllib3.util.retry import Retry

from ..import_utils import is_openai_available, is_requests_available, is_vllm_ascend_available, is_vllm_available


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_openai_available():
    from openai import OpenAI


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator


logger = logging.getLogger(__name__)


def pil_to_base64(image, format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_data_url(image, format: str = "PNG") -> str:
    """Encode a PIL image as an OpenAI-style `image_url` data URI."""
    return f"data:image/{format.lower()};base64,{pil_to_base64(image, format=format)}"


class VLLMClient:
    """
    A client class to interact with a TRL vLLM server.

    Uses the OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints under the hood. The server also
    exposes TRL-only side-channel endpoints (weight sync, prefix-cache reset, batched teacher logprobs) that this
    client wraps.

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
        max_chat_workers (`int`, *optional*, defaults to `64`):
            Maximum number of concurrent threads used to fan out batched chat requests. The server gathers concurrent
            requests into a batched `vLLM.chat()` call, so a higher value here lets the server batch more aggressively.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.generation.vllm_client import VLLMClient

        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        {'prompt_ids': [[...], [...]], 'completion_ids': [[...], [...]], 'logprobs': None, 'logprob_token_ids': None}

        >>> from transformers import AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.init_communicator(device="cuda")
        >>> client.update_model_params(model)
        ```

        Because the server is OpenAI-compatible, you can also use the official OpenAI SDK directly:

        ```python
        >>> from openai import OpenAI

        >>> client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        >>> client.chat.completions.create(model="Qwen/Qwen2.5-7B", messages=[{"role": "user", "content": "Hi"}])
        ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
        max_chat_workers: int = 64,
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_openai_available():
            raise ImportError("openai is not installed. Please install it with `pip install openai`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install trl[vllm]`.")

        # requests session for TRL-only side-channel endpoints. The OpenAI SDK handles retries for /v1/* itself.
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            connect=5,
            read=5,
            status=3,
            status_forcelist=[500, 502, 503],
            backoff_factor=2,
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if base_url is not None:
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or "http"
            self.base_url = f"{scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f"http://{self.host}:{self.server_port}"
        self.group_port = group_port

        self.check_server(connection_timeout)  # check server and fail after timeout

        # OpenAI client for /v1/* endpoints. `api_key` is required by the SDK but the server ignores it.
        self._openai = OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY", max_retries=5)

        # Resolve the served model name once so callers don't have to pass it on every request.
        self._model_name = self._openai.models.list().data[0].id

        # Pool for fanning out concurrent /v1/chat/completions calls (one per conversation).
        self._chat_pool = ThreadPoolExecutor(max_workers=max_chat_workers)

        self.communicator = None

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
                        "sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    if "X-Forwarded-For" in response.headers:
                        self.host = response.headers["X-Forwarded-For"]
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    # --------------------------------------------------------------------------------------------
    # Generation — OpenAI-compatible endpoints
    # --------------------------------------------------------------------------------------------

    @staticmethod
    def _extract_choice_logprobs(choice: dict) -> tuple[list[list[float | None]] | None, list[list[int]] | None]:
        """Extract TRL-format `(top_logprobs_values, top_logprobs_token_ids)` from a choice's logprobs payload.

        Returns `(None, None)` when the choice carries no logprobs.
        """
        lp = choice.get("logprobs")
        if not lp:
            return None, None
        # /v1/completions encodes them as flat fields, /v1/chat/completions packs them into `content` entries.
        if isinstance(lp, dict) and "top_logprobs_values" in lp:
            return lp.get("top_logprobs_values"), lp.get("top_logprobs_token_ids")
        if isinstance(lp, dict) and "content" in lp and lp["content"] is not None:
            values = [[entry["logprob"] for entry in pos["top_logprobs"]] for pos in lp["content"]]
            token_ids = [[entry["token_id"] for entry in pos["top_logprobs"]] for pos in lp["content"]]
            return values, token_ids
        return None, None

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        images: list | None = None,
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
        Generates model completions for the provided prompts using `/v1/completions`.

        Args:
            prompts (`list[str]` or `list[list[int]]`):
                List of text prompts or list of token ID lists for which the model will generate completions.
            images (`list[list[PIL.Image] | None]`, *optional*):
                List of image lists for VLM support. Each element is a list of PIL images for the corresponding prompt,
                or `None` if no images for that prompt. Passed as the `images` extension of `/v1/completions`.
            n (`int`, *optional*, defaults to `1`):
                Number of completions to generate for each prompt.
            repetition_penalty (`float`, *optional*, defaults to `1.0`):
                Parameter for repetition penalty. `1.0` means no penalty.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature parameter for sampling. Higher values increase diversity.
            top_p (`float`, *optional*, defaults to `1.0`):
                Top-p sampling parameter. `1.0` means no truncation.
            top_k (`int`, *optional*, defaults to `0`):
                Top-k sampling parameter. `0` (or `-1`) means no truncation.
            min_p (`float`, *optional*, defaults to `0.0`):
                Minimum probability for sampling.
            max_tokens (`int`, *optional*, defaults to `16`):
                Maximum number of tokens to generate for each prompt.
            logprobs (`int` or `None`, *optional*, defaults to `0`):
                Number of top logprobs to return per token. When `0`, only the sampled token's logprob is returned.
                When `N > 0`, returns up to `N+1` logprobs sorted by descending probability. When `None`, no logprobs
                are requested.
            structured_outputs_regex (`str`, *optional*):
                Regular expression to guide the decoding process.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters to pass to the vLLM `SamplingParams`. If it contains keys that
                conflict with the other parameters, they will override them.

        Returns:
            `dict` with keys:
                - `prompt_ids` (`list[list[int]]`):
                    Token IDs for each input prompt.
                - `completion_ids` (`list[list[int]]`):
                    Token IDs for each generated completion (one per `prompt × n`).
                - `logprobs` (`list[list[list[float]]]` or `None`):
                    Per-token logprobs of shape `(num_sequences, seq_len, num_logprobs)`, sorted by descending
                    probability. `None` if no logprobs were requested.
                - `logprob_token_ids` (`list[list[list[int]]]` or `None`):
                    Token IDs corresponding to each logprob, same shape as `logprobs`.
        """
        # Encode images (if any) as base64 strings for the `images` extension field.
        encoded_images: list[list[str] | None] | None = None
        if images:
            encoded_images = [
                [pil_to_base64(img) for img in img_list] if img_list is not None else None for img_list in images
            ]

        extra_body: dict = {
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "guided_regex": structured_outputs_regex,
            "generation_kwargs": generation_kwargs or {},
        }
        if encoded_images is not None:
            extra_body["images"] = encoded_images

        # The openai SDK's `logprobs` is a count (int) for /v1/completions and bool for /v1/chat/completions.
        response = self._openai.completions.create(
            model=self._model_name,
            prompt=prompts,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=logprobs,
            extra_body=extra_body,
        )

        data = response.model_dump()
        choices = data["choices"]

        # Server emits all `n` choices for prompt 0, then prompt 1, etc. With `len(prompts)` prompts and `n`
        # completions each, the (prompt_idx, choice_idx) for `choices[k]` is `(k // n, k % n)`.
        num_prompts = len(choices) // n
        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float | None]] | None] = []
        all_logprob_token_ids: list[list[list[int]] | None] = []
        any_logprobs = False
        for prompt_idx in range(num_prompts):
            prompt_ids.append(choices[prompt_idx * n]["prompt_token_ids"])
            for choice_idx in range(n):
                choice = choices[prompt_idx * n + choice_idx]
                completion_ids.append(choice["token_ids"])
                values, token_ids = self._extract_choice_logprobs(choice)
                all_logprobs.append(values)
                all_logprob_token_ids.append(token_ids)
                if values is not None:
                    any_logprobs = True

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": all_logprobs if any_logprobs else None,
            "logprob_token_ids": all_logprob_token_ids if any_logprobs else None,
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
    ) -> dict[str, list[list[int]]]:
        """
        Generates model completions for the provided chat conversations using `/v1/chat/completions`.

        One HTTP request is sent per conversation (OpenAI semantics); the server batches concurrent requests internally
        so DP workers stay busy.

        Args:
            messages (`list[list[dict]]`):
                List of conversations. Each conversation is a list of message dicts with `"role"` / `"content"` keys.
                Image parts may be provided either in OpenAI's standard form (`{"type": "image_url", "image_url":
                {"url": "data:..."}}`) or as PIL parts (`{"type": "image_pil", "image_pil": <PIL.Image>}`); the latter
                are encoded to data URIs.
            n (`int`, *optional*, defaults to `1`):
                Number of completions per conversation.
            repetition_penalty, temperature, top_p, top_k, min_p, max_tokens, logprobs:
                Sampling parameters; see [`generate`].
            structured_outputs_regex (`str`, *optional*):
                Regular expression to guide decoding.
            generation_kwargs (`dict`, *optional*):
                Additional generation parameters forwarded to `SamplingParams`.
            chat_template_kwargs (`dict`, *optional*):
                Extra keyword arguments for the chat template.
            tools (`list[dict | Callable]`, *optional*):
                Tool functions / specs available for tool calling.
            chat_template (`str`, *optional*):
                Chat template override (forwarded as the `chat_template` extra-body field).

        Returns:
            `dict` with keys: `prompt_ids`, `completion_ids`, `logprobs`, `logprob_token_ids` — see [`generate`] for
            shapes.
        """
        # Convert PIL image parts to OpenAI's `image_url` form, leaving other parts untouched. Copy to avoid mutating
        # the caller's structures.
        messages = copy.deepcopy(messages)
        for conversation in messages:
            for message in conversation:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part.get("type") == "image_pil":
                            part["type"] = "image_url"
                            part["image_url"] = {"url": pil_to_data_url(part.pop("image_pil"))}

        # Convert callable tools to JSON schemas (`get_json_schema` handles function signatures).
        if isinstance(tools, list) and len(tools) > 0:
            tools = [get_json_schema(tool) if callable(tool) else tool for tool in tools]

        # OpenAI's chat completions `logprobs` is bool + `top_logprobs` int; map TRL's int convention.
        logprobs_bool = logprobs is not None
        top_logprobs = logprobs if logprobs and logprobs > 0 else None

        extra_body: dict = {
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "guided_regex": structured_outputs_regex,
            "generation_kwargs": generation_kwargs or {},
            "chat_template_kwargs": chat_template_kwargs or {},
        }
        if chat_template is not None:
            extra_body["chat_template"] = chat_template

        def _one(conversation):
            response = self._openai.chat.completions.create(
                model=self._model_name,
                messages=conversation,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs_bool,
                top_logprobs=top_logprobs,
                tools=tools,
                extra_body=extra_body,
            )
            return response.model_dump()

        responses = list(self._chat_pool.map(_one, messages))

        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        all_logprobs: list[list[list[float | None]] | None] = []
        all_logprob_token_ids: list[list[list[int]] | None] = []
        any_logprobs = False
        for response in responses:
            prompt_ids.append(response["prompt_token_ids"])
            for choice in response["choices"]:
                completion_ids.append(choice["token_ids"])
                values, token_ids = self._extract_choice_logprobs(choice)
                all_logprobs.append(values)
                all_logprob_token_ids.append(token_ids)
                if values is not None:
                    any_logprobs = True

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": all_logprobs if any_logprobs else None,
            "logprob_token_ids": all_logprob_token_ids if any_logprobs else None,
        }

    # --------------------------------------------------------------------------------------------
    # Weight sync — TRL-only endpoints
    # --------------------------------------------------------------------------------------------

    def init_communicator(self, device: torch.device | str | int = 0):
        """
        Initializes the weight update group in a distributed setup for model synchronization.

        Args:
            device (`torch.device`, `str`, or `int`, *optional*, defaults to `0`):
                Device of trainer main process. It's the device that will be used for the weights synchronization. Can
                be a `torch.device` object, a string like `'cuda:0'`, or an integer device index.
        """
        # Get the world size from the server
        url = f"{self.base_url}/world_size"
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f"{self.base_url}/init_communicator"
        # Will simplify it after torch xpu 2.9 support get uuid.
        if is_torch_xpu_available():
            if hasattr(torch.xpu.get_device_properties(device), "uuid"):
                client_device_uuid = str(torch.xpu.get_device_properties(device).uuid)
            else:
                client_device_uuid = "42"
        else:
            client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)

        # Set the weight update group's host to "0.0.0.0" so that
        # clients from different IPs can send updated weights
        response = self.session.post(
            url,
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(
                host_name=self.host, port=self.group_port, world_size=world_size, is_master=(self.rank == 0)
            )
            prefixed_store = c10d.PrefixStore("client2server", store)
            xccl_options = c10d.ProcessGroupXCCL.Options()
            pg = c10d.ProcessGroupXCCL(
                store=prefixed_store,
                rank=self.rank,
                size=world_size,
                options=xccl_options,
            )
            self.communicator = pg
        else:
            pg = StatelessProcessGroup.create(
                host=self.host, port=self.group_port, rank=self.rank, world_size=world_size
            )
            self.communicator = PyNcclCommunicator(pg, device=device)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"{self.base_url}/update_named_param"
        response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape})
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        if is_torch_xpu_available():
            # Use XCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weights, root=self.rank)
            self.communicator.barrier()
        else:
            # Use NCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weights, src=self.rank)
            self.communicator.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def get_sequence_logprobs(
        self,
        sequences: list[list[int]],
        prompt_lengths: list[int],
        top_logprobs: int = 100,
        temperature: float = 1.0,
        use_binary: bool = True,
        chunk_size: int = 0,
        max_concurrent_requests: int = 4,
    ) -> dict[str, list]:
        """
        Computes teacher logprobs for existing token sequences without generating new tokens.

        Sends full sequences (prompt + completion) to the vLLM server and retrieves per-token top-k logprobs for the
        completion region only. This is used for knowledge distillation where the teacher model evaluates existing
        sequences rather than generating new ones.

        When `chunk_size > 0`, splits the batch into chunks and dispatches them concurrently via a thread pool, keeping
        the server's data-parallel workers busy.

        When `use_binary=True`, uses base64-encoded numpy arrays for fast serialization instead of nested JSON lists.

        Args:
            sequences (`list[list[int]]`):
                List of full token ID sequences (prompt + completion).
            prompt_lengths (`list[int]`):
                Number of prompt tokens in each sequence. Logprobs are returned starting from this position.
            top_logprobs (`int`, *optional*, defaults to `100`):
                Number of top logprobs to return per token position.
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature used when scoring the teacher distribution.
            use_binary (`bool`, *optional*, defaults to `True`):
                Use binary (base64 numpy) response format for faster serialization.
            chunk_size (`int`, *optional*, defaults to `0`):
                If > 0, split batch into chunks of this size and dispatch concurrently. If 0, send the entire batch in
                a single request.
            max_concurrent_requests (`int`, *optional*, defaults to `4`):
                Maximum number of concurrent requests when using chunked dispatch.

        Returns:
            `dict` with keys:
                - `logprobs` (`list[list[list[float]]]`):
                    Per-token logprobs of shape (batch, completion_len, top_logprobs), sorted by descending
                    probability.
                - `logprob_token_ids` (`list[list[list[int]]]`):
                    Token IDs corresponding to each logprob, same shape as `logprobs`.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        url = f"{self.base_url}/get_sequence_logprobs"
        response_format = "binary" if use_binary else "json"

        if chunk_size > 0 and len(sequences) > chunk_size:
            # Chunked concurrent dispatch
            n = len(sequences)
            chunks = []
            for i in range(0, n, chunk_size):
                chunks.append((sequences[i : i + chunk_size], prompt_lengths[i : i + chunk_size]))

            responses = [None] * len(chunks)

            def _send_chunk(idx, seqs, plens):
                resp = self.session.post(
                    url,
                    json={
                        "sequences": seqs,
                        "prompt_lengths": plens,
                        "top_logprobs": top_logprobs,
                        "temperature": temperature,
                        "response_format": response_format,
                    },
                )
                if resp.status_code != 200:
                    raise Exception(f"Request failed: {resp.status_code}, {resp.text}")
                return idx, resp.json()

            with ThreadPoolExecutor(max_workers=min(max_concurrent_requests, len(chunks))) as executor:
                futures = {
                    executor.submit(_send_chunk, idx, seqs, plens): idx for idx, (seqs, plens) in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx, result = future.result()
                    responses[idx] = result

            # Merge results
            if use_binary:
                return self._merge_binary_responses(responses, top_logprobs)
            else:
                all_logprobs = []
                all_token_ids = []
                for resp in responses:
                    all_logprobs.extend(resp["logprobs"])
                    all_token_ids.extend(resp["logprob_token_ids"])
                return {"logprobs": all_logprobs, "logprob_token_ids": all_token_ids}
        else:
            # Single request
            response = self.session.post(
                url,
                json={
                    "sequences": sequences,
                    "prompt_lengths": prompt_lengths,
                    "top_logprobs": top_logprobs,
                    "temperature": temperature,
                    "response_format": response_format,
                },
            )
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

            json_response = response.json()
            if use_binary:
                return self._decode_binary_logprobs(json_response)
            else:
                return {
                    "logprobs": json_response["logprobs"],
                    "logprob_token_ids": json_response["logprob_token_ids"],
                }

    @staticmethod
    def _decode_binary_logprobs(response: dict) -> dict[str, list]:
        """Decode base64-encoded numpy arrays back to nested lists.

        Returns a dict with:
            ``logprobs`` / ``logprob_token_ids`` — teacher's sorted top-k logprobs and
                token IDs (shape per sequence: ``(comp_len, top_k)``). Used for the forward KL term.
            ``actual_logprobs`` / ``actual_token_ids`` — teacher logprob for the actual
                token at each position (shape per sequence: ``(comp_len, 1)``). Used for the reverse KL term.
        """
        import numpy as np

        shape = response["shape"]  # [batch, max_comp_len, top_k]
        comp_lengths = response["completion_lengths"]

        logprobs_arr = np.frombuffer(base64.b64decode(response["logprobs_b64"]), dtype=np.float32).reshape(shape)
        token_ids_arr = np.frombuffer(base64.b64decode(response["token_ids_b64"]), dtype=np.int32).reshape(shape)

        # Convert back to nested lists, trimming padding
        all_logprobs = []
        all_token_ids = []
        for i, comp_len in enumerate(comp_lengths):
            all_logprobs.append(logprobs_arr[i, :comp_len, :].tolist())
            all_token_ids.append(token_ids_arr[i, :comp_len, :].tolist())

        result = {"logprobs": all_logprobs, "logprob_token_ids": all_token_ids}

        # Decode actual-token logprobs (for reverse KL)
        if "actual_logprobs_b64" in response:
            actual_shape = [shape[0], shape[1], 1]
            actual_lp = np.frombuffer(base64.b64decode(response["actual_logprobs_b64"]), dtype=np.float32).reshape(
                actual_shape
            )
            actual_ids = np.frombuffer(base64.b64decode(response["actual_token_ids_b64"]), dtype=np.int32).reshape(
                actual_shape
            )
            all_actual_lps = []
            all_actual_ids = []
            for i, comp_len in enumerate(comp_lengths):
                all_actual_lps.append(actual_lp[i, :comp_len, :].tolist())
                all_actual_ids.append(actual_ids[i, :comp_len, :].tolist())
            result["actual_logprobs"] = all_actual_lps
            result["actual_token_ids"] = all_actual_ids

        return result

    @staticmethod
    def _merge_binary_responses(responses: list[dict], top_logprobs: int) -> dict[str, list]:
        """Merge binary responses from multiple chunks into a single result."""

        all_logprobs = []
        all_token_ids = []
        all_actual_lps = []
        all_actual_ids = []
        for resp in responses:
            decoded = VLLMClient._decode_binary_logprobs(resp)
            all_logprobs.extend(decoded["logprobs"])
            all_token_ids.extend(decoded["logprob_token_ids"])
            if "actual_logprobs" in decoded:
                all_actual_lps.extend(decoded["actual_logprobs"])
                all_actual_ids.extend(decoded["actual_token_ids"])

        result = {"logprobs": all_logprobs, "logprob_token_ids": all_token_ids}
        if all_actual_lps:
            if len(all_actual_lps) != len(all_logprobs):
                raise ValueError(
                    f"Inconsistent chunks: {len(all_actual_lps)} actual_logprobs entries "
                    f"but {len(all_logprobs)} logprobs entries."
                )
            result["actual_logprobs"] = all_actual_lps
            result["actual_token_ids"] = all_actual_ids
        return result

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"{self.base_url}/reset_prefix_cache"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"{self.base_url}/close_communicator"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

        if self.communicator is not None:
            self.communicator = None


# Example usage
if __name__ == "__main__":
    device = "xpu" if is_torch_xpu_available() else "cuda"
    client = VLLMClient()
    client.init_communicator(device=device)

    # Generate completions
    responses = client.generate(["Hello, AI!", "Tell me a joke"], n=4, max_tokens=32)
    print("Responses:", responses)  # noqa

    # Update model weights
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to(device)
    client.update_model_params(model)
