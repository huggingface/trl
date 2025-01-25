# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import io

import requests
import torch


class VLLMClient:
    """
    Client for interacting with a VLLM server.

    Args:
        url (str): The URL of the VLLM server.

    Run the server with:
    ```bash
    trl vllm-serve
    ```

    Example:
    ```python
    >>> client = VLLMClient()
    >>> data = {"prompts": ["The closest planet to the Sun is", "The capital of France is"]}
    >>> response = client.generate(data["prompts"])
    >>> print(response)
    {'completions': [' Mercury.', ' Paris.']}
    ```
    """

    def __init__(self, url="http://127.0.0.1:5000"):
        self.url = url
        self.buffer = io.BytesIO()

    def load(self, model_name: str) -> None:
        requests.post(self.url + "/load", json={"model_name": model_name})

    def generate(self, prompts: list[str]) -> dict[str, list[str]]:
        data = {"prompts": prompts}
        response = requests.post(self.url + "/generate", json=data)
        return response.json()

    def chat(self, prompts: list[list[dict[str, str]]]) -> dict[str, list[str]]:
        data = {"prompts": prompts}
        response = requests.post(self.url + "/chat", json=data)
        return response.json()

    def load_weights(self, state_dict) -> None:
        torch.save(state_dict, self.buffer)
        self.buffer.seek(0)
        requests.post(self.url + "/load_weights", data=self.buffer.read())
