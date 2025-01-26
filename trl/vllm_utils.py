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

from .import_utils import is_flask_available, is_vllm_available


if is_flask_available():
    from flask import Flask, jsonify, request

if is_vllm_available():
    from vllm import LLM


class VLLMServer:
    """
    A vLLM server that exposes a REST API for generating completions and chatting with a vLLM model.

    Make sure to install the `vllm` and `flask` packages before using this class.Just run the following command:

    ```bash
    pip install vllm flask
    # or
    pip install trl[vllm]
    ```

    The server provides the following endpoints:
    - `/load`: Load a model (POST). Expects a JSON payload with the model name.
    - `/generate`: Generate completions from prompts (POST). Expects a JSON payload with a list of prompts.
    - `/chat`: Chat with the model (POST). Expects a JSON payload with a list of conversation turns.
    - `/load_weights`: Load model weights (POST). Allows dynamic weight updates.

    Args:
        model_name (`str` or `None`, *optional*, default to `None`):
            Name of the model to load. If not provided, the server starts without a model loaded. Models can
            be loaded later using the `/load` endpoint.
        host (`str`, *optional*, default to `"0.0.0.0"`):
            Host address to run the Flask server.
        port (`int`, *optional*, default to `5000`):
            Port to run the Flask server.

    Example:

    Run the server with a model loaded:

    ```python
    >>> from trl import VLLMServer
    >>> server = VLLMServer(model_name="Qwen/Qwen2.5-7B-Instruct")
    >>> server.run()
    ```

    Use the server to generate completions:
    ```shell
    $ curl -X POST "http://0.0.0.0:5000/generate" -H "Content-Type: application/json" -d '{"prompts": ["The closest planet to the Sun is"]}'
    {"completions":[" ____\nA. Sun\nB. Mercury\nC. Mars\nAnswer:\n\n"]}
    ```
    """

    def __init__(self, model_name=None, host: str = "0.0.0.0", port: int = 5000):
        if not is_flask_available():
            raise ImportError("vLLM server requires Flask. Please install it with `pip install flask`.")

        if not is_vllm_available():
            raise ImportError("vLLM server requires the `vllm` package. Please install it with `pip install vllm`.")

        self.host = host
        self.port = port
        self.app = Flask(__name__)  # Initialize Flask app

        self._add_routes()  # Add routes to the app
        if model_name is not None:
            self.llm = LLM(model=model_name)
        else:
            self.llm = None

    def _add_routes(self):
        """Add the Flask routes for the server."""

        @self.app.route("/load", methods=["POST"])
        def load():
            try:
                data = request.get_json()
                self.llm = LLM(model=data["model_name"])
                self.app.logger.info(f"Model {data['model_name']} loaded.")
                return jsonify({"status": "success", "message": "Model loaded."})

            except Exception as e:
                self.app.logger.error(f"Error loading model: {str(e)}")
                return jsonify({"error": str(e)}), 400

        @self.app.route("/generate", methods=["POST"])
        def generate():
            try:
                # Parse input JSON data
                data = request.get_json()
                prompts = data["prompts"]  # Expecting a key "prompts" containing a list of inputs

                # Perform inference
                outputs = self.llm.generate(prompts)
                completions = [output.outputs[0].text for output in outputs]

                return jsonify({"completions": completions})

            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @self.app.route("/chat", methods=["POST"])
        def chat():
            try:
                # Parse input JSON data
                data = request.get_json()
                prompts = data["prompts"]  # Expecting a key "prompts" containing a list of inputs

                # Perform inference
                outputs = self.llm.chat(prompts)
                completions = [output.outputs[0].text for output in outputs]

                return jsonify({"completions": completions})

            except Exception as e:
                return jsonify({"error": str(e)}), 400

        @self.app.route("/load_weights", methods=["POST"])
        def load_weights():
            try:
                # Parse binary data (weights file sent as bytes)
                weights_data = request.data
                buffer = io.BytesIO(weights_data)

                # Load the state_dict from the buffer
                state_dict = torch.load(buffer, weights_only=True).items()

                # Update the model's weights
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict)

                return jsonify({"status": "success", "message": "Model weights loaded."})

            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400

    def run(self):
        """Run the Flask server."""
        self.app.run(host=self.host, port=self.port)


class VLLMClient:
    """
    A Python client for interacting with the vLLM server.

    This client allows you to communicate with a running instance of the [`VLLMServer`] to load models,
    generate completions, chat with the model, and dynamically load model weights.

    Args:
        host (`str`, *optional*, default to `"0.0.0.0"`):
            Host address of the vLLM server.
        port (`int`, *optional*, default to `5000`):
            Port of the vLLM server.

    Example:
    ```python
    >>> from trl import VLLMClient
    >>> client = VLLMClient()
    >>> client.load("Qwen/Qwen2.5-7B-Instruct")
    >>> response = client.generate(prompts=["The capital of France is"])
    >>> print(response["completions"])
    [' Paris and the area of France is 643,801 km']
    ```
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.url = f"http://{host}:{port}"
        self.buffer = io.BytesIO()

    def load(self, model_name: str) -> None:
        """
        Load a model on the server.

        Args:
            model_name (`str`):
                Name of the model to load.
        """
        requests.post(self.url + "/load", json={"model_name": model_name})

    def generate(self, prompts: list[str]) -> dict[str, list[str]]:
        """
        Generate completions for a list of prompts.

        Args:
            prompts (`list[str]`):
                List of prompts to generate completions for.

        Returns:
            `dict[str, list[str]]`:
                A dictionary with a key `"completions"` containing the list of generated outputs.
        """
        data = {"prompts": prompts}
        response = requests.post(self.url + "/generate", json=data)
        return response.json()

    def chat(self, prompts: list[list[dict[str, str]]]) -> dict[str, list[str]]:
        """
        Chat with the model using a list of conversation turns.

        Args:
            prompts (`list[list[dict[str, str]]]`):
                List of conversations. Each conversation should be a list of dictionaries with keys like `"role"`
                (e.g., `"user"` or `"assistant"`) and `"content"`.

        Returns:
            `dict[str, list[str]]`:
                Dictionary with a key `"completions"` containing the list of chat outputs.
        """
        data = {"prompts": prompts}
        response = requests.post(self.url + "/chat", json=data)
        return response.json()

    def load_weights(self, state_dict) -> None:
        """
        Dynamically load weights to the model on the server.

        Args:
            state_dict:
                PyTorch state dictionary containing the model's weights. The state dictionary must be compatible with
                the model currently loaded on the server.
        """
        torch.save(state_dict, self.buffer)
        self.buffer.seek(0)
        requests.post(self.url + "/load_weights", data=self.buffer.read())
