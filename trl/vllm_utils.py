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
from flask import Flask, jsonify, request
from vllm import LLM


class VLLMServer:
    def __init__(self, model_name=None, host="0.0.0.0", port=5000):
        """
        Initialize the VLLM server.

        Args:
            model_name (str): Name of the model to load.
            host (str): Host address to run the Flask server.
            port (int): Port to run the Flask server.
        """
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
    Client for interacting with a vLLM server. A server launched with [`]

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
