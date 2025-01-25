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

import torch
from flask import Flask, jsonify, request
from vllm import LLM


def vllm_serve():
    llm = LLM(model="Qwen/Qwen2.5-0.5B")

    # Create Flask app
    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        """
        Endpoint to generate completions from prompts.

        Example:
        ```python
        >>> import requests
        >>> url = "http://127.0.0.1:5000/generate"
        >>> data = {"prompts": [
        ...     "The closest planet to the Sun is",
        ...     "The capital of France is"
        ... ]}
        >>> response = requests.post(url, json=data)
        >>> print(response.json())
        {'completions': ['Mercury.', 'Paris.']}
        ```
        """

        try:
            # Parse input JSON data
            data = request.get_json()
            prompts = data["prompts"]  # Expecting a key "inputs" containing a list of inputs

            # Perform inference
            outputs = llm.generate(prompts)
            completions = [output.outputs[0].text for output in outputs]

            return jsonify({"completions": completions})

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/chat", methods=["POST"])
    def chat():
        """
        Endpoint to chat with the model.

        Example:
        ```python
        >>> import requests
        >>> url = "http://127.0.0.1:5000/chat"
        >>> data = {"prompts": [
        ...     [{"role": "user", "content": "What is the capital of France?"}],
        ...     [{"role": "user", "content": "What is the capital of Italy?"}]
        ... ]}
        >>> response = requests.post(url, json=data)
        >>> print(response.json())
        {'completions': ['The capital of France is Paris.', 'The capital of Italy is Rome.']}
        ```
        """
        try:
            # Parse input JSON data
            data = request.get_json()
            prompts = data["prompts"]  # Expecting a key "inputs" containing a list of inputs

            # Perform inference
            outputs = llm.chat(prompts)
            completions = [output.outputs[0].text for output in outputs]

            return jsonify({"completions": completions})

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/load_weights", methods=["POST"])
    def load_weights():
        """
        Endpoint to dynamically update model weights.
        Expects a POST request with a serialized state_dict.

        Example:
        ```python
        >>> state_dict = model.state_dict()
        >>> buffer = io.BytesIO()
        >>> torch.save(state_dict, buffer)
        >>> buffer.seek(0)
        >>> url = "http://127.0.0.1:5000/load_weights"
        >>> response = requests.post(url, data=buffer.read())
        >>> print(response.json())
        {'status': 'success', 'message': 'Model weights loaded.'}
        ```
        """
        try:
            # Parse binary data (weights file sent as bytes)
            weights_data = request.data
            buffer = io.BytesIO(weights_data)

            # Load the state_dict from the buffer
            state_dict = torch.load(buffer, weights_only=True).items()

            # Update the model's weights
            llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict)

            # Return success response
            return jsonify({"status": "success", "message": "Model weights loaded."})

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    app.run(host="0.0.0.0", port=5000)  # Run the server on all network interfaces, port 5000


if __name__ == "__main__":
    vllm_serve()
