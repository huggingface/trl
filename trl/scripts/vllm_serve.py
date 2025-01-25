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


if __name__ == "__main__":
    # Create an instance of the server and run it
    server = VLLMServer(model_name="Qwen/Qwen2.5-0.5B", host="0.0.0.0", port=5000)
    server.run()
