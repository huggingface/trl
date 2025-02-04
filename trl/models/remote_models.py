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
import requests
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


class RemoteModel:
    def __init__(self, remote_model_url):
        self.remote_model_url = remote_model_url
        # Check if the remote server is healthy
        health_check_url = f"{self.remote_model_url}/health"
        response = requests.get(health_check_url)
        if response.status_code != 200:
            raise Exception(f"Server health check failed: {response.text}")

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int
    ) -> CausalLMOutputWithPast:
        """
        Sends a request to the remote server to perform a forward pass.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            logits_to_keep (int): The number of logits to keep.

        Returns:
            CausalLMOutputWithPast: Contains only the logits.
        """
        # Convert tensors to lists for JSON serialization
        device = input_ids.device
        input_ids_list = input_ids.tolist()
        attention_mask_list = attention_mask.tolist()

        # Prepare the request body
        request_body = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "logits_to_keep": logits_to_keep,
        }

        # Send the POST request to the server
        # add a few retries?
        response = requests.post(f"{self.remote_model_url}/forward", json=request_body)

        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Error from server: {response}")

        # Parse the response
        response_json = response.json()
        logits_list = response_json["logits"]

        # Convert the logits back to a tensor
        logits = torch.tensor(logits_list).to(device)

        return CausalLMOutputWithPast(logits=logits)
