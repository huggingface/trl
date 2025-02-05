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


class RemoteSDLangModel:
    def __init__(self, remote_model_url, stop_token=None):
        self.remote_model_url = remote_model_url
        self.stop_token = stop_token
        # Check if the remote server is healthy
        health_check_url = f"{self.remote_model_url}/health"
        response = requests.get(health_check_url)
        if response.status_code != 200:
            raise Exception(f"Server health check failed: {response.text}")

    @staticmethod
    def get_logits(response_json):
        input_logits = [prob[0] for prob in response_json["meta_info"]["input_token_logprobs"]]
        output_logits = [prob[0] for prob in response_json["meta_info"]["output_token_logprobs"]]
        
        # we exclude the first entry as that corresponds to the BOS, which is None
        return input_logits[1:] + output_logits

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
        # TODO: we should strip and put back the attn mask

        # Prepare the request body
        request_body = {
                "input_ids": input_ids_list,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    #"stop_token_ids": [tokenizer.eos_token_id],
                },
                "stream": False,
                "return_logprob": True,
                "logprob_start_len": 0,
        }

        # Send the POST request to the server
        # add a few retries?
        response = requests.post(f"{self.remote_model_url}/generate", json=request_body)
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"Error from server: {response}")

        # Parse the response
        response_json = response.json()

        # Convert the logits back to a tensor
        logits_list = self.get_logits(response_json)
        logits = torch.tensor(logits_list).to(device)

        return CausalLMOutputWithPast(logits=logits)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import requests 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    input_text = "What is the capital of France?"

    input_tokens = tokenizer.encode(input_text)
    print(f"Input Text: {input_text}")
    print(f"Tokenized Input: {input_tokens}")
    remote_model = RemoteSDLangModel("http://localhost:30010")
    
    input_tokens = torch.LongTensor(input_tokens)
    
    
    result = remote_model(input_tokens, input_tokens, 1)
    print(result)