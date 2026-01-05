# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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


from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.utils import DataCollatorForChatML

from ..testing_utils import TrlTestCase


class TestDataCollatorForChatML(TrlTestCase):
    def setup_method(self):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        # Token ID for "true", the last assistant's response in the example:
        self.ignore_index = -100
        self.max_length = 1024
        self.messages_key = "messages"

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset.to_list()

        # Initialize the data collator
        self.collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

    def test_data_collator_for_chatml(self):
        # Process the data
        data = self.collator(self.examples)

        # Verify basic shapes and types
        assert "input_ids" in data
        assert "attention_mask" in data
        assert "labels" in data
        assert "prompts" in data
        assert "prompt_attention_mask" in data

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Get the last assistant's response for comparison
        last_message = self.examples[0][self.messages_key][-1]
        assert last_message["role"] == "assistant", "Last message should be from assistant"
        last_assistant_response = last_message["content"]

        # Verify that input_ids contain both prompt and response
        decoded_input = self.tokenizer.decode(input_ids)
        assert last_assistant_response in decoded_input, "Input should contain assistant's response"

        # Verify that prompts only contain the conversation up to the last response
        decoded_prompt = self.tokenizer.decode(prompt_only)
        assert last_assistant_response not in decoded_prompt, "Prompt should not contain assistant's response"

        # Verify labels are -100 for non-assistant parts
        prompt_length = len(prompt_only)
        assert all(label == self.ignore_index for label in labels[:prompt_length]), (
            "Labels should be ignore_index for prompt tokens"
        )

        # Verify labels match assistant response after prompt
        # Add a filter to remove any trailing tokens after the first <|im_end|>
        last_assistant_response_with_end = last_assistant_response + self.tokenizer.eos_token
        last_assistant_response_tokens = self.tokenizer.encode(
            last_assistant_response_with_end, add_special_tokens=False
        )

        response_labels = []
        for label in labels[prompt_length:]:
            if label == self.ignore_index:
                continue
            response_labels.append(label)
            if label == self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
        assert response_labels == last_assistant_response_tokens, "Labels should match assistant response tokens"

        # Verify there isn't a generation prompt at the end
        generation_prompt = "<|im_start|>assistant"
        assert not decoded_input.strip().endswith(generation_prompt), (
            f"Input should not end with generation prompt '{generation_prompt}'"
        )

        assert response_labels == last_assistant_response_tokens, "Labels should match assistant response tokens"
