# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest

import torch
from transformers import AutoTokenizer

from trl import DataCollatorForCompletionOnlyLM


class DataCollatorForCompletionOnlyLMTester(unittest.TestCase):
    def test_data_collator_finds_response_template_llama2_tokenizer(self):
        # this should ideally be tested with meta-llama/Llama-2-7b-hf
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.instruction = """### System: You are a helpful assistant.

### User: How much is 2+2?

### Assistant: 2+2 equals 4"""
        self.instruction_template = "\n### User:"
        self.response_template = "\n### Assistant:"

        # GPT2Tokenizer: [198, 21017, 11787, 25] -> [11787, 25]
        # Llama2Tokenizer: [29871, 13, 2277, 29937, 4911, 29901] -> [2277, 29937, 4911, 29901]
        self.tokenized_instruction_w_context = self.tokenizer.encode(
            self.instruction_template, add_special_tokens=False
        )[2:]

        # GPT2Tokenizer: [198, 21017, 15286, 25] -> [15286, 25]
        # Llama2Tokenizer: [29871, 13, 2277, 29937, 4007, 22137, 29901] -> [2277, 29937, 4007, 22137, 29901]
        self.tokenized_response_w_context = self.tokenizer.encode(self.response_template, add_special_tokens=False)[2:]

        # Plain check on string
        self.assertIn(self.response_template, self.instruction)
        self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)

        # Test the fix for #598
        # Pass already tokenized (w context) and truncated response_template so token_ids are like in the instruction + response
        self.collator = DataCollatorForCompletionOnlyLM(self.tokenized_response_w_context, tokenizer=self.tokenizer)
        self.collator.torch_call([self.tokenized_instruction])

        # Test for PR #749
        # Pass already tokenized (w context) instruction and response both so token_ids are like in the instruction + response
        self.collator = DataCollatorForCompletionOnlyLM(
            self.tokenized_response_w_context, self.tokenized_instruction_w_context, tokenizer=self.tokenizer
        )
        self.collator.torch_call([self.tokenized_instruction])

    def test_data_collator_handling_of_long_sequences(self):
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
        self.instruction = """### System: You are a helpful assistant.

### User: How much is 2+2? I'm asking because I'm not sure. And I'm not sure because I'm not good at math.
"""
        self.response_template = "\n### Assistant:"
        # check DataCollatorForCompletionOnlyLM using response template only
        self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)
        self.collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)
        encoded_instance = self.collator.torch_call([self.tokenized_instruction])
        result = torch.all(encoded_instance["labels"] == -100)
        self.assertTrue(result, "Not all values in the tensor are -100.")

        # check DataCollatorForCompletionOnlyLM using response template and instruction template
        self.instruction_template = "\n### User:"
        self.collator = DataCollatorForCompletionOnlyLM(
            self.response_template, self.instruction_template, tokenizer=self.tokenizer
        )
        encoded_instance = self.collator.torch_call([self.tokenized_instruction])
        result = torch.all(encoded_instance["labels"] == -100)
        self.assertTrue(result, "Not all values in the tensor are -100.")
