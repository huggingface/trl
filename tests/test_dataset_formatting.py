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


from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl.models.utils import clone_chat_template

from .testing_utils import TrlTestCase


class TestCloneChatTemplate(TrlTestCase):
    def test_clone(self):
        # This tokenizer doesn't have a chat_template by default
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        # This one has a chat_template by default
        source = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        _, modified_tokenizer, _ = clone_chat_template(model, tokenizer, source)

        # Check if special tokens are correctly set
        assert modified_tokenizer.eos_token == "<|im_end|>"

    def test_clone_with_resize(self):
        # This tokenizer doesn't have a chat_template by default
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        # This one has a chat_template by default
        source = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        modified_model, modified_tokenizer, _ = clone_chat_template(
            model, tokenizer, source, resize_to_multiple_of=123
        )

        # Check that the input embeddings have been resized to a multiple of 123
        assert (modified_model.vocab_size % 123) == 0
        # Check that the input embeddings size matches the tokenizer vocabulary size
        assert model.vocab_size == len(modified_tokenizer.vocab)

    def test_clone_with_resize_and_extra_tokens_already_in_vocab(self):
        # This tokenizer doesn't have a chat_template by default
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        # This one has a chat_template by default
        source = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        # This will add <extra_id_0>, <extra_id_1>, ... to the tokenizer
        modified_model, modified_tokenizer, _ = clone_chat_template(
            model, tokenizer, source, resize_to_multiple_of=123
        )
        # Try if we can resize a tokenizer that already has extra these extra tokens
        modified_model, modified_tokenizer, _ = clone_chat_template(
            modified_model, modified_tokenizer, source, resize_to_multiple_of=124
        )

        # Check that the input embeddings have been resized to a multiple of 123
        assert (modified_model.vocab_size % 124) == 0
        # Check that the input embeddings size matches the tokenizer vocabulary size
        assert model.vocab_size == len(modified_tokenizer.vocab)

    def test_apply_new_chat_template(self):
        # This tokenizer doesn't have a chat_template by default
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-BloomForCausalLM")
        # This one has a chat_template by default
        source = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        _, modified_tokenizer, _ = clone_chat_template(model, tokenizer, source)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help you?"},
        ]
        prompt = modified_tokenizer.apply_chat_template(messages, tokenize=False)

        assert (
            prompt
            == "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHi, how can I help you?<|im_end|>\n"
        )

    def test_clone_with_sequence_classification_model(self):
        # This tokenizer doesn't have a chat_template by default
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptNeoXForSequenceClassification")
        model = AutoModelForSequenceClassification.from_pretrained(
            "trl-internal-testing/tiny-GptNeoXForSequenceClassification"
        )
        # This one has a chat_template by default
        source = "trl-internal-testing/tiny-Qwen3ForCausalLM"
        _, modified_tokenizer, _ = clone_chat_template(model, tokenizer, source)

        # Check if special tokens are correctly set
        assert modified_tokenizer.eos_token == "<|im_end|>"
