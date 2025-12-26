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

import textwrap

import pytest
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import clone_chat_template
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)

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


class TestAddResponseSchema:
    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0.dev0"),
        reason="Response parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    def test_add_response_schema(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        tokenizer = add_response_schema(tokenizer)
        assistant_text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
        parsed = tokenizer.parse_response(assistant_text)
        expected = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}],
        }
        assert parsed == expected


class TestIsChatTemplatePrefixPreserving:
    def test_prefix_preserving_template(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        tokenizer.chat_template = textwrap.dedent(r"""
        {%- for message in messages %}

        {%- if message.role == 'user' %}
            {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
        {%- elif message.role == 'assistant' %}
            {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
        {%- endif %}

        {%- endfor %}

        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
        {%- endif %}""")
        assert is_chat_template_prefix_preserving(tokenizer) is True

    def test_non_prefix_preserving_template(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        # The following template is quite typical of models like Qwen3 and GPT-OSS, where the thinking part is
        # only present for last assistant message, which makes it non-prefix-preserving.
        # docstyle-ignore
        tokenizer.chat_template = textwrap.dedent(r"""
        {%- if messages[0].role == 'system' %}
            {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
        {%- endif %}
        {%- set ns = namespace(last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if message.role == "user" and message.content is string %}
                {%- set ns.last_query_index = index %}
                {%- break %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- set content = message.content if message.content is string else '' %}
            {%- if message.role == "user" or (message.role == "system" and not loop.first) %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if loop.last or (not loop.last and reasoning_content) %}
                        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                    {%- else %}
                        {{- '<|im_start|>' + message.role + '\n' + content }}
                    {%- endif %}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}""")
        assert is_chat_template_prefix_preserving(tokenizer) is False


class TestGetTrainingChatTemplate:
    def test_qwen3(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        assert is_chat_template_prefix_preserving(tokenizer) is False
        tokenizer.chat_template = get_training_chat_template(tokenizer)
        assert is_chat_template_prefix_preserving(tokenizer) is True


@pytest.mark.xfail(
    condition=Version(transformers.__version__) < Version("5.0.0.dev0"),
    reason="Tool parsing is not supported in transformers versions below 5.0.0",
    strict=True,
)
class TestParseResponse:
    def test_parse_response(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        tokenizer = add_response_schema(tokenizer)
        # docstyle-ignore
        text = textwrap.dedent("""\
            <tool_call>
            {"name": "multiply", "arguments": {"a": 3, "b": 4}}
            </tool_call><|im_end|>""")
        assistant_text = tokenizer(text)["input_ids"]
        parsed = parse_response(tokenizer, assistant_text)
        expected = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}],
        }
        assert parsed == expected

    def test_parse_response_multiple_tool_calls(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        tokenizer = add_response_schema(tokenizer)
        # docstyle-ignore
        text = textwrap.dedent("""\
            <tool_call>
            {"name": "multiply", "arguments": {"a": 3, "b": 4}}
            </tool_call>
            <tool_call>
            {"name": "addition", "arguments": {"a": 3, "b": 4}}
            </tool_call><|im_end|>""")
        assistant_text = tokenizer(text)["input_ids"]
        parsed = parse_response(tokenizer, assistant_text)
        expected = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}},
                {"type": "function", "function": {"name": "addition", "arguments": {"a": 3, "b": 4}}},
            ],
        }
        assert parsed == expected

    def test_parse_response_no_tool_call(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        tokenizer = add_response_schema(tokenizer)
        text = "Here is the answer to your question.<|im_end|>"
        assistant_text = tokenizer(text)["input_ids"]
        parsed = parse_response(tokenizer, assistant_text)
        expected = {
            "role": "assistant",
            "content": "Here is the answer to your question.",
        }

        assert parsed == expected

    def test_parse_response_malformed_tool_call(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForSequenceClassification")
        tokenizer = add_response_schema(tokenizer)
        text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}\n</tool_call><|im_end|>'
        assistant_text = tokenizer(text)["input_ids"]
        parsed = parse_response(tokenizer, assistant_text)
        expected = {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}\n</tool_call>',
        }

        assert parsed == expected
