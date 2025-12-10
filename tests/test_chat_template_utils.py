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
from transformers import AutoTokenizer

from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)


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
        text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
        assistant_text = tokenizer(text)["input_ids"]
        parsed = parse_response(tokenizer, assistant_text)
        expected = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}],
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
