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

import functools
from typing import TypeVar

from transformers import PreTrainedTokenizer, ProcessorMixin


# These schemas are copy-pasted from https://github.com/huggingface/transformers/blob/main/tests/utils/test_chat_parsing_utils.py
cohere_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": r"<\|START_RESPONSE\|>(.*?)(?:<\|END_RESPONSE\|>|$)"},
        "thinking": {"type": "string", "x-regex": r"<\|START_THINKING\|>(.*?)(?:<\|END_THINKING\|>|$)"},
        "tool_calls": {
            "x-regex": r"<\|START_ACTION\|>(.*?)(?:<\|END_ACTION\|>|$)",
            "x-parser": "json",
            "x-parser-args": {
                "transform": "[*].{type: 'function', function: {name: tool_name, arguments: parameters}}"
            },
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

ernie_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": "<response>\n(.*?)\n?</response>"},
        "thinking": {"type": "string", "x-regex": r"(?:^|<think>\s*)(.*?)\s*<\/think>"},
        "tool_calls": {
            "x-regex-iterator": "<tool_call>(.*?)</tool_call>",
            "type": "array",
            "items": {
                "type": "object",
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

gpt_oss_schema = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string", "x-regex": r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)"},
        "thinking": {"type": "string", "x-regex": r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"},
        "tool_calls": {
            "x-regex-iterator": r"<\|channel\|>commentary (to=functions\..*?<\|message\|>.*?)(?:<\|call\|>|$)",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"^to=functions\.(\w+)"},
                            "arguments": {
                                "type": "object",
                                "x-regex": r"<\|message\|>(.*)",
                                "x-parser": "json",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

smollm_schema = {
    "x-regex": r"(?:<think>\n?(?P<thinking>.+?)\n?</think>)?\s*(?:<tool_call>(?P<tool_calls>.+?)</tool_call>)?\s*(?P<content>.+?)?\s*(?:<\|im_end\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "thinking": {"type": "string"},
        "tool_calls": {
            "x-parser": "json",
            "x-parser-args": {"transform": "[{type: 'function', function: @}]"},
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}


qwen3_schema = {
    "x-regex": r"^(?:<think>\n?(?P<thinking>.+?)\n?</think>\s*)?(?P<content>.*?)(?=(?:<tool_call>|<\|im_end\|>|$))(?:<tool_call>(?P<tool_calls>.+?)</tool_call>)?\s*(?:<\|im_end\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "thinking": {"type": "string"},
        "tool_calls": {
            "x-parser": "json",
            "x-parser-args": {"transform": "[{type: 'function', function: @}]"},
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                    },
                },
            },
        },
    },
}

# docstyle-ignore
qwen3_chat_template = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
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
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""

TokenizerOrProcessor = TypeVar("TokenizerOrProcessor", PreTrainedTokenizer, ProcessorMixin)


def add_response_schema(processor: TokenizerOrProcessor) -> TokenizerOrProcessor:
    """
    Adds the appropriate response schema to the given tokenizer or processor based on its chat template.

    At the time of initial implementation, most tokenizers do not have built-in support for response schemas. While
    waiting for broader adoption, we provide this utility function to manually set the response schema for known chat
    templates.

    Args:
        processor (`TokenizerOrProcessor`):
            Tokenizer or processor to which the response schema will be added.

    Returns:
        `TokenizerOrProcessor`:
            Tokenizer or processor with the added response schema.
    """
    if processor.chat_template == qwen3_chat_template:
        # The qwen3 response schema seems to be smollm_schema, and not the qwen3_schema. See
        # https://github.com/huggingface/transformers/issues/42220
        processor.response_schema = qwen3_schema
        return processor
    raise ValueError(
        "Unrecognized chat template, failed to add response schema. Please manually set the response schema on the "
        "tokenizer or processor."
    )


def is_chat_template_prefix_preserving(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Check whether the chat template preserves prefixes when applied.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer instance to check.

    Returns:
        `bool`:
            `True` if the chat template preserves prefixes, `False` otherwise.
    """
    messages1 = [
        {"role": "user", "content": "What color is the sky?"},
    ]
    messages2 = [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "It is blue."},
    ]
    messages3 = [
        {"role": "user", "content": "What color is the sky?"},
        {"role": "assistant", "content": "It is blue."},
        {"role": "user", "content": "And at night?"},
    ]

    text1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
    text2 = tokenizer.apply_chat_template(messages2, tokenize=False)
    text3 = tokenizer.apply_chat_template(messages3, tokenize=False)

    return text2.startswith(text1) and text3.startswith(text2)


# docstyle-ignore
qwen3_training_chat_template = r"""{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
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
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


def patch_chat_template_for_training(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Wrap `tokenizer.apply_chat_template()` to use a training-compatible chat template if needed.

    During training, we need a *prefix-preserving* template where each message strictly appends to previous ones. For
    example:

    ```python
    turn0 = {"role": "user", "content": "Hello!"}
    turn1 = {"role": "assistant", "content": "Hi!"}
    text0 = tokenizer.apply_chat_template([turn0], add_generation_prompt=True)
    text1 = tokenizer.apply_chat_template([turn0, turn1])
    assert text1.startswith(text0)
    ```

    Tokenizers typically use inference-ready templates that may differ from the template used in training. The
    inference template may not satisfy the prefix-preservation requirement. For example, Qwen3 and OpenAI GPT OSS drop
    thinking blocks from non-final turns.

    This function first checks if the template is prefix-preserving. If it is, no patching is needed. If not, it
    patches the `apply_chat_template()` method to temporarily swap in a training-compatible template during calls, then
    restore the original afterward. This ensures the chat template complies with training needs while preserving the
    original template for later inference. It also stores the training template in a `_training_chat_template`
    attribute, which is useful when you need to access it—for example, when using vLLM inference.

    Currently supported: Qwen3 models only.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer instance to patch.

    Returns:
        `PreTrainedTokenizer`:
            The same tokenizer with `apply_chat_template()` patched (if needed and supported).
    """
    # First check if patching is needed
    if is_chat_template_prefix_preserving(tokenizer):
        return tokenizer  # No patching needed

    original_method = tokenizer.apply_chat_template
    original_chat_template = tokenizer.chat_template

    if tokenizer.chat_template == qwen3_chat_template:
        tokenizer._training_chat_template = qwen3_training_chat_template
    else:
        raise ValueError(
            "The tokenizer's chat template is not prefix-preserving and patching is not supported for this template. "
            "Please manually modify the tokenizer's chat template for training."
        )

    @functools.wraps(original_method)
    def wrapper(*args, **kwargs):
        tokenizer.chat_template = tokenizer._training_chat_template
        try:
            result = original_method(*args, **kwargs)
        finally:
            tokenizer.chat_template = original_chat_template
        return result

    tokenizer.apply_chat_template = wrapper
    return tokenizer


def parse_response(tokenizer: PreTrainedTokenizer, ids: list[list[int]]) -> list[dict]:
    """
    Parse token sequences into structured response dictionaries with fallback handling.

    Attempts to parse each sequence using `tokenizer.parse_response()`. If parsing fails (e.g., due to malformed tool
    calls like `<tool_call>{"type":"function"</tool_call>`), falls back to decoding as plain text.

    Also removes incorrectly appended EOS tokens from tool call content when present.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer with a `parse_response()` method.
        ids (`list[list[int]]`):
            List of token sequences.

    Returns:
        `list[dict]`:
            List of response dictionaries.
    """

    outputs = []
    for seq in ids:
        try:
            parsed = tokenizer.parse_response(seq)
            # Hotfix: remove incorrectly appended EOS token from tool calls
            # See https://github.com/huggingface/transformers/issues/42249
            parsed["content"] = parsed["content"].removesuffix(tokenizer.eos_token)
        except Exception:
            # Fallback: decode as plain text if parsing fails. This happens if the model outputs malformed tool calls.
            content = tokenizer.decode(seq, skip_special_tokens=True)
            parsed = {"role": "assistant", "content": content}
        outputs.append(parsed)
    return outputs
