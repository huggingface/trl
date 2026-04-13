# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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
    supports_tool_calling,
)

from .testing_utils import TrlTestCase, require_jmespath


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


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        pytest.param("trl-internal-testing/tiny-Glm4MoeForCausalLM", id="glm4moe"),
        pytest.param("trl-internal-testing/tiny-GptOssForCausalLM", id="gptoss"),
        pytest.param("trl-internal-testing/tiny-Qwen3MoeForCausalLM", id="qwen3"),
        pytest.param("trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", id="qwen3_vl"),
        pytest.param("trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration", id="qwen35"),
    ],
)
@pytest.mark.xfail(
    condition=Version(transformers.__version__) < Version("5.0.0"),
    reason="Response parsing is not supported in transformers versions below 5.0.0",
    strict=True,
)
@require_jmespath
class TestAddResponseSchema:
    def test_add_response_schema(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer = add_response_schema(tokenizer)
        messages = [
            {"role": "user", "content": "What is 3*4?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}],
            },
        ]
        prefix = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        response = text[len(prefix) :]
        # Here, we just test that the parsing doesn't raise an error.
        # The correctness of the parsing is tested in TestParseResponse
        tokenizer.parse_response(response)


class TestSupportsToolCalling:
    @pytest.mark.parametrize(
        "model_id",
        [
            pytest.param(
                "trl-internal-testing/tiny-Glm4MoeForCausalLM",
                id="glm4moe",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="GLM4 tokenizer requires transformers>=5.0.0",
                ),
            ),
            pytest.param("trl-internal-testing/tiny-GptOssForCausalLM", id="gptoss"),
            pytest.param("trl-internal-testing/tiny-LlamaForCausalLM-3", id="llama3"),
            pytest.param("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", id="qwen2.5"),
            pytest.param("trl-internal-testing/tiny-Qwen3ForCausalLM", id="qwen3"),
            pytest.param("trl-internal-testing/tiny-Qwen3MoeForCausalLM", id="qwen3moe"),
            pytest.param(
                "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration",
                id="qwen35",
                marks=pytest.mark.skipif(
                    Version(transformers.__version__) < Version("5.0.0"),
                    reason="Qwen3.5 tokenizer requires transformers>=5.0.0",
                ),
            ),
        ],
    )
    def test_supports_tool_calling(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        assert supports_tool_calling(tokenizer) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            # No chat template
            pytest.param("trl-internal-testing/tiny-BloomForCausalLM", id="bloom"),
            pytest.param("trl-internal-testing/tiny-GPT2LMHeadModel", id="gpt2"),
            pytest.param("trl-internal-testing/tiny-GPTNeoXForCausalLM", id="gptneox"),
            pytest.param("trl-internal-testing/tiny-OPTForCausalLM", id="opt"),
            # TemplateError: rejects tool role sequence
            pytest.param("trl-internal-testing/tiny-CohereForCausalLM", id="cohere"),
            pytest.param("trl-internal-testing/tiny-FalconMambaForCausalLM", id="falconmamba"),
            pytest.param("trl-internal-testing/tiny-GemmaForCausalLM", id="gemma"),
            pytest.param("trl-internal-testing/tiny-Gemma2ForCausalLM", id="gemma2"),
            # Silently ignores tool messages
            pytest.param("trl-internal-testing/tiny-Cohere2ForCausalLM", id="cohere2"),
            pytest.param("trl-internal-testing/tiny-Phi3ForCausalLM", id="phi3"),
        ],
    )
    def test_does_not_support_tool_calling(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        assert supports_tool_calling(tokenizer) is False

    @pytest.mark.parametrize(
        "model_id",
        [
            # TypeError: template concatenates arguments as string (needs template patch)
            pytest.param("trl-internal-testing/tiny-DeepseekV3ForCausalLM", id="deepseekv3"),
            pytest.param("trl-internal-testing/tiny-DeepseekV3ForCausalLM-0528", id="deepseekv3-0528"),
        ],
    )
    @pytest.mark.xfail(reason="DeepseekV3 template expects arguments as JSON string, needs patch", strict=True)
    def test_deepseek_tool_calling(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        assert supports_tool_calling(tokenizer) is True


class TestIsChatTemplatePrefixPreserving:
    def test_prefix_preserving_template(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForCausalLM")
        # docstyle-ignore
        tokenizer.chat_template = textwrap.dedent(r"""
        {%- for message in messages %}

        {%- if message.role == 'user' %}
            {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
        {%- elif message.role == 'assistant' %}
            {{- '<|im_start|>assistant\n' + message.content }}
            {%- if message.tool_calls %}
                {%- for tool_call in message.tool_calls %}
                    {%- if tool_call.function %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {{- '<tool_call>' + tool_call.name + '</tool_call>' }}
                {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
        {%- elif message.role == 'tool' %}
            {{- '<|im_start|>tool\n' + message.content + '<|im_end|>\n' }}
        {%- endif %}

        {%- endfor %}

        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
        {%- endif %}""")
        assert is_chat_template_prefix_preserving(tokenizer) is True

    def test_non_prefix_preserving_template(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForCausalLM")
        # The following template is quite typical of models like Qwen3 and GPT-OSS, where the thinking part (even
        # empty) is only present for last assistant message, which makes it non-prefix-preserving: appending a tool
        # message changes the earlier output.
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
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>' + tool_call.name + '</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {{- '<|im_start|>tool\n' + content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}""")
        assert is_chat_template_prefix_preserving(tokenizer) is False


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        pytest.param("trl-internal-testing/tiny-GptOssForCausalLM", id="gptoss"),
        pytest.param("trl-internal-testing/tiny-LlamaForCausalLM-3", id="llama3"),
        pytest.param("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", id="qwen2.5"),
        pytest.param("trl-internal-testing/tiny-Qwen3MoeForCausalLM", id="qwen3"),
    ],
)
class TestGetTrainingChatTemplate:
    def test_new_chat_template_is_prefix_preserving(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.chat_template = get_training_chat_template(tokenizer)
        assert is_chat_template_prefix_preserving(tokenizer) is True

    def test_behavior_unchanged_single_user_no_generation_prompt(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [{"role": "user", "content": "What color is the sky?"}]
        before = tokenizer.apply_chat_template(messages, tokenize=False)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_single_user_with_generation_prompt(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [{"role": "user", "content": "What color is the sky?"}]
        before = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=new_chat_template,
        )
        assert before == after

    def test_behavior_unchanged_single_user_and_final_assistant_plain_content(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ]
        before = tokenizer.apply_chat_template(messages, tokenize=False)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_final_assistant_with_reasoning_content(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {
                "role": "assistant",
                "content": "It is blue.",
                "reasoning_content": "The sky appears blue due to Rayleigh scattering.",
            },
        ]
        before = tokenizer.apply_chat_template(messages, tokenize=False)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_final_assistant_with_existing_think_tags(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {
                "role": "assistant",
                "content": "<think>\nThe sky scatters shorter wavelengths.\n</think>\n\nIt is blue.",
            },
        ]
        before = tokenizer.apply_chat_template(messages, tokenize=False)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_assistant_with_tool_calls(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [
            {"role": "user", "content": "Multiply 3 by 4."},
            {
                "role": "assistant",
                "content": "I will call a tool.",
                "tool_calls": [{"name": "multiply", "arguments": {"a": 3, "b": 4}}],
            },
        ]
        before = tokenizer.apply_chat_template(messages, tokenize=False)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_with_tools_with_and_without_system_message(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "Multiply 3 by 4."}]
        before = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_with_tools_with_system_message(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Multiply 3 by 4."},
        ]
        before = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools)
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, chat_template=new_chat_template)
        assert before == after

    def test_behavior_unchanged_generation_prompt_with_enable_thinking_false(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [{"role": "user", "content": "What color is the sky?"}]
        before = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        new_chat_template = get_training_chat_template(tokenizer)
        after = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            chat_template=new_chat_template,
        )
        assert before == after

    def test_assistant_masks(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ]
        chat_template = get_training_chat_template(tokenizer)
        result = tokenizer.apply_chat_template(
            messages, chat_template=chat_template, return_assistant_tokens_mask=True, return_dict=True
        )
        masks = result["assistant_masks"]
        assert 1 in masks
        # The first tokens (user turn) should not be masked
        assert masks[0] == 0
        # The last tokens (assistant turn ending with <|im_end|>) should be masked
        assert masks[-1] == 1

    def test_assistant_masks_multi_turn(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]
        chat_template = get_training_chat_template(tokenizer)
        result = tokenizer.apply_chat_template(
            messages, chat_template=chat_template, return_assistant_tokens_mask=True, return_dict=True
        )
        masks = result["assistant_masks"]
        # Should have two masked regions (two assistant turns): 0→1, 1→0, 0→1
        transitions = sum(1 for i in range(1, len(masks)) if masks[i] != masks[i - 1])
        assert transitions == 3


@pytest.mark.parametrize(
    "tokenizer_name",
    [
        pytest.param("trl-internal-testing/tiny-Glm4MoeForCausalLM", id="glm4moe"),
        pytest.param("trl-internal-testing/tiny-GptOssForCausalLM", id="gptoss"),
        pytest.param("trl-internal-testing/tiny-Qwen3MoeForCausalLM", id="qwen3"),
        pytest.param("trl-internal-testing/tiny-Qwen3VLForConditionalGeneration", id="qwen3_vl"),
        pytest.param("trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration", id="qwen35"),
        pytest.param(
            "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
            id="gemma4",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.5.0"),
                reason="Gemma4 models were introduced in transformers-5.5.0",
            ),
        ),
    ],
)
@pytest.mark.xfail(
    condition=Version(transformers.__version__) < Version("5.0.0"),
    reason="Response parsing is not supported in transformers versions below 5.0.0",
    strict=True,
)
@require_jmespath
class TestParseResponse:
    def test_parse_response(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        messages = [
            {"role": "user", "content": "What is 3*4?"},
            {"role": "assistant", "content": "12"},
        ]
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True).input_ids
        text = tokenizer.apply_chat_template(messages).input_ids
        response = text[len(prefix) :]
        parsed = parse_response(tokenizer, response)
        assert parsed == messages[-1]

    def test_parse_response_with_reasoning_content(self, tokenizer_name):
        if tokenizer_name in (
            "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
        ):
            pytest.skip("This tokenizer doesn't support inline reasoning_content.")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        messages = [
            {"role": "user", "content": "What is 3*4?"},
            {"role": "assistant", "reasoning_content": "Hmmm.", "content": "12"},
        ]
        # enable_thinking=True is required here because for Qwen3.5, the thinking is disabled by default for the
        # generation prompt.
        prefix = tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True, enable_thinking=True
        ).input_ids
        text = tokenizer.apply_chat_template(messages).input_ids
        response = text[len(prefix) :]
        parsed = parse_response(tokenizer, response)
        assert parsed == messages[-1]

    def test_parse_response_tool_call(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        tool_calls = [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}]
        messages = [
            {"role": "user", "content": "What is 3*4?"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True).input_ids
        text = tokenizer.apply_chat_template(messages).input_ids
        response = text[len(prefix) :]
        parsed = parse_response(tokenizer, response)
        assert parsed == messages[-1]

    def test_parse_response_tool_call_with_content(self, tokenizer_name):
        if tokenizer_name == "trl-internal-testing/tiny-Gemma4ForConditionalGeneration":
            # Gemma4 response_schema regex doesn't capture content after tool calls.
            # Remove once https://huggingface.co/google/gemma-4-31B-it/discussions/19 is merged.
            pytest.xfail("Gemma4 response_schema regex bug.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        tool_calls = [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}]
        messages = [
            {"role": "user", "content": "What is 3*4?"},
            {"role": "assistant", "content": "Let's call the tool.", "tool_calls": tool_calls},
        ]
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True).input_ids
        text = tokenizer.apply_chat_template(messages).input_ids
        response = text[len(prefix) :]
        parsed = parse_response(tokenizer, response)
        assert parsed == messages[-1]

    def test_parse_response_tool_call_without_arguments(self, tokenizer_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        tool_calls = [{"type": "function", "function": {"name": "ping", "arguments": {}}}]
        messages = [
            {"role": "user", "content": "Ping the service."},
            {"role": "assistant", "tool_calls": tool_calls},
        ]
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True).input_ids
        text = tokenizer.apply_chat_template(messages).input_ids
        response = text[len(prefix) :]
        parsed = parse_response(tokenizer, response)
        assert parsed == {"role": "assistant", "content": "", "tool_calls": tool_calls}

    def test_parse_response_multiple_tool_calls(self, tokenizer_name):
        if tokenizer_name == "trl-internal-testing/tiny-GptOssForCausalLM":
            pytest.skip("GPT-OSS template only renders one tool call per assistant message.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        tool_calls = [
            {"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}},
            {"type": "function", "function": {"name": "addition", "arguments": {"a": 4, "b": 3}}},
        ]
        messages = [
            {"role": "user", "content": "What is 3*4?"},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
        prefix = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True).input_ids
        text = tokenizer.apply_chat_template(messages).input_ids
        response = text[len(prefix) :]
        parsed = parse_response(tokenizer, response)
        assert parsed == messages[-1]

    def test_parse_response_malformed_tool_call(self, tokenizer_name):
        if tokenizer_name != "trl-internal-testing/tiny-Qwen3MoeForCausalLM":
            pytest.skip("For simplicity, we only test the malformed tool call case on one tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if getattr(tokenizer, "response_schema", None) is None:
            tokenizer = add_response_schema(tokenizer)
        text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}\n</tool_call><|im_end|>'
        assistant_text = tokenizer(text)["input_ids"]
        parsed = parse_response(tokenizer, assistant_text)
        expected = {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}\n</tool_call>',
        }

        assert parsed == expected
