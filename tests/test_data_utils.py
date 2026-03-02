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

import copy
import textwrap
from time import strftime

import pytest
import transformers
from datasets import Dataset, DatasetDict
from packaging.version import Version
from transformers import AutoProcessor, AutoTokenizer, is_vision_available

from trl.data_utils import (
    apply_chat_template,
    extract_prompt,
    is_conversational,
    is_conversational_from_value,
    maybe_apply_chat_template,
    maybe_convert_to_chatml,
    maybe_extract_prompt,
    maybe_unpair_preference_dataset,
    pack_dataset,
    prepare_multimodal_messages,
    prepare_multimodal_messages_vllm,
    truncate_dataset,
    unpair_preference_dataset,
)

from .testing_utils import TrlTestCase, require_vision


if is_vision_available():
    from PIL import Image


@require_vision
class TestPrepareMultimodalMessages:
    def test_basic_user_assistant_conversation(self):
        """Test basic conversation with user and assistant messages."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ]
        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        assert messages == expected

    def test_first_user_message_gets_image(self):
        """Test that only the first user message gets an image."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
            {"role": "user", "content": "How about the grass?"},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "How about the grass?"}],
            },
        ]

        assert messages == expected

    def test_multiple_images(self):
        """Test that multiple images are added to the first user message."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ]
        images = [Image.new("RGB", (10, 10), color=color) for color in ["red", "green", "blue"]]
        messages = prepare_multimodal_messages(messages, images=images)

        expected = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "image", "image": images[2]},
                    {"type": "text", "text": "What color is the sky?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        assert messages == expected

    def test_system_message_transformation(self):
        """Test that system messages are properly transformed."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What color is the sky?"},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant"}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
        ]

        assert messages == expected

    def test_already_prepared_messages_unchanged(self):
        """Test that messages with list content are not modified."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant"}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is blue."}]},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant"}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        assert messages == expected

    def test_mixed_prepared_and_unprepared_messages(self):
        """Test handling of mixed prepared and unprepared messages."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": [{"type": "text", "text": "It is blue."}]},
            {"role": "user", "content": "What about the grass?"},
        ]

        image = Image.new("RGB", (10, 10), color="blue")
        messages = prepare_multimodal_messages(messages, images=[image])

        expected = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What color is the sky?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What about the grass?"}],
            },
        ]

        assert messages == expected


@require_vision
class TestPrepareMultimodalMessagesVLLM:
    def test_single_image_conversion(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (10, 10), color="blue")},
                    {"type": "text", "text": "What color is the sky?"},
                ],
            }
        ]

        result = prepare_multimodal_messages_vllm(messages)

        # Original should remain unchanged (deepcopy test)
        assert messages[0]["content"][0]["type"] == "image"

        # Converted version should have correct structure
        assert result[0]["content"][0]["type"] == "image_pil"
        assert "image_pil" in result[0]["content"][0]
        assert "image" not in result[0]["content"][0]
        assert isinstance(result[0]["content"][0]["image_pil"], Image.Image)
        assert result[0]["content"][1]["type"] == "text"

    def test_mixed_content_conversion(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the sky?"},
                    {"type": "image", "image": Image.new("RGB", (10, 10), color="blue")},
                ],
            }
        ]

        result = prepare_multimodal_messages_vllm(messages)

        # The image part should be converted, text should be unchanged
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_pil"

    def test_no_images(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "What color is the sky?"}]}]

        result = prepare_multimodal_messages_vllm(messages)

        # Should be identical since there are no images
        assert result == messages
        # And a deepcopy — not the same object
        assert result is not messages
        assert result[0] is not messages[0]

    def test_multiple_messages(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the sky?"},
                    {"type": "image", "image": Image.new("RGB", (10, 10), color="blue")},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is blue."}],
            },
        ]

        result = prepare_multimodal_messages_vllm(messages)

        assert result[0]["content"][1]["type"] == "image_pil"
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "It is blue."

    def test_deepcopy_integrity(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the sky?"},
                    {"type": "image", "image": Image.new("RGB", (10, 10), color="blue")},
                ],
            },
        ]
        original = copy.deepcopy(messages)

        _ = prepare_multimodal_messages_vllm(messages)

        # Original should not be mutated
        assert messages == original


class TestIsConversational(TrlTestCase):
    # fmt: off
    conversational_examples = [
        {  # Language modeling
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        },
        {  # Prompt-only
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
        },
        {  # Prompt-completion
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        },
        {  # Preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}],
        },
        {  # Preference with implicit prompt
            "chosen": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."},
            ],
        },
        {  # Preference with tool calls
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_color", "arguments": {"what": "sky"}}}]},
                {"role": "tool", "name": "get_color", "content": "blue"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_color", "arguments": {"what": "tree"}}}]},
                {"role": "tool", "name": "get_color", "content": "green"},
                {"role": "assistant", "content": "It is green."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "description": "Gets the color.",
                        "name": "get_color",
                        "parameters": {"properties": {"what": {"description": "What to get the color of.", "type": "string"}}, "required": ["what"], "type": "object"},
                        "return": {"description": "The color.", "type": "string"},
                    },
                },
            ],
        },
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
        {  # Language modeling with harmony
            "messages": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        },
        {  # Prompt-only with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
        },
        {  # Prompt-completion with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        },
        {  # Preference with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "chosen": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        },
        {  # Preference with implicit prompt and harmony
            "chosen": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        },
        {  # Unpaired preference with harmony
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "label": True,
        },
    ]
    # fmt: on

    non_conversational_examples = [
        {"prompt": "The sky is", "completion": " blue."},
        {"text": "The sky is blue."},
        {"prompt": "The sky is"},
        {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."},
        {"prompt": "The sky is", "completion": " blue.", "label": True},
    ]

    @pytest.mark.parametrize("example", conversational_examples)
    def test_conversational(self, example):
        assert is_conversational(example)

    @pytest.mark.parametrize("example", non_conversational_examples)
    def test_non_conversational(self, example):
        assert not is_conversational(example)


class TestIsConversationalFromValue(TrlTestCase):
    def test_positive_1(self):
        example = {
            "conversations": [
                {"from": "user", "value": "What color is the sky?"},
                {"from": "assistant", "value": "It is blue."},
            ],
        }
        assert is_conversational_from_value(example)

    def test_negative_1(self):
        example = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        }
        assert not is_conversational_from_value(example)

    def test_negative_2(self):
        example = {"text": "The sky is blue."}
        assert not is_conversational_from_value(example)


class TestApplyChatTemplate(TrlTestCase):
    tokenizers = [
        "trl-internal-testing/tiny-CohereForCausalLM",
        "trl-internal-testing/tiny-Cohere2ForCausalLM",
        "trl-internal-testing/tiny-DeepseekV3ForCausalLM",
        "trl-internal-testing/tiny-DeepseekV3ForCausalLM-0528",
        "trl-internal-testing/tiny-FalconMambaForCausalLM",
        "trl-internal-testing/tiny-Gemma2ForCausalLM",
        "trl-internal-testing/tiny-GemmaForCausalLM",
        "trl-internal-testing/tiny-GptOssForCausalLM",
        pytest.param(
            "trl-internal-testing/tiny-Glm4MoeForCausalLM",
            marks=pytest.mark.skipif(
                Version(transformers.__version__) < Version("5.0.0"),
                reason="GLM4 tokenizer requires transformers>=5.0.0",
            ),
        ),
        "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "trl-internal-testing/tiny-LlamaForCausalLM-3",
        "trl-internal-testing/tiny-MistralForCausalLM-0.1",
        "trl-internal-testing/tiny-MistralForCausalLM-0.2",
        "trl-internal-testing/tiny-Phi3ForCausalLM",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
    ]

    conversational_examples = [
        {  # Language modeling
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
        },
        {  # Prompt-only
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
        },
        {  # Prompt-completion
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        },
        {  # Preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "chosen": [{"role": "assistant", "content": "It is blue."}],
            "rejected": [{"role": "assistant", "content": "It is green."}],
        },
        {  # Preference with implicit prompt
            "chosen": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is green."},
            ],
        },
        {  # Unpaired preference
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
            "label": True,
        },
    ]

    non_conversational_examples = [
        {"text": "The sky is blue."},  # Language modeling
        {"prompt": "The sky is"},  # Prompt-only
        {"prompt": "The sky is", "completion": " blue."},  # Prompt-completion
        {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."},  # Preference
        {"chosen": "The sky is blue.", "rejected": "The sky is green."},  # Preference with implicit prompt
        {"prompt": "The sky is", "completion": " blue.", "label": True},  # Unpaired preference
    ]

    @pytest.mark.parametrize("example", conversational_examples)
    @pytest.mark.parametrize("tokenizer_id", tokenizers)
    def test_apply_chat_template(self, tokenizer_id, example):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        result = apply_chat_template(example, tokenizer)

        # Checking if the result is a dictionary
        assert isinstance(result, dict)

        # The chat template should be applied to the following keys
        for key in ["prompt", "chosen", "rejected", "completion"]:
            if key in example:
                assert key in result
                assert isinstance(result[key], str)

        # Exception for messages, the key is "text" once the chat template is applied
        if "messages" in example:
            assert "text" in result
            assert isinstance(result["text"], str)

        # The label should be kept
        if "label" in example:
            assert "label" in result
            assert isinstance(result["label"], bool)
            assert result["label"] == example["label"]

    # both conversational and non-conversational examples
    @pytest.mark.parametrize("example", conversational_examples + non_conversational_examples)
    @pytest.mark.parametrize("tokenizer_id", tokenizers)
    def test_maybe_apply_chat_template(self, tokenizer_id, example):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        result = maybe_apply_chat_template(example, tokenizer)

        # Checking if the result is a dictionary
        assert isinstance(result, dict)

        # The chat template should be applied to the following keys
        for key in ["prompt", "chosen", "rejected", "completion"]:
            if key in example:
                assert key in result
                assert isinstance(result[key], str)

        # Exception for messages, the key is "text" once the chat template is applied
        if "messages" in example:
            assert "text" in result
            assert isinstance(result["text"], str)

        # The label should be kept
        if "label" in example:
            assert "label" in result
            assert isinstance(result["label"], bool)
            assert result["label"] == example["label"]

    def test_apply_chat_template_with_chat_template_kwargs(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen3ForCausalLM")

        example = {
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            # with this tokenizer, when you pass enable_thinking=False, it will add "<think>\n\n</think>\n\n"
            "chat_template_kwargs": {"enable_thinking": False},
        }
        result = apply_chat_template(example, tokenizer)

        # docstyle-ignore
        expected = textwrap.dedent("""\
        <|im_start|>user
        What color is the sky?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        """)

        assert result["prompt"] == expected

    def test_apply_chat_template_with_tools(self):
        tokenizer = AutoProcessor.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")

        # Define dummy test tools
        def get_current_temperature(location: str):
            """
            Gets the temperature at a given location.

            Args:
                location: The location to get the temperature for
            """
            return 22.0

        # Define test case
        test_case = {
            "prompt": [
                {"content": "What's the temperature in London?", "role": "user"},
            ]
        }
        # Test with tools
        result_with_tools = apply_chat_template(test_case, tokenizer, tools=[get_current_temperature])

        # Verify tools are included in the output
        assert "get_current_temperature" in result_with_tools["prompt"]

        # Test without tools
        result_without_tools = apply_chat_template(test_case, tokenizer, tools=None)

        # Verify tools are not included in the output
        assert "get_current_temperature" not in result_without_tools["prompt"]


class TestApplyChatTemplateHarmony(TrlTestCase):
    def test_language_modeling(self):
        messages = {
            "messages": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        }
        output = apply_chat_template(
            messages,
            tokenizer=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>""")

        assert output["text"] == expected

    def test_prompt_only(self):
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
        }
        output = apply_chat_template(
            messages,
            tokenizer=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")

        assert output["prompt"] == expected

    def test_prompt_completion(self):
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
        }
        output = apply_chat_template(
            messages,
            tokenizer=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_prompt = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")
        expected_completion = "<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>"

        assert output["prompt"] == expected_prompt
        assert output["completion"] == expected_completion

    def test_preference(self):
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "chosen": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        }
        output = apply_chat_template(
            messages,
            tokenizer=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_prompt = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")
        expected_chosen = "<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>"
        expected_rejected = "<|channel|>analysis<|message|>The user asks the color of the tree...<|end|><|start|>assistant<|channel|>final<|message|>It is green.<|return|>"

        assert output["prompt"] == expected_prompt
        assert output["chosen"] == expected_chosen
        assert output["rejected"] == expected_rejected

    def test_preference_with_implicit_prompt(self):
        messages = {
            "chosen": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "rejected": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "thinking": "The user asks the color of the tree...", "content": "It is green."},
            ],
        }
        output = apply_chat_template(
            messages,
            tokenizer=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_chosen = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>""")

        # docstyle-ignore
        expected_rejected = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant<|channel|>analysis<|message|>The user asks the color of the tree...<|end|><|start|>assistant<|channel|>final<|message|>It is green.<|return|>""")

        assert output["chosen"] == expected_chosen
        assert output["rejected"] == expected_rejected

    def test_unpaired_preference(self):
        messages = {
            "prompt": [
                {"role": "system", "content": "Respond in a friendly manner."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            "completion": [
                {"role": "assistant", "thinking": "The user asks the color of the sky...", "content": "It is blue."},
            ],
            "label": True,
        }
        output = apply_chat_template(
            messages,
            tokenizer=AutoTokenizer.from_pretrained("trl-internal-testing/tiny-GptOssForCausalLM"),
            reasoning_effort="low",
            model_identity="You are HuggingGPT.",
        )

        # docstyle-ignore
        expected_prompt = textwrap.dedent(f"""\
        <|start|>system<|message|>You are HuggingGPT.
        Knowledge cutoff: 2024-06
        Current date: {strftime("%Y-%m-%d")}

        Reasoning: low

        # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

        Respond in a friendly manner.

        <|end|><|start|>user<|message|>What color is the sky?<|end|><|start|>assistant""")
        expected_completion = "<|channel|>analysis<|message|>The user asks the color of the sky...<|end|><|start|>assistant<|channel|>final<|message|>It is blue.<|return|>"

        assert output["prompt"] == expected_prompt
        assert output["completion"] == expected_completion
        assert output["label"]


class TestUnpairPreferenceDataset(TrlTestCase):
    paired_dataset = Dataset.from_dict(
        {
            "prompt": ["The sky is", "The sun is"],
            "chosen": [" blue.", " in the sky."],
            "rejected": [" green.", " in the sea."],
        }
    )

    unpaired_dataset = Dataset.from_dict(
        {
            "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
            "completion": [" blue.", " in the sky.", " green.", " in the sea."],
            "label": [True, True, False, False],
        }
    )

    def test_unpair_preference_dataset(self):
        # Test that a paired dataset is correctly converted to unpaired
        unpaired_dataset = unpair_preference_dataset(self.paired_dataset)
        assert unpaired_dataset.to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_unpair_preference_dataset_dict(self):
        # Test that a paired dataset dict is correctly converted to unpaired
        paired_dataset_dict = DatasetDict({"abc": self.paired_dataset})
        unpaired_dataset_dict = unpair_preference_dataset(paired_dataset_dict)
        assert unpaired_dataset_dict["abc"].to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_maybe_unpair_preference_dataset(self):
        # Test that a paired dataset is correctly converted to unpaired with maybe_unpair_preference_dataset
        unpaired_dataset = maybe_unpair_preference_dataset(self.paired_dataset)
        assert unpaired_dataset.to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_maybe_unpair_preference_dataset_dict(self):
        # Test that a paired dataset dict is correctly converted to unpaired with maybe_unpair_preference_dataset
        paired_dataset_dict = DatasetDict({"abc": self.paired_dataset})
        unpaired_dataset_dict = maybe_unpair_preference_dataset(paired_dataset_dict)
        assert unpaired_dataset_dict["abc"].to_dict() == self.unpaired_dataset.to_dict(), (
            "The paired dataset should be converted to unpaired."
        )

    def test_maybe_unpair_preference_dataset_already_paired(self):
        # Test that a paired dataset remains unchanged with maybe_unpair_preference_dataset
        unpaired_dataset = maybe_unpair_preference_dataset(self.unpaired_dataset)
        assert unpaired_dataset.to_dict() == self.unpaired_dataset.to_dict(), (
            "The unpaired dataset should remain unchanged."
        )

    def test_maybe_unpair_preference_dataset_dict_already_paired(self):
        # Test that a paired dataset dict remains unchanged with maybe_unpair_preference_dataset
        unpaired_dataset_dict = maybe_unpair_preference_dataset(DatasetDict({"abc": self.unpaired_dataset}))
        assert unpaired_dataset_dict["abc"].to_dict() == self.unpaired_dataset.to_dict(), (
            "The unpaired dataset should remain unchanged."
        )


class TestExtractPrompt(TrlTestCase):
    example_implicit_prompt_conversational = {
        "chosen": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is blue."},
        ],
        "rejected": [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "It is green."},
        ],
    }

    example_explicit_prompt_conversational = {
        "prompt": [
            {"role": "user", "content": "What color is the sky?"},
        ],
        "chosen": [
            {"role": "assistant", "content": "It is blue."},
        ],
        "rejected": [
            {"role": "assistant", "content": "It is green."},
        ],
    }

    example_implicit_prompt_standard = {
        "chosen": "The sky is blue.",
        "rejected": "The sky is green.",
    }

    example_explicit_prompt_standard = {
        "prompt": "The sky is",
        "chosen": " blue.",
        "rejected": " green.",
    }

    def test_extract_prompt_conversational(self):
        # Test that the prompt is correctly extracted from the dataset
        example_extracted_prompt = extract_prompt(self.example_implicit_prompt_conversational)
        assert example_extracted_prompt == self.example_explicit_prompt_conversational, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_conversational(self):
        # Test that the prompt is correctly extracted from the dataset with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_implicit_prompt_conversational)
        assert example_extracted_prompt == self.example_explicit_prompt_conversational, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_conversational_already_explicit(self):
        # Test that the prompt remains unchanged with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_explicit_prompt_conversational)
        assert example_extracted_prompt == self.example_explicit_prompt_conversational, (
            "The prompt should remain unchanged."
        )

    def test_extract_prompt_standard(self):
        # Test that the prompt is correctly extracted from the dataset
        example_extracted_prompt = extract_prompt(self.example_implicit_prompt_standard)
        assert example_extracted_prompt == self.example_explicit_prompt_standard, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_standard(self):
        # Test that the prompt is correctly extracted from the dataset with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_implicit_prompt_standard)
        assert example_extracted_prompt == self.example_explicit_prompt_standard, (
            "The prompt is not correctly extracted from the dataset."
        )

    def test_maybe_extract_prompt_standard_already_explicit(self):
        # Test that the prompt remains unchanged with maybe_extract_prompt
        example_extracted_prompt = maybe_extract_prompt(self.example_explicit_prompt_standard)
        assert example_extracted_prompt == self.example_explicit_prompt_standard, "The prompt should remain unchanged."


class TestPackDatasetWrapped(TrlTestCase):
    def test_with_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="wrapped")
        assert dataset.to_dict() == expected_output

    def test_with_iterable_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        seq_length = 3
        expected_output = {
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1], [1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="wrapped")
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.batch(batch_size=num_examples))) == expected_output


class TestPackDatasetBfd(TrlTestCase):
    def test_simple(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "input_ids": [[4, 5, 6, 7], [1, 2, 3, 8]],
            "seq_lengths": [[4], [3, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd")
        assert dataset.to_dict() == expected_output

    def test_with_iterable_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        seq_length = 4
        expected_output = {
            "input_ids": [[4, 5, 6, 7], [1, 2, 3, 8]],
            "seq_lengths": [[4], [3, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd")
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.batch(batch_size=num_examples))) == expected_output

    def test_with_overlong_0(self):
        examples = {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11], [12]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "input_ids": [[1, 2, 3, 4], [8, 9, 10, 11], [6, 7, 5, 12]],
            "seq_lengths": [[4], [4], [2, 1, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd-requeue")
        assert dataset.to_dict() == expected_output

    def test_with_overlong_two_coluns(self):
        examples = {
            "col1": [[1, -2, 3, -4, 5, -6], [7, -8, 9], [-10, 11, -12], [13, -14, 15, -16]],
            "col2": [[-1, 2, -3, 4, -5, 6], [-7, 8, -9], [10, -11, 12], [-13, 14, -15, 16]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        expected_output = {
            "col1": [[1, -2, 3, -4], [13, -14, 15, -16], [7, -8, 9], [-10, 11, -12], [5, -6]],
            "col2": [[-1, 2, -3, 4], [-13, 14, -15, 16], [-7, 8, -9], [10, -11, 12], [-5, 6]],
            "seq_lengths": [[4], [4], [3], [3], [2]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd-requeue")
        assert dataset.to_dict() == expected_output

    def test_with_non_power_of_2(self):
        examples = {
            "input_ids": [[1, 2, 3, 4, 5], [6], [7, 8, 9, 10], [11, 12, 13]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 5
        expected_output = {
            "input_ids": [[1, 2, 3, 4, 5], [7, 8, 9, 10, 6], [11, 12, 13]],
            "seq_lengths": [[5], [4, 1], [3]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd-requeue")
        assert dataset.to_dict() == expected_output

    def test_default_no_requeue(self):
        """Test default 'bfd' strategy for SFT datasets (truncates overflow)."""
        examples = {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11], [12]],
        }
        dataset = Dataset.from_dict(examples)
        seq_length = 4
        # With default 'bfd' strategy, overflow tokens are discarded
        expected_output = {
            "input_ids": [[1, 2, 3, 4], [8, 9, 10, 11], [6, 7, 12]],
            "seq_lengths": [[4], [4], [2, 1]],
        }
        dataset = pack_dataset(dataset, seq_length, strategy="bfd")
        assert dataset.to_dict() == expected_output


class TestTruncateExamples(TrlTestCase):
    def test_with_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        assert dataset.to_dict() == expected_output

    def test_with_iterable_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.batch(batch_size=num_examples))) == expected_output

    def test_with_extra_column(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = Dataset.from_dict(examples)
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = truncate_dataset(dataset, max_length)
        assert dataset.to_dict() == expected_output


class TestMaybeConvertToChatML(TrlTestCase):
    def test_with_conversations_key(self):
        # Particular case where the key is "conversations": we rename it to "messages"
        example = {
            "conversations": [
                {"from": "user", "value": "What color is the sky?"},
                {"from": "assistant", "value": "It is blue."},
            ]
        }
        expected_output = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ]
        }
        assert maybe_convert_to_chatml(example) == expected_output

    def test_without_conversations_key(self):
        # Same as before, but we don't rename the keys
        example = {
            "prompt": [{"from": "user", "value": "What color is the sky?"}],
            "completion": [{"from": "assistant", "value": "It is blue."}],
        }
        expected_output = {
            "prompt": [{"role": "user", "content": "What color is the sky?"}],
            "completion": [{"role": "assistant", "content": "It is blue."}],
        }
        assert maybe_convert_to_chatml(example) == expected_output

    def test_not_conversional(self):
        # When not needed, the example should remain unchanged
        example = {"text": "The sky is blue."}
        assert maybe_convert_to_chatml(example) == example

    def test_already_chatml(self):
        # When the example is already in ChatML format, it should remain unchanged
        example = {
            "messages": [
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "It is blue."},
            ]
        }
        assert maybe_convert_to_chatml(example) == example
