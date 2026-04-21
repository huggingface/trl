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

from pathlib import Path
from typing import TypeVar

from jinja2 import TemplateError
from transformers import AddedToken, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .data_utils import prepare_multimodal_messages


_CHAT_TEMPLATES_DIR = Path(__file__).parent / "chat_templates"


def clone_chat_template(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    source_tokenizer_path: str,
    resize_to_multiple_of: int | None = 64,
) -> tuple[PreTrainedModel, PreTrainedTokenizer, list[int]]:
    """
    Clones a chat template from a source tokenizer to the target tokenizer and updates the model accordingly.

    This function:
    - Copies the chat template from a source tokenizer to the target tokenizer.
    - Adds any new tokens from the source tokenizer to the target tokenizer.
    - Sets and synchronizes the EOS token across the tokenizer and model.
    - Resizes the model's token embeddings to match the new vocabulary size, optionally rounding it up to a multiple of
      a specified value. In such cases, dummy tokens are added to the tokenizer to ensure the vocabulary size matches
      the embedding dimensions.

    Args:
        model ([`~transformers.PreTrainedModel`]):
            Model to update.
        tokenizer ([`~transformers.PreTrainedTokenizer`]):
            Tokenizer to update.
        source_tokenizer_path (`str`):
            Path or identifier of the pretrained tokenizer to clone from.
        resize_to_multiple_of (`int` or `None`, *optional*, defaults to `64`):
            The embedding layer will be resized to the new vocabulary size. If this is not `None`, it will round up the
            new vocabulary size to the nearest multiple of this value.

    Returns:
        model ([`~transformers.PreTrainedModel`]):
            Updated model with resized token embeddings and EOS token configured.
        tokenizer ([`~transformers.PreTrainedTokenizer`]):
            Updated tokenizer with the chat template and special tokens applied.
        added_tokens (`list[int]`):
            List of tokens that were added to the tokenizer from the source tokenizer.

    Example:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import clone_chat_template

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model, tokenizer, added_tokens = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")
    ```
    """
    # Load the source tokenizer containing the desired chat template
    tokenizer_source = AutoTokenizer.from_pretrained(source_tokenizer_path)

    # Copy the chat template from the source tokenizer
    tokenizer.chat_template = tokenizer_source.get_chat_template()

    # Ensure all added tokens from the source are available in the target tokenizer
    added_tokens = [
        token for token in tokenizer_source.added_tokens_decoder.values() if token.content not in tokenizer.vocab
    ]
    tokenizer.add_tokens(added_tokens)

    # Set the EOS token from the source tokenizer (important for generation)
    tokenizer.eos_token = tokenizer_source.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    if model.can_generate():  # Non-generative models (e.g. SequenceClassification) may not have a generation_config
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Resize model embeddings to include any new tokens, optionally rounding up to a multiple
    model.resize_token_embeddings(
        # After studying many tokenizers, we found that len(tokenizer.vocab) is the most reliable way to get the vocab
        # size. Avoid using tokenizer.vocab_size or tokenizer.vocab_size + len(tokenizer.added_tokens_encoder),
        # as handling of special and added tokens varies across tokenizers.
        new_num_tokens=len(tokenizer.vocab),
        pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None,
    )

    # After resizing, the embedding matrix size may exceed the vocabulary size. Add dummy tokens to the tokenizer to
    # ensure vocabulary size matches the embedding matrix dimensions.
    idx = 0
    while model.vocab_size > len(tokenizer.vocab):
        dummy_token = AddedToken(f"<extra_id_{idx}>")
        is_added = tokenizer.add_tokens(dummy_token)
        idx += 1
        if is_added == 1:
            added_tokens.append(dummy_token)

    # Verify that vocabulary size now matches embedding dimensions
    if len(tokenizer.vocab) != model.vocab_size:
        raise RuntimeError(
            f"Vocabulary size mismatch after resizing: tokenizer vocab size is {len(tokenizer.vocab)}, but model "
            f"embedding size is {model.vocab_size}. This indicates an internal error in the token alignment process."
        )
    added_tokens = [token.content for token in added_tokens]
    added_tokens = tokenizer.convert_tokens_to_ids(added_tokens)
    return model, tokenizer, added_tokens


glm4moe_schema = {
    "x-regex": r"^(?:\n?<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>.*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|$))(?P<tool_calls>(?:<tool_call>.+?</tool_call>\s*)+)?$",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"^(\S+)"},
                            "arguments": {
                                "type": "object",
                                "x-regex-key-value": r"<arg_key>(?P<key>[^<]+)</arg_key>\s*\n<arg_value>(?P<value>.*?)</arg_value>",
                                "default": {},
                                "additionalProperties": {
                                    "x-parser": "json",
                                    "x-parser-args": {"allow_non_json": True},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

gptoss_schema = {
    # Normalize final content to analysis format so both map to the same "content" group.
    "x-regex-substitutions": [
        [r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", r"<|channel|>analysis<|message|>\1<|end|>"],
    ],
    "x-regex": r"^(?:<\|channel\|>analysis<\|message\|>(?P<content>.*?)<\|end\|>(?:<\|start\|>assistant)?)?\s*(?P<tool_calls>to=functions\.\S+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>)?$",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"(to=functions\.\S+<\|channel\|>commentary json<\|message\|>.*?<\|call\|>)",
            "items": {
                # Convert "to=functions.NAME<|channel|>commentary json<|message|>ARGS<|call|>"
                # into '{"name": "NAME", "arguments": ARGS}' so it can be parsed as JSON.
                "x-regex-substitutions": [
                    [
                        r"to=functions\.(\S+)<\|channel\|>commentary json<\|message\|>(.*?)<\|call\|>",
                        r'{"name": "\1", "arguments": \2}',
                    ],
                ],
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
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

# Adapted and corrected versions of the schemas from:
# https://github.com/huggingface/transformers/blob/main/tests/utils/test_chat_parsing_utils.py
qwen3_schema = {
    "x-regex": r"^(?:<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>.*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>.+?</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
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

llama3_schema = {
    # Llama 3.1 / 3.2 render a tool call as a single bare JSON object using the key "parameters" instead of
    # "arguments": `{"name": "<name>", "parameters": <args_json>}<|eot_id|>`. There is no surrounding marker, no
    # support for content alongside a tool call, and at most one tool call per assistant turn (the template raises
    # otherwise). Either we match a tool call (capturing the JSON) or we treat the response as plain content.
    "x-regex": r'^(?:(?P<tool_calls>\{"name":\s*".+?",\s*"parameters":\s*.+\})|(?P<content>.*?))(?:<\|eot_id\|>|$)',
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r'(\{"name":\s*".+?",\s*"parameters":\s*.+\})',
            "items": {
                # Rewrite "parameters" → "arguments" so the JSON parses into the standard tool-call shape. Anchored
                # on the leading `{"name": "..."` so a stray `"parameters"` inside argument values is not touched.
                "x-regex-substitutions": [
                    [r'^(\{"name":\s*"[^"]+",\s*)"parameters":', r'\1"arguments":'],
                ],
                "x-parser": "json",
                "x-parser-args": {"transform": "{type: 'function', function: @}"},
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

qwen3_5_schema = {
    "x-regex": r"^(?:(?:<think>\n?)?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>.*?)(?:\n+(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>.+?</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.+?)\s*</tool_call>",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "x-regex": r"<function=([^\n>]+)>"},
                            "arguments": {
                                "type": "object",
                                "x-regex-key-value": r"<parameter=(?P<key>[^>\n]+)>\n(?P<value>.*?)\n</parameter>",
                                "default": {},
                                "additionalProperties": {
                                    "x-parser": "json",
                                    "x-parser-args": {"allow_non_json": True},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


deepseekv3_chat_template = (_CHAT_TEMPLATES_DIR / "deepseekv3.jinja").read_text()

glm4moe_chat_template = (_CHAT_TEMPLATES_DIR / "glm4moe.jinja").read_text()

gptoss_chat_template = (_CHAT_TEMPLATES_DIR / "gptoss.jinja").read_text()

llama3_chat_template = (_CHAT_TEMPLATES_DIR / "llama3.jinja").read_text()

llama3_1_chat_template = (_CHAT_TEMPLATES_DIR / "llama3_1.jinja").read_text()

llama3_2_chat_template = (_CHAT_TEMPLATES_DIR / "llama3_2.jinja").read_text()

qwen2_5_chat_template = (_CHAT_TEMPLATES_DIR / "qwen2_5.jinja").read_text()

qwen3_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3.jinja").read_text()

qwen3_vl_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_vl.jinja").read_text()

qwen3_5_chat_template_2b_and_below = (_CHAT_TEMPLATES_DIR / "qwen3_5_2b_and_below.jinja").read_text()

qwen3_5_chat_template_4b_and_above = (_CHAT_TEMPLATES_DIR / "qwen3_5_4b_and_above.jinja").read_text()


ProcessingClassT = TypeVar("ProcessingClassT", PreTrainedTokenizer, ProcessorMixin)


def add_response_schema(processing_class: ProcessingClassT) -> ProcessingClassT:
    r"""
    Adds the appropriate response schema to the given tokenizer based on its chat template.

    At the time of initial implementation, most tokenizers do not have built-in support for response schemas. While
    waiting for broader adoption, we provide this utility function to manually set the response schema for known chat
    templates.

    When given a VLM processor, the schema is set on the inner tokenizer, since `parse_response` is a tokenizer method
    and reads `self.response_schema` from the tokenizer instance.

    Args:
        processing_class (`PreTrainedTokenizer` or `ProcessorMixin`):
            Tokenizer or VLM processor to which the response schema will be added.

    Returns:
        `PreTrainedTokenizer` or `ProcessorMixin`:
            The same object that was passed in, with the response schema set on the underlying tokenizer.

    Examples:

    ```python
    >>> from trl.chat_template_utils import add_response_schema
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> tokenizer = add_response_schema(tokenizer)
    >>> assistant_text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
    >>> tokenizer.parse_response(assistant_text)
    {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'multiply', 'arguments': {'a': 3, 'b': 4}}}]}
    ```
    """
    # For VLM processors, set the schema on the inner tokenizer (where `parse_response` reads it from).
    # Match against the top-level chat_template, since that's what was used historically and processors
    # may carry their own VLM-specific template separate from the inner tokenizer's.
    chat_template = processing_class.chat_template
    if isinstance(processing_class, ProcessorMixin):
        tokenizer = processing_class.tokenizer
    else:
        tokenizer = processing_class
    if chat_template == glm4moe_chat_template:
        tokenizer.response_schema = glm4moe_schema
    elif chat_template == gptoss_chat_template:
        tokenizer.response_schema = gptoss_schema
    elif chat_template in [llama3_1_chat_template, llama3_2_chat_template]:
        tokenizer.response_schema = llama3_schema
    elif chat_template in [qwen3_chat_template, qwen3_vl_chat_template]:
        tokenizer.response_schema = qwen3_schema
    elif chat_template in [qwen3_5_chat_template_2b_and_below, qwen3_5_chat_template_4b_and_above]:
        tokenizer.response_schema = qwen3_5_schema
    else:
        raise ValueError(
            "Unrecognized chat template, failed to add response schema. Please manually set the response schema on "
            "the tokenizer or processor. See the Transformers "
            "[docs](https://huggingface.co/docs/transformers/main/en/chat_response_parsing#response-parsing) for more "
            "details on response parsing."
        )
    return processing_class


def supports_tool_calling(processing_class) -> bool:
    """
    Check if the processing class's chat template can render a full tool-calling conversation.

    This tests that (1) the template doesn't error when rendering a conversation with ``user → assistant (with
    tool_calls) → tool`` roles, and (2) every part of the tool-calling exchange — the assistant's tool call name, its
    arguments, and the tool message content — actually appears in the rendered output. Some templates silently swallow
    `tool_calls` (e.g. the basic Llama 3 template, which only reads `message['content']`) or tool messages (e.g.
    Cohere2, Phi3); both cases must be rejected.

    For VLMs (processors), the messages are converted to multimodal format via
    [`~trl.data_utils.prepare_multimodal_messages`] before rendering.

    Args:
        processing_class (`PreTrainedTokenizer` or `ProcessorMixin`):
            Tokenizer or processor instance to check.

    Returns:
        `bool`:
            `True` if the chat template supports tool-calling conversations, `False` otherwise.
    """
    if processing_class.chat_template is None:
        return False

    is_vlm = isinstance(processing_class, ProcessorMixin)
    # Distinct sentinels so we can tell which part of the exchange a template drops.
    _name_sentinel = "tool_name_a8f3e2b1"
    _arg_key_sentinel = "tool_arg_key_b9d4f5c2"
    _arg_val_sentinel = "tool_arg_val_d6e7a9f3"
    _content_sentinel = "tool_content_c4f9a8e2"
    tool_calls = [
        {
            "type": "function",
            "function": {"name": _name_sentinel, "arguments": {_arg_key_sentinel: _arg_val_sentinel}},
        }
    ]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "name": _name_sentinel, "content": _content_sentinel},
    ]
    # VLMs expect content as [{"type": "text", "text": "..."}] instead of plain strings
    if is_vlm:
        messages = prepare_multimodal_messages(messages)

    try:
        rendered = processing_class.apply_chat_template(messages, tokenize=False)
    except TemplateError:
        # TemplateError: template rejects the role sequence (Cohere, FalconMamba, Gemma, Gemma2, Gemma3)
        # UndefinedError (subclass): template indexes into content as a list for all roles, including tool
        #   (Idefics2, Idefics3, LlavaNext, SmolVLM)
        return False
    # All four sentinels must survive: the tool name and arguments (assistant tool_calls) AND the tool message
    # content. Templates that silently drop either side (basic Llama 3 drops tool_calls; Cohere2/Phi3 drop tool
    # messages) will fail this check.
    return all(s in rendered for s in (_name_sentinel, _arg_key_sentinel, _arg_val_sentinel, _content_sentinel))


def is_chat_template_prefix_preserving(processing_class: PreTrainedTokenizer | ProcessorMixin) -> bool:
    """
    Check whether the chat template preserves prefixes when applied.

    A prefix-preserving chat template renders earlier messages identically regardless of what messages follow. This
    property is required by `_get_tool_suffix_ids`, which extracts tool response formatting tokens by comparing
    tokenizations with and without tool messages appended.

    Args:
        processing_class (`PreTrainedTokenizer` or `ProcessorMixin`):
            Tokenizer or processor instance to check.

    Returns:
        `bool`:
            `True` if the chat template preserves prefixes, `False` otherwise.
    """
    # Use the same dummy messages as _get_tool_suffix_ids to test the exact property it relies on.
    dummy_tool_calls = [{"type": "function", "function": {"name": "dummy", "arguments": {}}}]
    messages1 = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "", "tool_calls": dummy_tool_calls},
    ]
    messages2 = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "", "tool_calls": dummy_tool_calls},
        {"role": "tool", "name": "dummy", "content": "dummy"},
    ]
    # VLM processors expect structured list-of-blocks content, and image-token expansion only kicks in when an image
    # is actually present, so include a dummy image to exercise the real code path.
    is_vlm = isinstance(processing_class, ProcessorMixin)
    if is_vlm:
        from PIL import Image

        dummy_image = Image.new("RGB", (8, 8))
        messages1 = prepare_multimodal_messages(messages1, images=[dummy_image])
        messages2 = prepare_multimodal_messages(messages2, images=[dummy_image])

    try:
        ids1 = processing_class.apply_chat_template(messages1, tokenize=True, return_dict=False)
        ids2 = processing_class.apply_chat_template(
            messages2, tokenize=True, return_dict=False, add_generation_prompt=True
        )
    except TypeError:
        # Best-effort fallback for templates that reject dict args (e.g. DeepSeek-V3). This is a chat template
        # bug (see transformers#45419), and the training chat template fixes it to avoid blocking users.
        dummy_tool_calls = [{"type": "function", "function": {"name": "dummy", "arguments": "{}"}}]
        messages1[1]["tool_calls"] = dummy_tool_calls
        messages2[1]["tool_calls"] = dummy_tool_calls
        ids1 = processing_class.apply_chat_template(messages1, tokenize=True, return_dict=False)
        ids2 = processing_class.apply_chat_template(
            messages2, tokenize=True, return_dict=False, add_generation_prompt=True
        )

    # VLM processors return batched output (list of lists), unbatch for single conversation
    if is_vlm:
        ids1 = ids1[0]
        ids2 = ids2[0]

    return ids2[: len(ids1)] == ids1


deepseekv3_training_chat_template = (_CHAT_TEMPLATES_DIR / "deepseekv3_training.jinja").read_text()

llama3_training_chat_template = (_CHAT_TEMPLATES_DIR / "llama3_training.jinja").read_text()

qwen2_5_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen2_5_training.jinja").read_text()

qwen3_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_training.jinja").read_text()

gptoss_training_chat_template = (_CHAT_TEMPLATES_DIR / "gptoss_training.jinja").read_text()


def get_training_chat_template(tokenizer: PreTrainedTokenizer) -> str | None:
    r"""
    Get a training-compatible chat template, if needed.

    Returns a patched chat template that is prefix-preserving and includes `{%% generation %%}` / `{%% endgeneration
    %%}` markers for assistant-only loss masking. Returns `None` if the tokenizer's template already satisfies both
    requirements. Currently DeepSeek-V3, GPT-OSS, LLaMA 3, Qwen2.5, and Qwen3 are supported.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer instance to check.

    Returns:
        `str` or `None`:
            Training-compatible chat template, or `None` if no patching is needed.

    Example:

    ```python
    >>> from trl.chat_template_utils import get_training_chat_template
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> messages1 = [
    ...     {"role": "user", "content": "What is 2 * 3?"},
    ...     {
    ...         "role": "assistant",
    ...         "content": "",
    ...         "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 2, "b": 3}}}],
    ...     },
    ... ]
    >>> messages2 = messages1 + [
    ...     {"role": "tool", "name": "multiply", "content": "6"},
    ... ]
    >>> tokenizer.apply_chat_template(messages1, tokenize=False)
    '<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n'

    >>> tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    '<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n6\n</tool_response><|im_end|>\n<|im_start|>assistant\n'

    >>> #                                                        ^ think tags missing
    >>> chat_template = get_training_chat_template(tokenizer)
    >>> tokenizer.apply_chat_template(messages1, tokenize=False, chat_template=chat_template)
    '<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n'

    >>> tokenizer.apply_chat_template(
    ...     messages2, tokenize=False, add_generation_prompt=True, chat_template=chat_template
    ... )
    '<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n6\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
    ```
    """
    # First check if patching is needed
    if is_chat_template_prefix_preserving(tokenizer) and "{% generation %}" in tokenizer.chat_template:
        return None  # No patching needed

    if tokenizer.chat_template == deepseekv3_chat_template:
        return deepseekv3_training_chat_template

    if tokenizer.chat_template == gptoss_chat_template:
        return gptoss_training_chat_template

    if tokenizer.chat_template == llama3_chat_template:
        return llama3_training_chat_template

    if tokenizer.chat_template == qwen2_5_chat_template:
        return qwen2_5_training_chat_template

    if tokenizer.chat_template == qwen3_chat_template:
        return qwen3_training_chat_template

    raise ValueError(
        "The tokenizer's chat template is not training-compatible (missing prefix-preservation or "
        "`{% generation %}` markers) and patching is not supported for this template. "
        "Please manually modify the tokenizer's chat template for training."
    )


def _validate_tool_calls(tool_calls: list | None) -> None:
    """
    Validate tool_calls to ensure all required fields exist with valid values.

    Raises ValueError when the model generates malformed tool calls (e.g., missing 'arguments' field) that are
    partially parsed.

    Args:
        tool_calls: List of tool call dictionaries, or None.
    """
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list or None.")

    for idx, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            raise ValueError(f"tool_calls[{idx}] must be a dict.")

        # Handle nested function structure: {"type": "function", "function": {"name": ..., "arguments": ...}}
        if "function" in tool_call:
            func = tool_call["function"]
            if not isinstance(func, dict):
                raise ValueError(f"tool_calls[{idx}]['function'] must be a dict.")
            if not isinstance(func.get("name"), str):
                raise ValueError(f"tool_calls[{idx}]['function']['name'] must be a string.")
            # Some templates (e.g. Qwen3.5) omit arguments for valid no-arg calls; normalize to {}.
            if "arguments" not in func or func["arguments"] is None:
                func["arguments"] = {}
        else:
            # Handle flat structure: {"name": ..., "arguments": ...}
            if not isinstance(tool_call.get("name"), str):
                raise ValueError(f"tool_calls[{idx}]['name'] must be a string.")
            # Some templates (e.g. Qwen3.5) omit arguments for valid no-arg calls; normalize to {}.
            if "arguments" not in tool_call or tool_call["arguments"] is None:
                tool_call["arguments"] = {}


def parse_response(processing_class: PreTrainedTokenizer | ProcessorMixin, ids: list[int]) -> dict:
    r"""
    Parse a token sequence into structured response dictionaries with fallback handling.

    Attempts to parse the sequence using `tokenizer.parse_response()`. If parsing fails (e.g., due to malformed tool
    calls like `<tool_call>{"type":"function"</tool_call>`), falls back to decoding as plain text.

    Also removes incorrectly appended EOS tokens from tool call content when present, and validates tool_calls to
    ensure all required fields exist.

    For VLM processors, automatically uses the inner tokenizer for parsing.

    Args:
        processing_class (`PreTrainedTokenizer` or VLM processor):
            Tokenizer or processor with a `parse_response()` method (directly or via inner tokenizer).
        ids (`list[int]`):
            List of token sequences.

    Returns:
        `dict`:
            Response dictionary.

    Example:
    ```python
    >>> from trl.chat_template_utils import parse_response, add_response_schema
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    >>> tokenizer = add_response_schema(tokenizer)  # temporary until built-in support
    >>> text = '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call><|im_end|>'
    >>> ids = tokenizer(text)["input_ids"]
    >>> parse_response(tokenizer, ids)
    {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'multiply', 'arguments': {'a': 3, 'b': 4}}}]}
    ```
    """
    # VLM processors don't have parse_response directly; use the inner tokenizer
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    try:
        parsed = tokenizer.parse_response(ids)
        # Hotfix: remove incorrectly appended EOS token from tool calls
        # See https://github.com/huggingface/transformers/issues/42249
        if isinstance(parsed.get("content"), str):
            parsed["content"] = parsed["content"].removesuffix(tokenizer.eos_token)
        # Normalize: ensure content is always a string (some models omit it or set it to None)
        if not parsed.get("content"):
            parsed["content"] = ""
        # Validate tool_calls to prevent Jinja2 Undefined errors when fields are missing
        if "tool_calls" in parsed:
            _validate_tool_calls(parsed["tool_calls"])
    except (ValueError, TypeError):
        # Fallback: decode as plain text if parsing fails. This happens if the model outputs malformed tool calls.
        content = tokenizer.decode(ids, skip_special_tokens=True)
        parsed = {"role": "assistant", "content": content}
    return parsed
