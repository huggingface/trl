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


gptoss_chat_template = (_CHAT_TEMPLATES_DIR / "gptoss.jinja").read_text()

qwen3_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3.jinja").read_text()

qwen3_5_chat_template_2b_and_below = (_CHAT_TEMPLATES_DIR / "qwen3_5_2b_and_below.jinja").read_text()

qwen3_5_chat_template_4b_and_above = (_CHAT_TEMPLATES_DIR / "qwen3_5_4b_and_above.jinja").read_text()


def add_response_schema(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    r"""
    Adds the appropriate response schema to the given tokenizer based on its chat template.

    At the time of initial implementation, most tokenizers do not have built-in support for response schemas. While
    waiting for broader adoption, we provide this utility function to manually set the response schema for known chat
    templates.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer to which the response schema will be added.

    Returns:
        `PreTrainedTokenizer`:
            Tokenizer with the added response schema.

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
    if tokenizer.chat_template == gptoss_chat_template:
        tokenizer.response_schema = gptoss_schema
        return tokenizer
    if tokenizer.chat_template == qwen3_chat_template:
        tokenizer.response_schema = qwen3_schema
        return tokenizer
    if tokenizer.chat_template in [qwen3_5_chat_template_2b_and_below, qwen3_5_chat_template_4b_and_above]:
        tokenizer.response_schema = qwen3_5_schema
        return tokenizer
    raise ValueError(
        "Unrecognized chat template, failed to add response schema. Please manually set the response schema on the "
        "tokenizer or processor. See the Transformers "
        "[docs](https://huggingface.co/docs/transformers/main/en/chat_response_parsing#response-parsing) for more "
        "details on response parsing."
    )


def supports_tool_calling(processing_class) -> bool:
    """
    Check if the processing class's chat template can render a full tool-calling conversation.

    This tests two things: (1) the template doesn't error when rendering a conversation with ``user → assistant (with
    tool_calls) → tool`` roles, and (2) the tool message content actually appears in the rendered output (some
    templates silently swallow tool messages).

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
    _sentinel = "TOOL_CONTENT_c4f9a8e2"
    tool_calls = [{"type": "function", "function": {"name": "test", "arguments": {}}}]
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "name": "test", "content": _sentinel},
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
    # Some templates (e.g. Cohere2, Phi3) accept tool messages without error but silently ignore them.
    # Check that the tool content actually appears in the rendered output.
    return _sentinel in rendered


def is_chat_template_prefix_preserving(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Check whether the chat template preserves prefixes when applied.

    A prefix-preserving chat template renders earlier messages identically regardless of what messages follow. This
    property is required by `_get_tool_suffix_ids`, which extracts tool response formatting tokens by comparing
    tokenizations with and without tool messages appended.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer instance to check.

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

    text1 = tokenizer.apply_chat_template(messages1, tokenize=False)
    text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)

    return text2.startswith(text1)


qwen3_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_training.jinja").read_text()


def get_training_chat_template(tokenizer: PreTrainedTokenizer) -> str | None:
    r"""
    Get a prefix-preserving chat template for training, if needed.

    If the tokenizer's template isn't prefix-preserving, returns a training-compatible template (currently Qwen3
    supported). Otherwise, returns `None`.

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
    if is_chat_template_prefix_preserving(tokenizer):
        return None  # No patching needed

    if tokenizer.chat_template == qwen3_chat_template:
        return qwen3_training_chat_template
    else:
        raise ValueError(
            "The tokenizer's chat template is not prefix-preserving and patching is not supported for this template. "
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


def parse_response(tokenizer_or_processor, ids: list[int]) -> dict:
    r"""
    Parse a token sequence into structured response dictionaries with fallback handling.

    Attempts to parse the sequence using `tokenizer.parse_response()`. If parsing fails (e.g., due to malformed tool
    calls like `<tool_call>{"type":"function"</tool_call>`), falls back to decoding as plain text.

    Also removes incorrectly appended EOS tokens from tool call content when present, and validates tool_calls to
    ensure all required fields exist.

    For VLM processors, automatically uses the inner tokenizer for parsing.

    Args:
        tokenizer_or_processor (`PreTrainedTokenizer` or VLM processor):
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
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
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
