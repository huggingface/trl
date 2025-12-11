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

from transformers import AddedToken, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


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


# Adapted and corrected versions of the schemas from:
# https://github.com/huggingface/transformers/blob/main/tests/utils/test_chat_parsing_utils.py
qwen3_schema = {
    "x-regex": r"^(?:<think>\n?(?P<reasoning_content>.+?)\n?</think>\s*)?(?P<content>.*?)(?=(?:<tool_call>|<\|im_end\|>|$))(?:<tool_call>(?P<tool_calls>.+?)</tool_call>)?\s*(?:<\|im_end\|>|$)",
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {"type": "string"},
        "reasoning_content": {"type": "string"},
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
    if tokenizer.chat_template == qwen3_chat_template:
        tokenizer.response_schema = qwen3_schema
        return tokenizer
    raise ValueError(
        "Unrecognized chat template, failed to add response schema. Please manually set the response schema on the "
        "tokenizer or processor. See the Transformers "
        "[docs](https://huggingface.co/docs/transformers/main/en/chat_response_parsing#response-parsing) for more "
        "details on response parsing."
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


# Modifications:
# - {%- if '</think>' in content %}
# + {%- if '<think>' in content and '</think>' in content %}
#   Always check for both tags to avoid edge cases where the model generates only one tag, which would otherwise be parsed incorrectly
# - {%- if loop.index0 > ns.last_query_index %} ... {%- endif %}
# + {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
#   Always include thinking block during training. It's important to have a prefix-preserving template.
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
            {%- if '<think>' in content and '</think>' in content %}
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


def get_training_chat_template(tokenizer: PreTrainedTokenizer) -> str | None:
    r"""
    Get a prefix-preserving chat template for training, if needed.

    If the tokenizer's template isn't prefix-preserving, returns a training-compatible template (currently only Qwen3
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
    ...     {"role": "user", "content": "What color is the sky?"},
    ...     {"role": "assistant", "content": "It is blue."},
    ... ]
    >>> messages2 = [
    ...     {"role": "user", "content": "What color is the sky?"},
    ...     {"role": "assistant", "content": "It is blue."},
    ...     {"role": "user", "content": "And at night?"},
    ... ]
    >>> tokenizer.apply_chat_template(messages1, tokenize=False)
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nIt is blue.<|im_end|>\n'

    >>> tokenizer.apply_chat_template(messages2, tokenize=False)
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\nIt is blue.<|im_end|>\n<|im_start|>user\nAnd at night?<|im_end|>\n'

    >>> #                                                                       ^ think tags missing
    >>> chat_template = get_training_chat_template(tokenizer)
    >>> tokenizer.apply_chat_template(messages1, tokenize=False, chat_template=chat_template)
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nIt is blue.<|im_end|>\n'

    >>> tokenizer.apply_chat_template(messages2, tokenize=False, chat_template=chat_template)
    '<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nIt is blue.<|im_end|>\n<|im_start|>user\nAnd at night?<|im_end|>\n'
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


def parse_response(tokenizer: PreTrainedTokenizer, ids: list[int]) -> dict:
    r"""
    Parse a token sequence into structured response dictionaries with fallback handling.

    Attempts to parse the sequence using `tokenizer.parse_response()`. If parsing fails (e.g., due to malformed tool
    calls like `<tool_call>{"type":"function"</tool_call>`), falls back to decoding as plain text.

    Also removes incorrectly appended EOS tokens from tool call content when present.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer with a `parse_response()` method.
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
    try:
        parsed = tokenizer.parse_response(ids)
        # Hotfix: remove incorrectly appended EOS token from tool calls
        # See https://github.com/huggingface/transformers/issues/42249
        parsed["content"] = parsed["content"].removesuffix(tokenizer.eos_token)
    except ValueError:
        # Fallback: decode as plain text if parsing fails. This happens if the model outputs malformed tool calls.
        content = tokenizer.decode(ids, skip_special_tokens=True)
        parsed = {"role": "assistant", "content": content}
    return parsed
