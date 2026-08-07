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

import re
import warnings
from pathlib import Path
from typing import TypeVar

import transformers
from jinja2 import TemplateError
from packaging.version import Version
from transformers import AddedToken, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from .data_utils import prepare_multimodal_messages


_CHAT_TEMPLATES_DIR = Path(__file__).parent / "chat_templates"

# New-style `response_template` parsing (streamable chat parsing, with the `prefix=` argument to `parse_response`)
# landed in transformers 5.13.0 (huggingface/transformers#45847). Earlier versions (>= 5.0.0) only ship the legacy
# `response_schema` parser. We gate on `5.13.0.dev0` so the feature is also active on transformers `main` (which
# reports a dev version) ahead of the 5.13.0 release.
_SUPPORTS_RESPONSE_TEMPLATE = Version(transformers.__version__) >= Version("5.13.0.dev0")


def has_generation_markers(chat_template: str) -> bool:
    """
    Check whether the chat template defines `{% generation %}` markers, accounting for whitespace-trim variants such as
    `{%- generation %}` and `{%- generation -%}`.
    """
    return re.search(r"\{%-?\s*generation\s*-?%\}", chat_template) is not None


def clone_chat_template(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_tokenizer_path: str,
    resize_to_multiple_of: int | None = 64,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, list[int]]:
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
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer to update.
        source_tokenizer_path (`str`):
            Path or identifier of the pretrained tokenizer to clone from.
        resize_to_multiple_of (`int` or `None`, *optional*, defaults to `64`):
            The embedding layer will be resized to the new vocabulary size. If this is not `None`, it will round up the
            new vocabulary size to the nearest multiple of this value.

    Returns:
        model ([`~transformers.PreTrainedModel`]):
            Updated model with resized token embeddings and EOS token configured.
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Updated tokenizer with the chat template and special tokens applied.
        added_tokens (`list[int]`):
            List of tokens that were added to the tokenizer from the source tokenizer.

    Example:
    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from trl import clone_chat_template

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    >>> model, tokenizer, added_tokens = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")
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
    "x-regex": r"^(?:\n?<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>(?:(?!<tool_call>)[\s\S])*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|$))(?P<tool_calls>(?:<tool_call>(?:(?!</tool_call>)[\s\S])+</tool_call>\s*)+)?$",
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
    "x-regex": r"^(?:<think>\n?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>(?:(?!<tool_call>)[\s\S])*?)(?:\n(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>(?:(?!</tool_call>)[\s\S])+</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)",
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
    "x-regex": r"^(?:(?:<think>\n?)?(?:(?P<reasoning_content>.*?\S.*?)\n?|[\s]*)</think>\s*)?(?P<content>(?:(?!<tool_call>)[\s\S])*?)(?:\n+(?=<tool_call>))?(?=(?:<tool_call>|<\|im_end\|>|$))(?P<tool_calls>(?:<tool_call>(?:(?!</tool_call>)[\s\S])+</tool_call>\s*)+)?\s*(?:<\|im_end\|>|$)",
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


# New-style response templates (transformers >= 5.13 with PR huggingface/transformers#45847). These coexist with the
# legacy `*_schema` dicts above; `add_response_schema` sets the template on transformers >= 5.13 and the schema on
# older versions.
qwen3_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open": "<tool_call>",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {
            "close_pattern": r"<\|im_end\|>\s*",
            "content": "text",
        },
    },
}

qwen3_5_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^\n>]+)>",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"<\|im_end\|>\s*",
            "content": "text",
        },
    },
}

glm4moe_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|assistant|>",
    "fields": {
        "reasoning_content": {
            "open": "<think>",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<tool_call>(?P<name>\S+)\n?",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<arg_key>(?P<key>[^<]+)</arg_key>\s*\n<arg_value>(?P<value>.*?)</arg_value>",
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"<\|user\|>\s*|$",
            "content": "text",
        },
    },
}

llama3_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "fields": {
        "tool_calls": {
            "open_pattern": r'\{"name":\s*"(?P<name>[^"]+)",\s*"parameters":\s*',
            "close_pattern": r"\}<\|eot_id\|>\s*",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"<\|eot_id\|>\s*",
            "content": "text",
        },
    },
}

gptoss_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|start|>assistant",
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            "close_pattern": r"<\|end\|>\s*",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            "close_pattern": r"<\|return\|>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"to=functions\.(?P<name>\S+?)<\|channel\|>commentary\s+json<\|message\|>",
            "close_pattern": r"<\|call\|>\s*",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
    },
}

# An LFM2.5 argument value is either a single-quoted string, a JSON list/mapping, or a bare literal running up to the
# next `,` or `)`. Lists and mappings contain those very separators, so the pattern has to consume balanced brackets,
# which a regex can only do up to a fixed nesting depth. Three levels covers realistic tool arguments; deeper values
# truncate.
_LFM2_2_5_ATOM = r"[^\[\]{}'\"]|'[^']*'|\"[^\"]*\""
_LFM2_2_5_NESTED = _LFM2_2_5_ATOM
for _ in range(3):
    _LFM2_2_5_NESTED = rf"{_LFM2_2_5_ATOM}|[\[{{](?:{_LFM2_2_5_NESTED})*[\]}}]"
_LFM2_2_5_ARG_VALUE = rf"'[^']*'|[\[{{](?:{_LFM2_2_5_NESTED})*[\]}}]|[^,)]*"

lfm2_2_5_template = {
    # LFM2.5 renders every tool call of a turn into a single `<|tool_call_start|>[...]<|tool_call_end|>` block, as a
    # comma-separated list of Python-style calls: `[get_weather(city='Paris', days=3), ping()]`. The `open_pattern`
    # therefore matches either the opening bracket or the comma separating two calls, and `close_pattern` matches the
    # closing paren plus, for the last call, the trailing `]<|tool_call_end|>`.
    #
    # Argument values are rendered by the template's `format_arg_value` macro: strings single-quoted, mappings as JSON,
    # everything else via Jinja's `string` filter. `string_delims` turns the single-quoted strings back into JSON
    # strings so they parse. Note that this round-trip is lossy for Python literals the `string` filter emits but JSON
    # cannot express: `flag=False` parses back as the string `"False"`, not `False`. That's inherent to the upstream
    # template, not something we can recover here.
    #
    # The thinking field is named `thinking` (not `reasoning_content` as in other families) to match what the LFM2.5
    # template itself reads back off the message.
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {
            "open": "<think>",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"(?:<\|tool_call_start\|>\[|,\s*)(?P<name>\w+)\(",
            "close_pattern": r"\)(?:\]<\|tool_call_end\|>)?\s*",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": rf"(?P<key>\w+)=(?P<value>{_LFM2_2_5_ARG_VALUE})",
                "value_parser": {"name": "json", "args": {"string_delims": [["'", "'"]], "allow_non_json": True}},
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"<\|im_end\|>\s*",
            "content": "text",
        },
    },
}

nemotron_3_template = {
    # Nemotron 3 always prefills the assistant turn with the `<think>` opener (and a trailing `\n` when thinking is
    # enabled, or `</think>` when it is not), so the start anchor consumes it. Reasoning content is therefore not framed
    # by a `<think>` open in the model output; the zero-width `open_pattern` only activates the reasoning field when a
    # closing `</think>` is actually present, so a turn with no reasoning is parsed entirely as content. Tool calls use
    # the same `<function=...>` / `<parameter=...>` XML as Qwen3.5.
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"<\|im_start\|>assistant\n<think>(?:\n|</think>)?",
    "fields": {
        "reasoning_content": {
            "open_pattern": r"(?=[\s\S]*?</think>)",
            "close_pattern": r"</think>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>[^\n>]+)>",
            "close_pattern": r"</tool_call>\s*",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
                "value_parser": {"name": "json", "args": {"allow_non_json": True}},
            },
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"<\|im_end\|>\s*",
            "content": "text",
        },
    },
}

# Gemma 4 frames each field with its own sentinels (`<|channel>thought\n...<channel|>`, one
# `<|tool_call>call:<name>{...}<tool_call|>` block per call, content up to `<turn|>`/`<|tool_response>`/`<eos>`). Tool
# arguments are JSON with unquoted keys and `<|"|>`-delimited strings, hence the `json` parser config. Mirrors the
# new-style `response_template` recent Gemma 4 repos ship natively, but with `\s*`-tolerant close patterns.
gemma4_template = {
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"(?:<\|turn>model\n|<tool_response\|>)",
    "fields": {
        "reasoning_content": {
            "open": "<|channel>thought\n",
            "close_pattern": r"<channel\|>\s*",
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "close_pattern": r"<tool_call\|>\s*",
            "repeats": True,
            "content": "json",
            "content_args": {"string_delims": [['<|"|>', '<|"|>']], "unquoted_keys": True},
            "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}},
        },
        "content": {
            "close_pattern": r"(?:<turn\|>|<\|tool_response>|<eos>)\s*|$",
            "content": "text",
        },
    },
}


cohere_chat_template = (_CHAT_TEMPLATES_DIR / "cohere.jinja").read_text(encoding="utf-8")

cohere2_chat_template = (_CHAT_TEMPLATES_DIR / "cohere2.jinja").read_text(encoding="utf-8")

deepseekv3_chat_template = (_CHAT_TEMPLATES_DIR / "deepseekv3.jinja").read_text(encoding="utf-8")

diffusion_gemma_chat_template = (_CHAT_TEMPLATES_DIR / "diffusion_gemma.jinja").read_text(encoding="utf-8")

gemma_chat_template = (_CHAT_TEMPLATES_DIR / "gemma.jinja").read_text(encoding="utf-8")

gemma3_chat_template = (_CHAT_TEMPLATES_DIR / "gemma3.jinja").read_text(encoding="utf-8")

gemma4_chat_template = (_CHAT_TEMPLATES_DIR / "gemma4.jinja").read_text(encoding="utf-8")

glm4moe_chat_template = (_CHAT_TEMPLATES_DIR / "glm4moe.jinja").read_text(encoding="utf-8")

gptoss_chat_template = (_CHAT_TEMPLATES_DIR / "gptoss.jinja").read_text(encoding="utf-8")

idefics3_chat_template = (_CHAT_TEMPLATES_DIR / "idefics3.jinja").read_text(encoding="utf-8")

lfm2_chat_template = (_CHAT_TEMPLATES_DIR / "lfm2.jinja").read_text(encoding="utf-8")

lfm2_2_5_chat_template = (_CHAT_TEMPLATES_DIR / "lfm2_2_5.jinja").read_text(encoding="utf-8")

llama3_chat_template = (_CHAT_TEMPLATES_DIR / "llama3.jinja").read_text(encoding="utf-8")

llama3_1_chat_template = (_CHAT_TEMPLATES_DIR / "llama3_1.jinja").read_text(encoding="utf-8")

llama3_2_chat_template = (_CHAT_TEMPLATES_DIR / "llama3_2.jinja").read_text(encoding="utf-8")

llava_next_chat_template = (_CHAT_TEMPLATES_DIR / "llava_next.jinja").read_text(encoding="utf-8")

nemotron_3_nano_chat_template = (_CHAT_TEMPLATES_DIR / "nemotron_3_nano.jinja").read_text(encoding="utf-8")

nemotron_3_super_chat_template = (_CHAT_TEMPLATES_DIR / "nemotron_3_super.jinja").read_text(encoding="utf-8")

nemotron_3_ultra_chat_template = (_CHAT_TEMPLATES_DIR / "nemotron_3_ultra.jinja").read_text(encoding="utf-8")

phi3_chat_template = (_CHAT_TEMPLATES_DIR / "phi3.jinja").read_text(encoding="utf-8")

phi3_5_chat_template = (_CHAT_TEMPLATES_DIR / "phi3_5.jinja").read_text(encoding="utf-8")

qwen2_5_chat_template = (_CHAT_TEMPLATES_DIR / "qwen2_5.jinja").read_text(encoding="utf-8")

# Also matches Qwen2-VL, which ships a byte-identical chat template.
qwen2_5_vl_chat_template = (_CHAT_TEMPLATES_DIR / "qwen2_5_vl.jinja").read_text(encoding="utf-8")

qwen3_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3.jinja").read_text(encoding="utf-8")

qwen3_instruct_2507_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_instruct_2507.jinja").read_text(encoding="utf-8")

qwen3_vl_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_vl.jinja").read_text(encoding="utf-8")

qwen3_5_nothink_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_5_nothink.jinja").read_text(encoding="utf-8")

qwen3_5_think_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_5_think.jinja").read_text(encoding="utf-8")

qwen3_6_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_6.jinja").read_text(encoding="utf-8")


ProcessingClassT = TypeVar("ProcessingClassT", PreTrainedTokenizerBase, ProcessorMixin)


def add_response_schema(processing_class: ProcessingClassT) -> ProcessingClassT:
    r"""
    Adds the appropriate response template (or legacy schema) to the given tokenizer based on its chat template.

    At the time of initial implementation, most tokenizers do not have built-in support for response parsing. While
    waiting for broader adoption, we provide this utility function to manually set it for known chat templates. On
    transformers >= 5.13 the new-style `response_template` is set; on older versions (which only understand it) the
    legacy `response_schema` is set instead.

    When given a VLM processor, it is set on the inner tokenizer, since `parse_response` is a tokenizer method that
    reads `self.response_template` / `self.response_schema` from the tokenizer instance.

    Args:
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`):
            Tokenizer or VLM processor to which the response template or schema will be added.

    Returns:
        `PreTrainedTokenizerBase` or `ProcessorMixin`:
            The same object that was passed in, with the response template or schema set on the underlying tokenizer.

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
    # For VLM processors, set it on the inner tokenizer (where `parse_response` reads it from). Match against the
    # top-level chat_template, since that's what was used historically and processors may carry their own VLM-specific
    # template separate from the inner tokenizer's.
    chat_template = processing_class.chat_template
    if isinstance(processing_class, ProcessorMixin):
        tokenizer = processing_class.tokenizer
    else:
        tokenizer = processing_class
    if chat_template == glm4moe_chat_template:
        schema, template = glm4moe_schema, glm4moe_template
    elif chat_template == gptoss_chat_template:
        schema, template = gptoss_schema, gptoss_template
    elif chat_template in [llama3_1_chat_template, llama3_2_chat_template]:
        schema, template = llama3_schema, llama3_template
    elif chat_template in [
        qwen2_5_chat_template,
        qwen3_chat_template,
        qwen3_instruct_2507_chat_template,
        qwen3_vl_chat_template,
    ]:
        schema, template = qwen3_schema, qwen3_template
    elif chat_template in [
        qwen3_5_nothink_chat_template,
        qwen3_5_think_chat_template,
        qwen3_6_chat_template,
    ]:
        schema, template = qwen3_5_schema, qwen3_5_template
    elif chat_template in [
        nemotron_3_nano_chat_template,
        nemotron_3_super_chat_template,
        nemotron_3_ultra_chat_template,
    ]:
        schema, template = qwen3_5_schema, nemotron_3_template
    elif chat_template == lfm2_2_5_chat_template:
        # Only the new-style template; the legacy schema is on its way out, so it isn't worth adding for a family whose
        # tokenizer already requires transformers >= 5.0.0.
        schema, template = None, lfm2_2_5_template
    elif chat_template == gemma4_chat_template:
        # Only the new-style template; recent Gemma 4 repos ship a `response_template` natively, and the legacy
        # `response_schema` is being removed upstream (huggingface/transformers#47320).
        schema, template = None, gemma4_template
    else:
        raise ValueError(
            "Unrecognized chat template, failed to add response schema. Please manually set the response schema on "
            "the tokenizer or processor. See the Transformers "
            "[docs](https://huggingface.co/docs/transformers/main/en/chat_response_parsing#response-parsing) for more "
            "details on response parsing."
        )
    # New-style `response_template` is preferred where supported (transformers >= 5.13); older versions only understand
    # the legacy `response_schema`. Set exactly the one the installed transformers can use, so the presence of the
    # attribute reflects the parser actually in play.
    if _SUPPORTS_RESPONSE_TEMPLATE:
        tokenizer.response_template = template
    elif schema is None:
        raise ValueError(
            f"Response parsing for this chat template is only provided as a new-style `response_template`, which "
            f"requires transformers >= 5.13 (installed: {transformers.__version__}). Please upgrade transformers, or "
            f"manually set a legacy `response_schema` on the tokenizer or processor."
        )
    else:
        tokenizer.response_schema = schema
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
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`):
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
    except TypeError:
        # Best-effort fallback for templates that reject dict args (e.g. DeepSeek-V3). This is a chat template
        # bug (see transformers#45419), and the training chat template fixes it to avoid blocking users.
        tool_calls[0]["function"]["arguments"] = f'{{"{_arg_key_sentinel}": "{_arg_val_sentinel}"}}'
        try:
            rendered = processing_class.apply_chat_template(messages, tokenize=False)
        except TemplateError:
            return False
    # All four sentinels must survive: the tool name and arguments (assistant tool_calls) AND the tool message
    # content. Templates that silently drop either side (basic Llama 3 drops tool_calls; Cohere2/Phi3 drop tool
    # messages) will fail this check.
    return all(s in rendered for s in (_name_sentinel, _arg_key_sentinel, _arg_val_sentinel, _content_sentinel))


def is_chat_template_prefix_preserving(processing_class: PreTrainedTokenizerBase | ProcessorMixin) -> bool:
    """
    Check whether the chat template preserves prefixes when applied.

    A prefix-preserving chat template renders earlier messages identically regardless of what messages follow. This
    property is required by `_get_tool_suffix_ids`, which extracts tool response formatting tokens by comparing
    tokenizations with and without tool messages appended.

    Args:
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`):
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


def is_chat_template_stop_token_trained(
    processing_class: PreTrainedTokenizerBase | ProcessorMixin, chat_template: str | None = None
) -> bool:
    """
    Check whether the chat template includes an assistant turn's end-of-turn token in the loss mask.

    Prefix preservation guarantees that earlier turns render identically, but not that the token the model must emit to
    *end* its turn is part of the loss. Some templates attribute an assistant turn's end-of-turn token to the message
    that follows it, so when masking with `return_assistant_tokens_mask=True` that token falls outside the assistant
    span and the model is never trained to stop. This renders an assistant turn followed by a user message and checks
    that the assistant's masked span ends on an end-of-turn token rather than on content.

    The template must define `{% generation %}` / `{% endgeneration %}` markers (see [`get_training_chat_template`]),
    otherwise the assistant mask is empty and this returns `False`.

    Args:
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`):
            Tokenizer or processor instance to check.
        chat_template (`str`, *optional*):
            Chat template to check. Defaults to the one attached to `processing_class`.

    Returns:
        `bool`:
            `True` if the assistant turn's end-of-turn token is included in the loss mask, `False` otherwise.
    """
    # The assistant turn is followed by a user message because some templates never terminate the final assistant
    # turn; the boundary with the next message is where the end-of-turn token must be attributed.
    messages = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "dummy"},
        {"role": "user", "content": "dummy"},
    ]
    is_vlm = isinstance(processing_class, ProcessorMixin)
    if is_vlm:
        # Probe without images: assistant masks are computed before multimodal token expansion and not re-aligned
        # afterwards, so any image would zero or shift the mask regardless of what the template attributes.
        for message in messages:
            message["content"] = [{"type": "text", "text": message["content"]}]

    try:
        output = processing_class.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            chat_template=chat_template,
        )
    except (TemplateError, TypeError, ValueError):
        return False

    input_ids = output["input_ids"]
    assistant_masks = output["assistant_masks"]
    if is_vlm:
        input_ids = input_ids[0]
        assistant_masks = assistant_masks[0]

    # The model stops by emitting an end-of-turn token, which is part of the added vocabulary rather than produced by
    # the base tokenizer's merges. Ignoring trailing whitespace, the last masked token must be that terminator; if it
    # is plain content, the end-of-turn token was attributed to the following message and is never trained.
    tokenizer = processing_class.tokenizer if is_vlm else processing_class
    added_ids = set(tokenizer.get_added_vocab().values())
    masked_ids = [token_id for token_id, masked in zip(input_ids, assistant_masks, strict=False) if masked]
    for token_id in reversed(masked_ids):
        if tokenizer.decode([token_id]).strip() == "":
            continue
        return token_id in added_ids
    return False


cohere_training_chat_template = (_CHAT_TEMPLATES_DIR / "cohere_training.jinja").read_text(encoding="utf-8")

cohere2_training_chat_template = (_CHAT_TEMPLATES_DIR / "cohere2_training.jinja").read_text(encoding="utf-8")

deepseekv3_training_chat_template = (_CHAT_TEMPLATES_DIR / "deepseekv3_training.jinja").read_text(encoding="utf-8")

diffusion_gemma_training_chat_template = (_CHAT_TEMPLATES_DIR / "diffusion_gemma_training.jinja").read_text(
    encoding="utf-8"
)

gemma_training_chat_template = (_CHAT_TEMPLATES_DIR / "gemma_training.jinja").read_text(encoding="utf-8")

gemma3_training_chat_template = (_CHAT_TEMPLATES_DIR / "gemma3_training.jinja").read_text(encoding="utf-8")

glm4moe_training_chat_template = (_CHAT_TEMPLATES_DIR / "glm4moe_training.jinja").read_text(encoding="utf-8")

gptoss_training_chat_template = (_CHAT_TEMPLATES_DIR / "gptoss_training.jinja").read_text(encoding="utf-8")

idefics3_training_chat_template = (_CHAT_TEMPLATES_DIR / "idefics3_training.jinja").read_text(encoding="utf-8")

lfm2_training_chat_template = (_CHAT_TEMPLATES_DIR / "lfm2_training.jinja").read_text(encoding="utf-8")

llama3_training_chat_template = (_CHAT_TEMPLATES_DIR / "llama3_training.jinja").read_text(encoding="utf-8")

llava_next_training_chat_template = (_CHAT_TEMPLATES_DIR / "llava_next_training.jinja").read_text(encoding="utf-8")

nemotron_3_nano_training_chat_template = (_CHAT_TEMPLATES_DIR / "nemotron_3_nano_training.jinja").read_text(
    encoding="utf-8"
)

nemotron_3_super_training_chat_template = (_CHAT_TEMPLATES_DIR / "nemotron_3_super_training.jinja").read_text(
    encoding="utf-8"
)

nemotron_3_ultra_training_chat_template = (_CHAT_TEMPLATES_DIR / "nemotron_3_ultra_training.jinja").read_text(
    encoding="utf-8"
)

phi3_training_chat_template = (_CHAT_TEMPLATES_DIR / "phi3_training.jinja").read_text(encoding="utf-8")

phi3_5_training_chat_template = (_CHAT_TEMPLATES_DIR / "phi3_5_training.jinja").read_text(encoding="utf-8")

qwen2_5_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen2_5_training.jinja").read_text(encoding="utf-8")

qwen2_5_vl_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen2_5_vl_training.jinja").read_text(encoding="utf-8")

qwen3_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_training.jinja").read_text(encoding="utf-8")

qwen3_instruct_2507_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_instruct_2507_training.jinja").read_text(
    encoding="utf-8"
)

qwen3_vl_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_vl_training.jinja").read_text(encoding="utf-8")

qwen3_5_nothink_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_5_nothink_training.jinja").read_text(
    encoding="utf-8"
)

qwen3_5_think_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_5_think_training.jinja").read_text(
    encoding="utf-8"
)

qwen3_6_training_chat_template = (_CHAT_TEMPLATES_DIR / "qwen3_6_training.jinja").read_text(encoding="utf-8")


def get_training_chat_template(
    processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str | None:
    r"""
    Get a training-compatible chat template, if needed.

    Returns a patched chat template that is prefix-preserving and includes `{%% generation %%}` / `{%% endgeneration
    %%}` markers for assistant-only loss masking. Returns `None` if the template already satisfies both requirements.
    Currently Cohere, Cohere 2, DeepSeek-V3, Gemma, Gemma 2, Gemma 3, GLM-4-MoE, GPT-OSS, Idefics3, LFM2, LLaMA 3,
    Phi-3, Phi-3.5, Qwen2-VL, Qwen2.5, Qwen2.5-VL, Qwen3 (including the Instruct-2507 variant), Qwen3-VL, Qwen3.5, and
    Qwen3.6 are supported.

    Args:
        processing_class (`PreTrainedTokenizerBase` or `ProcessorMixin`):
            Tokenizer or processor instance to check.

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
    if tokenizer is not None:
        if processing_class is not None:
            raise TypeError(
                "Pass only `processing_class`; `tokenizer` is a deprecated alias for backward compatibility."
            )
        warnings.warn(
            "The `tokenizer` argument of `get_training_chat_template` is deprecated and will be removed in TRL 2.0. "
            "Use `processing_class` instead.",
            FutureWarning,
            stacklevel=2,
        )
        processing_class = tokenizer
    if processing_class is None:
        raise TypeError("get_training_chat_template() missing required argument: 'processing_class'")

    # First check if patching is needed. Prefix-preservation only matters when the template actually supports tools
    # (the check itself renders a tool message), so skip it otherwise.
    prefix_ok = not supports_tool_calling(processing_class) or is_chat_template_prefix_preserving(processing_class)
    if prefix_ok and has_generation_markers(processing_class.chat_template):
        return None  # No patching needed

    if processing_class.chat_template == cohere_chat_template:
        return cohere_training_chat_template

    if processing_class.chat_template == cohere2_chat_template:
        return cohere2_training_chat_template

    if processing_class.chat_template == deepseekv3_chat_template:
        return deepseekv3_training_chat_template

    if processing_class.chat_template == diffusion_gemma_chat_template:
        return diffusion_gemma_training_chat_template

    if processing_class.chat_template == gemma_chat_template:
        return gemma_training_chat_template

    if processing_class.chat_template == gemma3_chat_template:
        return gemma3_training_chat_template

    if processing_class.chat_template == glm4moe_chat_template:
        return glm4moe_training_chat_template

    if processing_class.chat_template == gptoss_chat_template:
        return gptoss_training_chat_template

    if processing_class.chat_template == idefics3_chat_template:
        return idefics3_training_chat_template

    if processing_class.chat_template == lfm2_chat_template:
        return lfm2_training_chat_template

    if processing_class.chat_template == llama3_chat_template:
        return llama3_training_chat_template

    if processing_class.chat_template == llava_next_chat_template:
        return llava_next_training_chat_template

    if processing_class.chat_template == nemotron_3_nano_chat_template:
        return nemotron_3_nano_training_chat_template

    if processing_class.chat_template == nemotron_3_super_chat_template:
        return nemotron_3_super_training_chat_template

    if processing_class.chat_template == nemotron_3_ultra_chat_template:
        return nemotron_3_ultra_training_chat_template

    if processing_class.chat_template == phi3_chat_template:
        return phi3_training_chat_template

    if processing_class.chat_template == phi3_5_chat_template:
        return phi3_5_training_chat_template

    if processing_class.chat_template == qwen2_5_chat_template:
        return qwen2_5_training_chat_template

    if processing_class.chat_template == qwen2_5_vl_chat_template:
        return qwen2_5_vl_training_chat_template

    if processing_class.chat_template == qwen3_chat_template:
        return qwen3_training_chat_template

    if processing_class.chat_template == qwen3_instruct_2507_chat_template:
        return qwen3_instruct_2507_training_chat_template

    if processing_class.chat_template == qwen3_vl_chat_template:
        return qwen3_vl_training_chat_template

    if processing_class.chat_template == qwen3_5_nothink_chat_template:
        return qwen3_5_nothink_training_chat_template

    if processing_class.chat_template == qwen3_5_think_chat_template:
        return qwen3_5_think_training_chat_template

    if processing_class.chat_template == qwen3_6_chat_template:
        return qwen3_6_training_chat_template

    raise ValueError(
        "The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) "
        "and patching is not supported for this template. Please manually modify the chat template for training."
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


def parse_response(tokenizer: PreTrainedTokenizerBase, ids: list[int], prefix: list[int] | None = None) -> dict:
    r"""
    Parse a token sequence into structured response dictionaries with fallback handling.

    Attempts to parse the sequence using `tokenizer.parse_response()`. If parsing fails (e.g., due to malformed tool
    calls like `<tool_call>{"type":"function"</tool_call>`), falls back to decoding as plain text.

    Also removes incorrectly appended EOS tokens from tool call content when present, and validates tool_calls to
    ensure all required fields exist.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer with a `parse_response()` method.
        ids (`list[int]`):
            List of token sequences.
        prefix (`list[int]`, *optional*):
            Token IDs of the chat-prompt context that came before `ids` (e.g. the rendered chat template up to and
            including the assistant header / any template-emitted thinking opener). Only used when the tokenizer has a
            new-style `response_template` set and the installed transformers supports it (>= 5.13); ignored for legacy
            `response_schema`.

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
        # `prefix=` is only supported by the new-style parser (transformers >= 5.13, when a `response_template` is set).
        # Older transformers only ship the legacy `response_schema` parser, whose `parse_response` rejects the kwarg, so
        # we gate on `_SUPPORTS_RESPONSE_TEMPLATE` (matching `add_response_schema`, which only sets `response_template`
        # on those versions) and omit `prefix=` everywhere else, including the legacy `response_schema` path.
        if (
            prefix is not None
            and _SUPPORTS_RESPONSE_TEMPLATE
            and getattr(tokenizer, "response_template", None) is not None
        ):
            parsed = tokenizer.parse_response(ids, prefix=prefix)
        else:
            parsed = tokenizer.parse_response(ids)
        if parsed is None:  # this can happen if the response is heavily truncated and even the content is lost
            raise ValueError("parse_response returned None")
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
