# Chat Templates

Jinja2 chat templates stored here serve two purposes:

1. **Identity comparison**: detecting which model is being used (by comparing `tokenizer.chat_template` against known templates) to add the appropriate response schema (`add_response_schema`) or swap in a training template (`get_training_chat_template`).
2. **Training patches**: modified templates that fix prefix-preservation issues for the GRPO tool call loop.

**Why prefix-preserving?** The GRPO tool call loop extracts tool response formatting tokens by comparing tokenizations with and without tool messages appended (`_get_tool_suffix_ids`). This requires the chat template to be *prefix-preserving*: appending messages must not change how earlier messages are rendered.

## Original templates

Used for identity comparison only.

### `gptoss.jinja`

Original GPT-OSS chat template.

### `qwen3.jinja`

Original Qwen3 chat template.

### `qwen3_5_2b_and_below.jinja` / `qwen3_5_4b_and_above.jinja`

Original Qwen3.5 chat templates.

## Training templates

Patched templates that fix prefix-preservation issues. Swapped in at init when tools are enabled.

### `qwen3_training.jinja`

Patched Qwen3 template that always includes thinking blocks, making it prefix-preserving. Diff vs `qwen3.jinja`:

Require both `<think>` and `</think>` to be present before parsing, to avoid incorrect splitting when the model generates only one tag:

```diff
- {%- if '</think>' in content %}
+ {%- if '<think>' in content and '</think>' in content %}
```

Always include the thinking block regardless of message position. The original conditionally omits it based on `loop.last`, which changes the assistant rendering when a tool message is appended — breaking prefix-preservation:

```diff
- {%- if loop.index0 > ns.last_query_index %}
-     {%- if loop.last or (not loop.last and reasoning_content) %}
-         {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
-     {%- else %}
-         {{- '<|im_start|>' + message.role + '\n' + content }}
-     {%- endif %}
- {%- else %}
-     {{- '<|im_start|>' + message.role + '\n' + content }}
- {%- endif %}
+ {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
```
