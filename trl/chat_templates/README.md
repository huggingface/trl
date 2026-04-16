# Chat Templates

Jinja2 chat templates stored here serve two purposes:

1. **Identity comparison**: detecting which model is being used (by comparing `tokenizer.chat_template` against known templates) to add the appropriate response schema (`add_response_schema`) or swap in a training template (`get_training_chat_template`).
2. **Training patches**: modified templates that fix training-specific issues (prefix-preservation for GRPO, `{% generation %}` markers for SFT assistant-only loss).

**Why prefix-preserving?** The GRPO tool call loop extracts tool response formatting tokens by comparing tokenizations with and without tool messages appended (`_get_tool_suffix_ids`). This requires the chat template to be *prefix-preserving*: appending messages must not change how earlier messages are rendered.

**Why generation-tagged?** SFT with `assistant_only_loss=True` requires the chat template to include `{% generation %}` / `{% endgeneration %}` markers around assistant output, so `return_assistant_tokens_mask=True` can produce correct masks. Most model templates don't include these markers natively.

## Original templates

Used for identity comparison only.

### `deepseekv3.jinja`

Original DeepSeek-V3 chat template.

### `glm4moe.jinja`

Original GLM-4-MoE chat template.

### `gptoss.jinja`

Original GPT-OSS chat template.

### `llama3.jinja`

Original Llama 3 chat template.

### `llama3_1.jinja` / `llama3_2.jinja`

Original Llama 3.1 / 3.2 chat templates. Both render tool calls as a single bare JSON object using the key `parameters` (instead of `arguments`) and support at most one tool call per assistant turn.

### `qwen2_5.jinja`

Original Qwen2.5 chat template.

### `qwen3.jinja`

Original Qwen3 chat template.

### `qwen3_vl.jinja`

Original Qwen3-VL chat template. Unlike text-only Qwen3, this template is already prefix-preserving (no conditional thinking blocks), so no training patch is needed.

### `qwen3_5_2b_and_below.jinja` / `qwen3_5_4b_and_above.jinja`

Original Qwen3.5 chat templates.

## Training templates

Patched templates that fix training-specific issues. Swapped in at init when tools are enabled (GRPO) or when `assistant_only_loss=True` (SFT).

### `deepseekv3_training.jinja`

Patched DeepSeek-V3 template. Diff vs `deepseekv3.jinja`:

- Uses `| tojson` on `tool['function']['arguments']` so that `arguments` can be passed as a `dict` (the documented format per [transformers docs](https://huggingface.co/docs/transformers/en/chat_extras#tool-calling-example)). The original template uses raw string concatenation, which crashes on dict inputs.
- Wraps assistant message output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss.

### `qwen3_training.jinja`

Patched Qwen3 template. Diff vs `qwen3.jinja`:

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

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `gptoss_training.jinja`

Patched GPT-OSS template. Diff vs `gptoss.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `llama3_training.jinja`

Patched Llama 3 template. Diff vs `llama3.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `qwen2_5_training.jinja`

Patched Qwen2.5 template. Diff vs `qwen2_5.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.
