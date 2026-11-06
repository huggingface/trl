# Chat Templates

A [chat template](https://huggingface.co/docs/transformers/en/chat_templating) is a Jinja2 snippet that formats messages into the string a model was trained on. For example:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
>>> tokenizer.chat_template
"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
>>> tokenizer.apply_chat_template([{"role": "user", "content": "Hi!"}], tokenize=False)
'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n'
```

In most cases you don't need to worry about chat templates: models ship their template along with the tokenizer, and TRL applies it for you. The whole thing is transparent. But some TRL recipes rely on features that most shipped templates don't include:

- **SFT with `assistant_only_loss=True`** needs `{% generation %}` / `{% endgeneration %}` markers around assistant output, so the loss mask can target only assistant tokens.
- **GRPO with tool calls** needs the template to be *prefix-preserving*: appending a tool message must not change how earlier messages are rendered.

TRL ships patched templates under [`trl/chat_templates/`](https://github.com/huggingface/trl/tree/main/trl/chat_templates) for common families (Qwen, Llama, DeepSeek-V3, GPT-OSS, ...) and swaps them in automatically for supported models. For any other model, you'll need to patch its template yourself. The rest of this page catalogs what's bundled.

## Supported model families

TRL stores reference copies of the original templates so it can identify supported models at init and swap in a training template when needed. The following families are recognized: Cohere, Cohere2, DeepSeek-V3 (including DeepSeek-R1, DeepSeek-R1-0528 and DeepSeek-V3-0324), DeepSeek-R1-Distill, DiffusionGemma, Gemma, Gemma3, Gemma4 (including Gemma-4-12B, Gemma-4-26B-A4B and Gemma-4-31B), GLM-4-MoE (4.5, 4.6, 4.7), GLM-5, GLM-5.1, GLM-5.2, GLM-5.3, GLM-5.3-Flash, GPT-OSS, Idefics3, LFM2, LFM2.5 (including LFM2.5-1.2B-Instruct, LFM2.5-2.6B and LFM2.5-24B-A2B), LFM2.5-VL (including LFM2.5-VL-450M), Llama 3 / 3.1 / 3.2, Llava-Next, Mistral Large 4, Mistral Medium 3.5, Muse Glimmer, Nemotron 3 (Nano, Super, Ultra), Nemotron 3.5 Lightning, Phi-3, Phi-3.5, Qwen2-VL, Qwen2.5 (including the base checkpoints), Qwen2.5-VL, Qwen3 (including the Instruct-2507 and Thinking-2507 variants), Qwen3-Coder (including Qwen3-Coder-Next), Qwen3-VL, Qwen3.5, Qwen3.6, Qwen3.8, SmolVLM (including SmolVLM2).

## Training templates

Patched templates that fix training-specific issues. Swapped in at init when tools are enabled (GRPO) or when `assistant_only_loss=True` (SFT).

### `cohere_training.jinja`

Patched Cohere template. Diff vs `cohere.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `cohere2_training.jinja`

Patched Cohere2 template. Diff vs `cohere2.jinja`:

Move the trailing `<|END_OF_TURN_TOKEN|>` from after the role-dispatch `{% endif %}` into each role branch, and wrap the assistant branch (`<|START_RESPONSE|>...<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`) with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `deepseek_r1_distill_training.jinja`

Patched DeepSeek-R1-Distill template. Diff vs `deepseek_r1_distill.jinja`:

- Removes the `'</think>'` split on assistant content. The original drops everything up to and including `</think>`, which would silently remove the reasoning from the training target.
- Wraps assistant message output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss. The `<｜Assistant｜>` and `<｜tool▁outputs▁end｜>` that open an assistant turn stay outside the markers, since the prompt already ends with them.

### `deepseekv3_training.jinja`

Patched DeepSeek-V3 template. Diff vs `deepseekv3.jinja`:

- Uses `| tojson` on `tool['function']['arguments']` so that `arguments` can be passed as a `dict` (the documented format per [transformers docs](https://huggingface.co/docs/transformers/en/chat_extras#tool-calling-example)). The original template uses raw string concatenation, which crashes on dict inputs.
- Removes the `'</think>'` split on assistant content, so the reasoning block is always rendered. The original drops everything up to and including `</think>`, which would silently remove the reasoning from the training target.
- Wraps assistant message output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss. The `<｜Assistant｜>` and `<｜tool▁outputs▁end｜>` that open an assistant turn stay outside the markers, since the prompt already ends with them.

### `deepseekv3_v2_training.jinja`

Patched DeepSeek-R1-0528 template. Diff vs `deepseekv3_v2.jinja`:

- Uses `| tojson` on `tool['function']['arguments']` so that `arguments` can be passed as a `dict`, as in `deepseekv3_training.jinja`.
- Removes the `'</think>'` split on assistant content, so the reasoning block is always rendered. The original drops everything up to and including `</think>`, which would silently remove the reasoning from the training target.
- Wraps assistant message output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss. The `<｜tool▁outputs▁end｜>` that opens an assistant turn after tool outputs stays outside the markers, since the prompt already ends with it.

### `deepseekv3_v3_training.jinja`

Patched DeepSeek-V3 template (used by `deepseek-ai/DeepSeek-V3`). Same diff as `deepseek_r1_distill_training.jinja`, applied to `deepseekv3_v3.jinja`: `{% generation %}` / `{% endgeneration %}` markers around assistant message output for SFT assistant-only loss, with the `<｜Assistant｜>` and `<｜tool▁outputs▁end｜>` that open an assistant turn left outside them.

### `deepseekv3_v4_training.jinja`

Patched DeepSeek-V3-0324 template (used by `deepseek-ai/DeepSeek-V3-0324`). Same diff as `deepseekv3_v2_training.jinja`, applied to `deepseekv3_v4.jinja`: `| tojson` on tool-call arguments, and `{% generation %}` / `{% endgeneration %}` markers around the assistant output.

### `diffusion_gemma_training.jinja`

Patched DiffusionGemma template. Diff vs `diffusion_gemma.jinja`:

Wrap the parts of a model turn that the model emits (thinking channel, tool calls, content, and the closing `<turn|>`) with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss. Tool responses embedded in the turn come from the environment and stay outside the generation block.

### `gemma_training.jinja`

Patched Gemma template (shared by Gemma and Gemma2, which ship identical chat templates). Diff vs `gemma.jinja`:

Split the unified assistant output so that the `<start_of_turn>model\n` header (a prompt cue, not generated by the model) sits outside the generation block, and wrap the assistant content with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss.

### `gemma3_training.jinja`

Patched Gemma 3 template. Same diff as `gemma_training.jinja` (split the unified output line into role-specific branches so the `<start_of_turn>model\n` prompt cue sits outside the generation block, and wrap the assistant content with `{% generation %}` / `{% endgeneration %}`), applied to the Gemma 3 base template that supports system messages and multimodal content blocks.

### `glm4moe_training.jinja`

Patched GLM-4-MoE template. Diff vs `glm4moe.jinja`:

Close each assistant turn inside the generation block with the role marker that opens the next turn (`<|observation|>` after tool calls, otherwise `<|user|>` or `<|system|>`). GLM's original template has no end-of-turn token: a turn is ended by the next message's role marker, and GLM declares `<|user|>` and `<|observation|>` as end-of-sequence tokens, so the training template appends the marker to the assistant turn to keep the stop token in the loss mask, and the following message no longer renders its own marker.

Wrap assistant message output (including the thinking block and tool calls) with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss. With `enable_thinking=False`, the thinking block stays outside the markers, since the generation prompt already ends with the empty `\n<think></think>`.

### `qwen3_training.jinja`

Patched Qwen3 template. Diff vs `qwen3.jinja`:

Always include the thinking block regardless of message position. The original conditionally omits it based on `loop.last`, which changes the assistant rendering when a tool message is appended, breaking prefix-preservation:

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

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss. The part of the thinking block that the generation prompt pre-fills (`<think>\n\n</think>\n\n` with `enable_thinking=False`) stays outside the markers, since the prompt already ends with it. The other Qwen3 training templates do the same with what their own generation prompt pre-fills, such as `<think>\n`.

### `qwen3_v2_training.jinja`

Patched original Qwen3 template (shipped by the Qwen3 checkpoints before May 2025, and still by the Qwen3 base checkpoints). Same diff as `qwen3_training.jinja`, applied to `qwen3_v2.jinja`.

### `qwen3_vl_training.jinja`

Patched Qwen3-VL template. Diff vs `qwen3_vl.jinja`:

Wrap assistant message output (both `content` and `tool_calls`) with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `gptoss_training.jinja`

Patched GPT-OSS template. Diff vs `gptoss.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `idefics3_training.jinja`

Patched Idefics3 template. Diff vs `idefics3.jinja`:

Split the assistant message into its own branch so the `{% generation %}` / `{% endgeneration %}` markers wrap the assistant content. This enables `return_assistant_tokens_mask=True` to produce correct masks for SFT assistant-only loss.

### `lfm2_training.jinja`

Patched LFM2 template. Diff vs `lfm2.jinja`:

Split the unified message output line into role-specific branches, so the `<|im_start|>assistant\n` prompt cue sits outside the generation block (it is not generated by the model), while the assistant's content and `<|im_end|>\n` (which the model must learn to produce and to stop on) sit inside. Wrap the assistant content with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `lfm2_v2_training.jinja`

Patched LFM2 template, later revision. Diff vs `lfm2_v2.jinja`:

Wrap the assistant output with `{% generation %}` / `{% endgeneration %}`, in both the branch that renders `tool_calls` and the plain-content branch, so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss. The `<|im_start|>assistant\n` prompt cue stays outside the generation block, while the content, the tool calls and `<|im_end|>\n` sit inside.

### `llama3_training.jinja`

Patched Llama 3 template. Diff vs `llama3.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `llava_next_training.jinja`

Patched Llava-Next template. Diff vs `llava_next.jinja`:

- Wraps assistant message output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss.
- Keeps the space that follows `</s>` outside the generation block. The tokenizer merges it into the next turn's first token (`▁[`), which would otherwise be masked as assistant output and hide the `</s>` stop token from the end of the masked span.

### `mistral_large_4_training.jinja`

Patched Mistral Large 4 template. Diff vs `mistral_large_4.jinja`: same change as `mistral_medium_3_5_training.jinja`, wrapping the assistant turn (content, tool calls and the end-of-turn token) with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `mistral_medium_3_5_training.jinja`

Patched Mistral Medium 3.5 template. Diff vs `mistral_medium_3_5.jinja`: the original is already prefix-preserving, so the only change is wrapping the assistant turn (content, tool calls and the end-of-turn token) with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `muse_glimmer_training.jinja`

Patched Muse Glimmer template. Diff vs `muse_glimmer.jinja`: wrap the whole assistant branch — every ATEM channel of the turn — with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `nemotron_3_nano_training.jinja`

Patched Nemotron Nano template. Diff vs `nemotron_3_nano.jinja`: the original is already prefix-preserving, so the only change is wrapping assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `nemotron_3_super_training.jinja`

Patched Nemotron Super template. Diff vs `nemotron_3_super.jinja`: same as `nemotron_3_nano_training.jinja` — the original is already prefix-preserving, so the only change is wrapping assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `nemotron_3_ultra_training.jinja`

Patched Nemotron Ultra template. Diff vs `nemotron_3_ultra.jinja`: same as `nemotron_3_nano_training.jinja` — the original is already prefix-preserving, so the only change is wrapping assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `nemotron_3_5_lightning_training.jinja`

Patched Nemotron 3.5 Lightning template. Diff vs `nemotron_3_5_lightning.jinja`: same as `nemotron_3_nano_training.jinja` — the original is already prefix-preserving, so the only change is wrapping assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `phi3_training.jinja`

Patched Phi-3 template. Diff vs `phi3.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `phi3_5_training.jinja`

Patched Phi-3.5 template. Diff vs `phi3_5.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `qwen2_5_training.jinja`

Patched Qwen2.5 template. Diff vs `qwen2_5.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `qwen2_5_base_training.jinja`

Patched Qwen2.5 base template (used by the base checkpoints, such as `Qwen2.5-0.5B`, whose template only differs from the instruct one in its default system prompt). Same diff as `qwen2_5_training.jinja`, applied to `qwen2_5_base.jinja`.

### `qwen2_5_vl_training.jinja`

Patched Qwen2.5-VL template (also used for Qwen2-VL, which ships a byte-identical template). Diff vs `qwen2_5_vl.jinja`:

Split the assistant message into its own branch so the `{% generation %}` / `{% endgeneration %}` markers wrap the assistant content.  This enables `return_assistant_tokens_mask=True` to produce correct masks for SFT assistant-only loss.

### `qwen3_instruct_2507_training.jinja`

Patched Qwen3-Instruct-2507 template (used by models like `Qwen3-4B-Instruct-2507`, which ship a simpler Qwen3 variant without `reasoning_content` / `<think>` parsing, `multi_step_tool` tracking, or the `enable_thinking` flag). Diff vs `qwen3_instruct_2507.jinja`:

Wrap assistant message output with `{% generation %}` / `{% endgeneration %}` so that `return_assistant_tokens_mask=True` produces correct masks for SFT assistant-only loss.

### `qwen3_thinking_2507_training.jinja`

Patched Qwen3-Thinking-2507 template (used by models like `Qwen3-4B-Thinking-2507`, which pre-fill `<think>\n` in the generation prompt). Diff vs `qwen3_thinking_2507.jinja`: same set of changes as `qwen3_training.jinja`, which drops the `loop.index0 > ns.last_query_index` conditional so the thinking block is always emitted (prefix-preservation) and wraps assistant output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss.

### `qwen3_5_think_training.jinja` / `qwen3_5_nothink_training.jinja`

Patched Qwen3.5 templates with shared logic across both flavors (they differ only in the default value of the `enable_thinking` flag):

- `qwen3_5_think_training.jinja` defaults to thinking enabled, used by Qwen3.5-4B and larger.
- `qwen3_5_nothink_training.jinja` defaults to thinking disabled, used by Qwen3.5-2B and smaller.

Diff vs `qwen3_5_think.jinja` / `qwen3_5_nothink.jinja`: same as `qwen3_training.jinja` (drop the `loop.index0 > ns.last_query_index` conditional so the thinking block is always emitted, wrap assistant output in `{% generation %}` / `{% endgeneration %}`).

### `qwen3_6_training.jinja`

Patched Qwen3.6 template. Diff vs `qwen3_6.jinja`: same as `qwen3_training.jinja` (drop the `loop.index0 > ns.last_query_index` conditional so the thinking block is always emitted, wrap assistant output in `{% generation %}` / `{% endgeneration %}`).

### `qwen3_8_training.jinja`

Patched Qwen3.8 template. Diff vs `qwen3_8.jinja`: drop the `preserve_thinking` / `loop.index0 > ns.last_query_index` conditional so the thinking block is always emitted — prefix-preservation then holds even when the caller passes `preserve_thinking=False` — and wrap assistant output with `{% generation %}` / `{% endgeneration %}` markers for SFT assistant-only loss, with the part of the thinking block that the generation prompt pre-fills left outside them.

### `smolvlm_training.jinja`

Patched SmolVLM template (also used for SmolVLM2, which ships a byte-identical template). Diff vs `smolvlm.jinja`:

Split the assistant message into its own branch so the `{% generation %}` / `{% endgeneration %}` markers wrap the assistant content. This enables `return_assistant_tokens_mask=True` to produce correct masks for SFT assistant-only loss.

## Related utilities

See [Chat Template Utilities](chat_template_utils) for the helper functions ([`clone_chat_template`], [`is_chat_template_prefix_preserving`], [`get_training_chat_template`]) that operate on these templates.
