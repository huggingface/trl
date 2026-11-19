---
name: check-new-model
description: Check that a newly released model, or a new revision of a model's repo files (chat template, tokenizer, config, generation config), works with every TRL trainer, then write a short report of the gaps, who should fix each (TRL, transformers, vLLM, PEFT, kernels or the model provider), and recommendations for the provider. Use when a model is released, when its Hub files change, or when asked whether TRL supports a model.
---

# Check a new model against TRL

Launch PRs reliably cover "loads, trains one step, chat template recognized". What past releases missed, sometimes for months: other sizes' template revisions, tool calling and parsing, the exact boundaries of the assistant mask, end-of-turn ids, packing through non-attention layers, attention-backend limits and quantized checkpoints. This skill checks those properties in a fixed order, on small files and a tiny random-weight model first, so most of it runs before the weights are public.

## Inputs

- The model id and the commit sha of the revision to check (`python -c "from huggingface_hub import model_info; print(model_info('<id>').sha)"`), or a local directory with `config.json`, the tokenizer files, the chat template, the processor files for a VLM, and `generation_config.json`.
- Whether the weights are available. Without them, skip step 6 and say so in the report.
- Every checkpoint of the release (sizes; base, instruct and thinking variants; quantized mirrors). Templates often differ across sizes.

Run everything from the root of the TRL repo, on a GPU machine with TRL's dev dependencies. The scripts live in `.agents/skills/check-new-model/scripts/` and print a markdown table: PASS, FAIL (a TRL path breaks or trains the wrong thing), GAP (support TRL or the ecosystem does not have yet), INFO (a fact the report may need).

## Procedure

1. **Fetch small files, pin the revision.** Download only non-weight files at the sha (`hf download <id> --revision <sha> --exclude "*.safetensors" --exclude "*.bin" --exclude "*.gguf" --local-dir <dir>`), and pass `--revision <sha>` to every script. Check that `AutoConfig.from_pretrained` works on the installed transformers without remote code, that `config.architectures`, `dtype` and `transformers_version` are set, and note the transformers release that introduced the `model_type`. Hash `tokenizer.chat_template` across every checkpoint of the release; each distinct template gets its own run of steps 2 to 4.
2. **Diff the template against TRL's stored templates.** `probe_template.py` reports an exact match with a file in `trl/chat_templates/` or the closest one. Read the diff: `diff trl/chat_templates/<closest>.jinja <dir>/chat_template.jinja`. A one-byte difference means TRL does not recognize the template, so no response parser and no training template.
3. **Run the template probes.** `python .agents/skills/check-new-model/scripts/probe_template.py <id> --revision <sha>`. It goes through TRL's public helpers (`has_generation_markers`, `supports_tool_calling`, `is_chat_template_prefix_preserving`, `is_chat_template_stop_token_trained`, `get_training_chat_template`, `add_response_schema`, `parse_response`) and checks, for every thinking setting the template reads: the generation prompt is a token prefix of the rendered turn, the exact tokens under the assistant mask, the end-of-turn id against the tokenizer and generation-config EOS ids, render-then-parse round trips with typed tool arguments, content forms, and images for a VLM. It prints the TRL, transformers and torch versions it ran with: report them.
4. **Run TRL's own template and data tests with the new model as the only model.** `python .agents/skills/check-new-model/scripts/make_tests.py <id> --revision <sha>` copies `tests/` to a temporary directory, makes the new model the only model of the model-parametrized tests in `test_chat_template_utils.py` and `test_data_utils.py`, and runs pytest there; the repo is not modified. Failures are expected where TRL has no stored template yet (every `TestGetTrainingChatTemplate` test fails when `get_training_chat_template` raises), and the parse tests slice the completion at the prompt length, so a generation-prompt mismatch from step 3 fails them too: read the GP rows first.
5. **Train a tiny copy in every trainer.** `python .agents/skills/check-new-model/scripts/smoke_trainers.py <id> --revision <sha>`. It builds a random-weight model from the real config (same class, vocabulary, special tokens, layer types, experts, vision tower; small widths), then checks the fused LM head against the model's logits, the padding-free swap test, and 2 steps of SFT (plain, `assistant_only_loss`, packing, padding-free), DPO, KTO, Reward, GRPO, GRPO with tools (generation scripted in the model's own tool-call format, and the episode's completion ids checked for a doubled stop token), RLOO, Distillation and LoRA; MoE adds the aux loss, VLMs add SFT, DPO and GRPO on images and text-only training through `AutoModelForCausalLM`. Every stage also checks that every trainable parameter moved. `--stages sft,dpo` reruns a subset.
6. **Real weights, when available** (smallest size first, one GPU):
   - Load by name the way trainers do (`SFTTrainer(model="<id>")`): quantized releases (MXFP4, FP8) must load dequantized and train, full and LoRA.
   - Generate with thinking on and off, a tool call with nested arguments, and an image for a VLM; parse each completion with `parse_response(tokenizer, ids, prefix=prompt_ids, tools=tools)`; check the completion stops on a declared EOS id and the model calls the tool in every thinking mode.
   - vLLM: `trl vllm-serve --model <id> --revision <sha>`, then one GRPO step with `use_vllm=True` on the tiny model from step 5 (colocate and server); compare trainer and vLLM log-probs after the first weight sync.
   - Real head dimensions and layer types only exist here: run SFT `padding_free=True` with `kernels-community/flash-attn3` on the smallest size.
7. **Write the report** with the template at the end of this file.

Reading the results:
- A FAIL on a template property is often already handled by TRL with a warning (for example the generation-prompt mismatch, where SFT starts the completion where the two renders diverge). It stays a provider gap: training and inference see different tokens.
- Random tiny weights hide numerical issues: logits stay far below a softcap, a channel-format model never emits its markers. The fused-LM-head check passing does not prove that the scale and softcap fields are read: check their names against `add_fused_lm_head` in `trl/trainer/utils.py`.
- The packing swap test compares bf16 log-probs of a document after two different documents: noise stays below 1e-2 and is uniform, a leak is larger on the document's first tokens. Missing fast kernels (causal-conv1d, flash-linear-attention) change the result, so report the installed versions.
- The tiny model uses `head_dim` 16; the static check reports the real head dimensions. A tiny model hides kernel limits such as FA3 rejecting `head_dim` above 256.
- An "unchanged params" FAIL lists parameter names: decide whether they legitimately get no gradient (MTP heads, buffers stored as parameters) or are a model or TRL bug.
- Do not rely on the `trl-lib/chat-template-inspector` Space alone (and never paste an unreleased template into it): it looks for the literal `{% generation %}` only, compares the tokenizer `eos_token` only, and never renders a tool message.

## Checklist by area

Each line: the property, how it is checked, and for past misses the PR that fixed it. "probe", "tests" and "smoke" are steps 3, 4 and 5.

**Repo files and loading (RF)**
- Chat template ships as `chat_template.jinja` (or the `tokenizer_config.json` key) and loads through `AutoTokenizer`/`AutoProcessor`, not only a Python encoder. probe.
- `model_type` is native in a released transformers; `architectures`, `dtype`, `transformers_version` in `config.json`; `generation_config.json` present. Step 1.
- One template across sizes and variants, or each revision stored: Qwen3.5 small vs large (https://github.com/huggingface/trl/pull/5405), Gemma 4 sizes (https://github.com/huggingface/trl/pull/7463), base checkpoints (https://github.com/huggingface/trl/pull/7974).
- `AutoModelForSequenceClassification` exists for the config, else Reward fails. smoke `reward`.

**Chat template identity and rendering (CT)**
- Exact match with a stored template, else add `<name>.jinja` (or `_vN`), its branches in `add_response_schema` and `get_training_chat_template`, and the source repo plus the most-downloaded checkpoint to `REPOS` in `scripts/check_chat_templates.py`. probe. Missed: LFM2.5 (https://github.com/huggingface/trl/pull/7909), Qwen3.6 (https://github.com/huggingface/trl/pull/8181).
- Named template variants (a `dict`) use `default`. probe; https://github.com/huggingface/trl/pull/7530.
- `tools=[]` renders like no tools. probe; https://github.com/huggingface/trl/pull/5380.
- String and list-of-parts content render identically (TRL always sends parts to a processor). probe.
- A tool-call turn renders the same with `content` missing, `None` or `""`. probe; https://github.com/huggingface/trl/pull/8228.
- Roles: a system message anywhere, unknown roles and empty assistant turns are rendered or rejected, never dropped silently. Manual render. Pre-render the whole training dataset before a long run.
- No `strftime_now` date or hardcoded year in the default system prompt. probe.

**Generation prompt and prompt/completion split (GP)**
- The generation prompt is a token prefix of the rendered turn, for every thinking setting, with and without reasoning. probe, tests. Missed: Gemma 4 12B+ (https://github.com/huggingface/trl/pull/7463).
- The tail of the generation prompt (role header, `<think>\n`, empty thinking block) per setting is known; masks and parsing depend on it. probe.
- `trl.data_utils.apply_chat_template` keeps the generation prompt in `prompt` (used by the VLM collators and experimental trainers). probe.
- Appending a user turn leaves earlier tokens unchanged (multi-turn SFT, AsyncGRPO re-rendering). probe. Missed: Gemma 4 dropped reasoning from history (https://github.com/huggingface/trl/issues/7503).
- `continue_final_message=True` works on a prompt ending with an assistant turn. probe.
- The template kwargs (`enable_thinking`, `reasoning_effort`, others) and their defaults per size are documented; invalid values raise. Read the template.

**Assistant-only loss (AM)**
- `get_training_chat_template(pc, require_prefix_preservation=False)` returns a template or `None`; a raise means `assistant_only_loss=True` fails at init until the provider ships `{% generation %}` markers or TRL stores `<name>_training.jinja`. probe, smoke. Missed: gpt-oss (https://github.com/huggingface/trl/pull/5484); open for Gemma 4 (https://github.com/huggingface/trl/issues/8095).
- The end-of-turn token is inside the mask: `is_chat_template_stop_token_trained`. probe. Missed: GLM (https://github.com/huggingface/trl/pull/6044), Llava-Next (https://github.com/huggingface/trl/pull/7375).
- The mask starts exactly where the generation prompt ends, for every thinking setting (a prefilled `<think>\n` or empty thinking block is not trained). probe, tests. Missed in 19 templates (https://github.com/huggingface/trl/pull/8110, https://github.com/huggingface/trl/pull/8072, https://github.com/huggingface/trl/pull/8219).
- A TRL training template renders byte-identically to the original, reasoning and tool calls included. tests (`test_behavior_unchanged_*`). Missed: https://github.com/huggingface/trl/pull/6363, https://github.com/huggingface/trl/pull/8061.
- VLM masks stay aligned with images (transformers >= 5.18). smoke `sft_vlm` with `assistant_only_loss=True` when a template exists.

**Tool calling (TC)**
- `supports_tool_calling`: a user, assistant tool call (dict arguments), tool exchange renders with every part. probe. Missed: Llama 3 dropped `tool_calls` (https://github.com/huggingface/trl/pull/5517).
- Prefix-preserving after a tool message, or a stored training template: `is_chat_template_prefix_preserving`, then `get_training_chat_template`. probe. Missed: Qwen3 (https://github.com/huggingface/trl/pull/5330).
- GRPO trains through a tool call in the model's own format: `tools/call_frequency` > 0, `tools/failure_frequency` == 0, and no stop id doubled where the tool result is appended (a template that ends a turn with the next role marker gets it twice). smoke `grpo_tools`.
- Argument values containing the format's delimiters (parentheses, quotes, newlines, closing tags) round-trip. Missed: LFM2.5 (https://github.com/huggingface/trl/pull/7942).
- The model actually calls tools with thinking on and off. Step 6.

**Reasoning (RS)**
- The template reads `reasoning_content` (or say which key it reads), and the parser writes `reasoning_content`. probe. Open for gpt-oss and LFM2.5 (https://github.com/huggingface/trl/issues/7673); Gemma 4 fixed upstream (https://github.com/huggingface/transformers/pull/49535).
- Missing reasoning in earlier turns neither raises nor gets trained as an empty block. tests.
- A thinking-off mode renders the same at generation and training time. probe (per-setting rows).

**Response parsing (RP)**
- A parser exists: a native `response_template` in the tokenizer config, or `add_response_schema` knows the template. Else GRPO, AsyncGRPO and Distillation with tools raise at init. probe. Missed: gpt-oss (https://github.com/huggingface/trl/pull/5464), Gemma 4 (https://github.com/huggingface/trl/pull/6535).
- Render, then `parse_response(..., prefix=prompt_ids, tools=tools)` returns the message: content, reasoning, a typed call (`"75001"` stays a string), two calls. probe, tests. Missed: Qwen3 parallel calls (https://github.com/huggingface/trl/pull/4709); typing (https://github.com/huggingface/trl/pull/7650).
- A completion ending on a stop token, including a next-role marker, still parses as a tool call. probe. Missed: GLM (https://github.com/huggingface/trl/pull/8032).
- Truncated tool calls: GRPO executes whatever the parser returns. probe (INFO).
- Long malformed completions do not hang the parser. tests. Missed: https://github.com/huggingface/trl/pull/5798.

**Tokenizer, special tokens, EOS and pad (TK)**
- The turn terminator and the tool-call terminator are in `generation_config.eos_token_id` or the tokenizer EOS, compared as ids; else completions run past the turn. probe. Missed: Gemma 3/4, Phi-3.5 (https://github.com/huggingface/trl/pull/7505).
- `eos_token` is the turn terminator; if not, text-format SFT appends the wrong token and `parse_response` does not strip the terminator. probe (INFO).
- A pad token distinct from EOS; pad, EOS and BOS ids agree across tokenizer, `config.json` and `generation_config.json`. probe (INFO).
- One BOS on TRL's tokenization path; decode then encode gives the same ids; structural tokens are special. probe.
- No near-greedy sampling defaults in `generation_config.json` (https://github.com/huggingface/trl/issues/5783). probe (INFO).

**Config and LM head (CF)**
- TRL's fused LM head matches the model's logits (logit scale, multipliers, final softcap). smoke `fused_lm_head`; field names in `add_fused_lm_head`. https://github.com/huggingface/trl/pull/7439.
- Fields TRL reads exist under the expected names on the text config: `head_dim`, expert counts, `output_router_logits`, `router_aux_loss_coef`, `layer_types`, `vocab_size`. smoke. Missed: https://github.com/huggingface/trl/pull/8236, https://github.com/huggingface/trl/pull/7919, https://github.com/huggingface/trl/pull/5592.
- Training does not change the saved config beyond dtype and `use_cache`. smoke `sft_config`. Missed: https://github.com/huggingface/trl/pull/8103.
- Text-model settings (softcap, special ids) sit on `text_config` for composite models; parameters kept in fp32 are in `_keep_in_fp32_modules` (`check_dtype_pattern` in `scripts/generate_tiny_models/_common.py`).

**Modeling: attention, packing, gradients (MD)**
- Head dimensions fit Flash Attention (at most 256 on FA3): padding-free and AsyncGRPO tree attention need it. smoke (static). Open for Gemma 4 (https://github.com/huggingface/trl/issues/8104).
- A packed document's log-probs do not depend on the document before it, through attention, conv and recurrent layers. smoke `packing` (swap test on the flash-attention path), `sft_packing`. Missed: Qwen3.5 gated DeltaNet (https://github.com/huggingface/trl/pull/7599).
- Every trainable parameter gets a gradient; gradient checkpointing (TRL's default) is supported. smoke.
- Model-specific attention semantics survive the backend (bidirectional image attention, softcapping, sliding windows). Missed: Gemma 3 under Flash Attention (https://github.com/huggingface/transformers/issues/50710).

**MoE (MO)**
- The model returns router logits and a non-zero aux loss when `router_aux_loss_coef` > 0; bias-routed models have none (TRL then trains without a balancing loss). smoke `aux_loss`. Missed: https://github.com/huggingface/trl/pull/7248.
- LoRA reaches the routed experts (`target_parameters` on fused experts) and the dense and shared layers. smoke `lora` (count on experts). Missed: https://github.com/huggingface/trl/pull/7769, https://github.com/huggingface/trl/pull/7771.
- `compute_flops_per_token` knows the `model_type` (else no MFU metric). smoke (INFO).

**Hybrid and recurrent layers (HY)**
- `layer_types` declares every non-attention layer; AsyncGRPO tree packing supports `linear_attention` and `conv` only. smoke (static). https://github.com/huggingface/trl/pull/7686, https://github.com/huggingface/trl/pull/7966; open for Mamba2 (https://github.com/huggingface/trl/pull/8457).
- The fast kernels (flash-linear-attention, causal-conv1d) are installed and used; a silent PyTorch fallback is about 10x slower. Check the log for fallback warnings.

**Vision-language models (VL)**
- Processor outputs are keys TRL batches; `config.image_token_id` equals the processor's; one run of image tokens per image. probe. Missed: Gemma 3 image token (https://github.com/huggingface/trl/pull/7773), Qwen2.5-VL `mm_token_type_ids` in DPO (https://github.com/huggingface/trl/pull/5279).
- SFT, DPO and GRPO on images, and text-only training of the checkpoint loaded with `AutoModelForCausalLM`. smoke. Missed: https://github.com/huggingface/trl/pull/7763.
- The vision tower is frozen for text-only training (https://github.com/huggingface/trl/issues/6617, open); the policy cannot sample the image token (https://github.com/huggingface/trl/pull/7695).

**Quantized checkpoints (QT)**
- MXFP4 and FP8 checkpoints load dequantized by name and train, full and LoRA. Step 6. Missed for 15 months on gpt-oss (https://github.com/huggingface/trl/pull/8070).
- Dequantize then `save_pretrained` gives a checkpoint that reloads with the same logits; FP8 block sizes divide every dimension.

**PEFT, vLLM, distributed (PF, VS, DS)**
- `LoraConfig(task_type="CAUSAL_LM")` without `target_modules` works (PEFT knows the `model_type`), else document `all-linear` or explicit targets that exist on the architecture. Missed: Nemotron 3 example targets (https://github.com/huggingface/trl/pull/6870).
- vLLM in TRL's supported range serves the architecture, the text-only class included, and weight sync maps every parameter name. Step 6. Missed: Qwen3.5 text-only class (https://github.com/huggingface/trl/issues/5269).
- Experimental trainers (`trl/experimental/`) drop `chat_template_kwargs` or tools in places and re-add BOS or EOS to rendered text; check the ones the user asks about with the same probes.
- FSDP2 wraps by `_no_split_modules`; expert parallelism and FSDP2 numerics need a fp32 per-tensor gradient comparison on 2+ GPUs, not a loss curve.

**Tests and docs (TT, DG)**
- Tiny model generator in `scripts/generate_tiny_models/` at the transformers version that introduced the model; `print_config_diff` shows only size fields; `check_dtype_pattern` passes; quantized like the reference. https://github.com/huggingface/trl/issues/7137.
- The model is added to the template tests, the trainer `test_train` lists (text and VLM), the fused-LM-head and FLOPs tests, with version skips; the docs list it ("Supported model families", training templates, GRPO tool-calling and VLM lists).

## Report template

Keep it to one screen. Every gap names the evidence (probe row, smoke stage, test id, or command) and one owner.

```markdown
# <model id> @ <sha>: TRL compatibility

TRL <version> (<commit>), transformers <version>, PEFT <version>, vLLM <version>. Weights checked: <yes/no>.
Template: <matches trl/chat_templates/<file> | new, closest <file>>.

## Verdict per trainer

| Trainer | Verdict | Note |
|---|---|---|
| SFT (assistant_only_loss, packing) | works / works with caveats / blocked | |
| DPO, KTO | | |
| Reward | | |
| GRPO, RLOO (tools) | | |
| Distillation | | |
| AsyncGRPO | | |
| VLM paths | | |
| LoRA | | |
| vLLM | | |

## Gaps by owner

- TRL: <gap> (<evidence>). Fix: <one line>.
- transformers / vLLM / PEFT / kernels: <gap> (<evidence>).
- Model provider: <gap> (<evidence>).

## Recommendations for the provider

1. <one line, from the list below, with the concrete token, file or field>
```

Recommendations to pick from (only the ones the checks support):
1. Ship `{% generation %}` markers whose span starts right after the generation prompt (role header and any prefilled thinking block excluded) and ends on the end-of-turn token.
2. End every assistant turn with an explicit end-of-turn token inside the turn, not with the next role marker.
3. Keep one chat template across sizes, or document the differences; version template edits instead of silently replacing the file. Share the final template before an embargoed launch.
4. Render finished turns the way the generation prompt prefills them (keep the empty thinking block, no `loop.last`-dependent history), so prompt plus completion extends the prompt.
5. Accept `tool_calls` with dict arguments and with `content` set to `None`, `""` or absent.
6. Ship a `response_template` whose output keys are the ones the template reads back, with `reasoning_content` for reasoning; type arguments through the tool schema.
7. Declare every end-of-turn id in `generation_config.eos_token_id`, make `eos_token` the turn terminator, ship a pad token distinct from EOS, avoid near-greedy sampling defaults.
8. Declare `head_dim`, `layer_types` for hybrid models and the standard MoE fields (`num_experts`, `output_router_logits`, `router_aux_loss_coef`), and return router logits and the aux loss.
9. Keep `head_dim` at most 256, or state the limit in the model card.
10. Quantized releases: publish a bf16 checkpoint or confirm dequantize and save round-trips; FP8 blocks that divide every dimension; declare fp32 and unquantized modules.
11. VLMs: make the processor's `image_token_id` the placeholder token, ship processor files with every checkpoint, keep `architectures` consistent with the repo contents.
12. Recurrent and conv layers: accept `seq_idx` / `cu_seq_lens` for packed inputs and call module-level kernel functions.
13. Model card: list LoRA target modules that exist on the architecture and whether gradient checkpointing is supported.
14. Render string and list-of-parts content identically; mark structural tokens special; no date-dependent default system prompt.
