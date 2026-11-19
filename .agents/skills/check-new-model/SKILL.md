---
name: check-new-model
description: Check that a newly released model, or a new revision of its Hub files, works with every TRL trainer, then report the gaps and recommendations for the model provider. Use when a model is released, when its chat template or config changes, or when asked whether TRL supports a model.
---

# Check a new model against TRL

Launch support usually covers "loads, trains a step, template recognized". The items below are what past releases got wrong anyway, often for months. Check them on the real tokenizer and config, and on a tiny random-weight model built from the real config (same class, layer types, experts, vision tower), so most of it runs before the weights are public. Check every checkpoint of the release, not only one.

## Chat template

- Every size and variant (base, instruct, thinking, quantized mirrors) ships the same template. TRL recognizes templates by exact match, so a one-byte difference loses the response parser and the training template.
- The generation prompt is a token prefix of the rendered assistant turn, for every thinking and reasoning-effort setting.
- Appending a user or tool turn leaves the earlier tokens unchanged. Dropping reasoning from history breaks this.
- With `{% generation %}` markers (or TRL's training template), the mask starts right after the generation prompt, so a prefilled `<think>` is not trained, and it includes the end-of-turn token.
- `tools=[]` renders like no tools.
- A tool-call turn renders the same with `content` missing, `None` or `""`.
- Tool-call arguments given as a dict render, and parsing types them through the tool schema (`"75001"` stays a string).
- String content and a list of text parts render identically: TRL sends parts to processors.
- No date in the default system prompt.

## Parsing and stopping

- The token that ends an assistant turn is a stop id. Compare ids, not strings, against `generation_config.eos_token_id`: a template that ends a turn with the next role marker runs past it.
- `parse_response` puts reasoning in `reasoning_content`, separate from `content`.
- A completion that ends on a stop token or on the next role marker still parses as a tool call, and a truncated tool call is not executed.
- In GRPO with tools, the stop token is not doubled where the tool result is appended.

## Config and loading

- `architectures[0]` in `config.json` is the class of its `model_type`: TRL loads a model id with that class.
- `generation_config.json` exists, and the pad, EOS and BOS ids agree between it, `config.json` and the tokenizer. The pad token is distinct from EOS.
- TRL's fused LM head (`add_fused_lm_head`) reads the logit scaling and final softcap under their field names on the text config. Random tiny weights never reach a softcap, so a passing test proves nothing here.
- `AutoModelForSequenceClassification` exists for the config, else the Reward trainer fails.
- A quantized release loads dequantized by name and trains, and dequantize, save, reload gives the same logits.

## Architecture

- MoE: the model returns router logits and a non-zero aux loss when `router_aux_loss_coef` > 0, and LoRA reaches the routed experts.
- Packing: a document's log-probs don't depend on the document packed before it, through attention, conv and recurrent layers. `layer_types` declares every non-attention layer (AsyncGRPO tree packing reads it).
- The fast kernels for linear attention and conv layers are installed and used: the PyTorch fallback is silent and about 10x slower.
- A `head_dim` above 256 rules out Flash Attention 3, so padding-free training and AsyncGRPO. Tiny models hide it.
- VLM: `config.image_token_id` matches the processor's, the checkpoint also trains text-only through `AutoModelForCausalLM`, and every processor output key reaches the model.
- vLLM in TRL's supported range serves the architecture, and weight sync maps every parameter name.

## Report

Keep it to one screen:
- the versions (TRL, transformers, vLLM, PEFT) and the revision checked;
- a verdict per trainer: SFT, DPO, KTO, Reward, GRPO and RLOO with tools, Distillation, AsyncGRPO, VLM paths, LoRA;
- each gap with its evidence and one owner: TRL, transformers, vLLM, PEFT or the provider;
- short recommendations for the provider, naming the exact token, file or field to change.
