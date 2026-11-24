---
name: check-new-model
description: Check that a newly released model, or a new revision of its Hub files, works with every TRL trainer, then report the gaps and recommendations for the model provider. Use when a model is released, when its chat template or config changes, or when asked whether TRL supports a model.
---

# Check a new model against TRL

Launch support usually covers "loads, trains a step, template recognized". The checks below are what past releases got wrong anyway, often for months. Most run on the tokenizer and config alone, or on a tiny random-weight model built from the real config, so they work before the weights are public. Run them on every checkpoint of the release (sizes, base and instruct, quantized mirrors), not only one. The snippets are starting points: adapt them to the model (its template kwargs, its tool-call format).

```python
import torch, transformers
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, ProcessorMixin
from trl.chat_template_utils import (
    add_response_schema, get_training_chat_template, is_chat_template_stop_token_trained, parse_response,
    supports_tool_calling,
)

tok = AutoTokenizer.from_pretrained(model_id, revision=sha)  # for a VLM, AutoProcessor, and its .tokenizer for ids
config = AutoConfig.from_pretrained(model_id, revision=sha)
gen = GenerationConfig.from_pretrained(model_id, revision=sha)
eos = gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id]
render = lambda msgs, **kw: tok.apply_chat_template(msgs, tokenize=False, **kw)
ids = lambda text: tok(text, add_special_tokens=False)["input_ids"]

user = [{"role": "user", "content": "Hi"}]
turn = user + [{"role": "assistant", "content": "Hello", "reasoning_content": "Greet."}]
tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up a zip code", "parameters": {
    "type": "object", "properties": {"zip": {"type": "string"}}, "required": ["zip"]}}}]
call = {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"zip": "75001"}}}]}
```

## Chat template

A model that ships `{% generation %}` markers, a prefix-preserving template and a native `response_template` needs nothing stored in TRL. Otherwise TRL falls back to its stored copies, matched by exact text, so every checkpoint's template has to match one:

```python
from pathlib import Path
known = {p.read_text(): p.name for p in Path("trl/chat_templates").glob("*.jinja")}
print(known.get(tok.chat_template, "not stored"))
```

TRL trains with its stored training template when it has one, which already fixes the gaps it knows about (Qwen3 dropping reasoning from history), so the checks below run on that template. `get_training_chat_template` raising means `assistant_only_loss=True` fails, and so does GRPO with tools when the template isn't prefix-preserving after a tool turn:

```python
template = get_training_chat_template(tok) or tok.chat_template
```

The generation prompt is a token prefix of the rendered turn, for every setting the template reads (`enable_thinking`, `reasoning_effort`, ...), with and without reasoning (reasoning with thinking off is a contradictory input, so skip it):

```python
answer = [*user, {"role": "assistant", "content": "Hello"}]
for kw, msgs in (({}, turn), ({}, answer), ({"enable_thinking": False}, answer)):
    prompt = ids(render(user, add_generation_prompt=True, chat_template=template, **kw))
    assert ids(render(msgs, chat_template=template, **kw))[: len(prompt)] == prompt, kw
```

Appending a turn leaves the earlier ones unchanged (dropping reasoning from history breaks it):

```python
follow_up = [*turn, {"role": "user", "content": "And?"}]
assert render(follow_up, chat_template=template).startswith(render(turn, chat_template=template))
```

The assistant mask starts right after the generation prompt (a prefilled `<think>` is not trained) and ends on the end-of-turn token:

```python
out = tok.apply_chat_template(turn, chat_template=template, return_dict=True, return_assistant_tokens_mask=True)
print(tok.decode([t for t, m in zip(out["input_ids"], out["assistant_masks"]) if m]))
assert is_chat_template_stop_token_trained(tok, template)
```

Small things that break rendering in training but not in a chat demo:

```python
assert render(user, tools=[]) == render(user)
assert len({render([*user, {**call, "content": c}]) for c in (None, "")} | {render([*user, call])}) == 1
if isinstance(tok, ProcessorMixin):  # TRL sends content as parts to processors only
    assert render([{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]) == render(user)
assert "SYSTEM_SENTINEL" in render([{"role": "system", "content": "SYSTEM_SENTINEL"}, *user])
assert supports_tool_calling(tok)
calls = [{"type": "function", "function": {"name": n, "arguments": {"zip": "75001"}}} for n in ("lookup_a", "lookup_b")]
assert all(n in render([*user, {"role": "assistant", "tool_calls": calls}]) for n in ("lookup_a", "lookup_b"))
json_args = {"type": "function", "function": {"name": "lookup", "arguments": '{"zip": "75001"}'}}
assert render([*user, {**call, "tool_calls": [json_args]}]) == render([*user, call])  # some callers send JSON strings
text_ids = ids(render(turn))
assert ids(tok.decode(text_ids)) == text_ids  # decode then encode gives the same ids
```

## Parsing and stopping

The token that ends an assistant turn is a stop id. Compare ids: a template that ends a turn with the next role marker, or a model that samples a different token than the template renders, runs past the turn:

```python
print(tok.convert_ids_to_tokens(ids(render(turn))[-3:]), tok.convert_ids_to_tokens(eos))
```

Parsing returns what the template rendered, with reasoning in `reasoning_content` and arguments typed by the tool schema:

```python
if getattr(tok, "response_template", None) is None:  # what the trainers check before falling back to TRL's copies
    tok = add_response_schema(tok)  # raises when TRL doesn't know the template
prompt = ids(render(user, add_generation_prompt=True))
assert parse_response(tok, ids(render(turn))[len(prompt):], prefix=prompt) == turn[-1]
prompt = ids(render(user, tools=tools, add_generation_prompt=True))
completion = ids(render([*user, call], tools=tools))[len(prompt):]
assert parse_response(tok, completion, prefix=prompt, tools=tools)["tool_calls"] == call["tool_calls"]
assert "tool_calls" not in parse_response(tok, completion[:-4], prefix=prompt, tools=tools)  # truncated: not executed
```

In GRPO with tools, look at the completion ids of an episode: the stop token must not appear twice where the tool result is appended.

## Config and loading

TRL loads a model id with `config.architectures[0]`, so it must be the class of the `model_type`:

```python
assert getattr(transformers, config.architectures[0]).config_class.model_type == config.model_type
```

`generation_config.json` exists, ids agree across files, and pad is not an EOS (else padding is trained as a stop or masked as one):

```python
text = config.get_text_config()
print(text.eos_token_id, gen.eos_token_id, tok.eos_token_id, tok.pad_token_id, tok.bos_token_id)
```

TRL's fused LM head (`add_fused_lm_head` in `trl/trainer/utils.py`) must read the logit scaling and softcap fields. Random tiny weights never reach a softcap, so compare names, not outputs:

```python
print({k: v for k, v in text.to_dict().items() if any(s in k for s in ("softcap", "logit", "scale", "multiplier"))})
```

A `head_dim` above 256 rules out Flash Attention 3, so padding-free and AsyncGRPO. Read it here: the tiny model below shrinks `text` in place, and tiny models hide it anyway:

```python
print({k: v for k, v in text.to_dict().items() if "head_dim" in k})
```

The Reward trainer needs `AutoModelForSequenceClassification.from_config(config)` to work. A quantized release must load by name dequantized (`SFTTrainer(model=model_id)`) and survive `save_pretrained` then reload with the same logits.

## Architecture

Build a tiny model from the real config: small widths, one layer of each `layer_types` entry, a few experts and at least one MoE layer (`first_k_dense_replace`, `mlp_only_layers`), a small vision tower, no tower you don't test (audio). Shrink every field that mirrors a text size (a vision config's `text_hidden_size`, `out_hidden_size`):

```python
mha = text.num_key_value_heads == text.num_attention_heads  # keep MHA models MHA (MLA needs it)
text.update({"hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4 if mha else 2,
             "head_dim": 16, "layer_types": list(dict.fromkeys(text.to_dict().get("layer_types") or ["full_attention"]))})
text.num_hidden_layers = len(text.layer_types)
model = getattr(transformers, config.architectures[0])(config).to("cuda", torch.bfloat16)
```

MoE: TRL computes the load-balancing loss itself (`add_fused_lm_head`), from the router logits of the backbone and `num_experts_per_tok`, so the backbone must return one `(tokens, experts)` tensor per MoE layer. `LoraConfig(task_type="CAUSAL_LM")` without `target_modules` must work (PEFT knows the `model_type`), and LoRA must reach the routed experts:

```python
x = torch.randint(0, 100, (1, 16), device="cuda")
router_logits = model.base_model(x, output_router_logits=True).router_logits
print(len(router_logits), router_logits[0].shape, text.num_experts_per_tok)
print([n for n, _ in peft_model.named_parameters() if "lora" in n and "expert" in n][:3])
```

Packing doesn't leak across documents, through attention, conv and recurrent layers (run with the flash attention path TRL's padding-free uses; noise stays under 1e-2, a leak is largest on the first tokens). Build the batch with TRL's padding-free collator and pass all its keys: gated-DeltaNet layers split documents on `cu_seq_lens_q` and `seq_idx`, not on `position_ids` alone:

```python
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id, padding_free=True)
def logps(before, doc):  # lists of token ids
    batch = collator([{"input_ids": before}, {"input_ids": doc}])
    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items() if k != "labels"}
    return model(**batch).logits[0, len(before):].float().log_softmax(-1)
print((logps(doc_a, doc) - logps(doc_b, doc)).abs().amax(-1))
```

Also:
- `layer_types` declares every non-attention layer: AsyncGRPO tree packing reads it.
- The fast kernels for linear attention and conv layers are installed: the PyTorch fallback is silent and about 10x slower, so grep the logs for "falling back".
- VLM: `config.image_token_id` is the processor's image token, the checkpoint also trains text-only through `AutoModelForCausalLM`, and every processor output key reaches the model.
- vLLM in TRL's range serves it (`trl vllm-serve --model <id> --revision <sha>`), and one GRPO step with `use_vllm=True` and one AsyncGRPO step sync every weight.

## What providers usually miss

Two things are missing in most compatibility requests:
- a response template, without which TRL can't parse completions, so GRPO, AsyncGRPO and Distillation with tools fail at init;
- `{% generation %}` markers, without which `assistant_only_loss=True` fails.

Recommend adding them, and send the fix: the new chat template with the markers, or the response template. Test it extensively before sending: add the model to the `pytest.param` lists of `tests/test_chat_template_utils.py` (`TestAddResponseSchema`, `TestSupportsToolCalling`, `TestIsChatTemplatePrefixPreserving`, `TestIsChatTemplateStopTokenTrained`, `TestGetTrainingChatTemplate`, `TestParseResponse`) and run them with `tests/test_data_utils.py`.

## Report

Keep it to one screen:
- the versions (TRL, transformers, vLLM, PEFT) and the revision checked;
- a verdict per trainer: SFT, DPO, KTO, Reward, GRPO and RLOO with tools, Distillation, AsyncGRPO, VLM paths, LoRA;
- each gap with its evidence and one owner: TRL, transformers, vLLM, PEFT or the provider;
- short recommendations for the provider, naming the exact token, file or field to change.
