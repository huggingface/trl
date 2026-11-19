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

# ruff: noqa: T201

"""
Probe a model's chat template, tokenizer and generation config against what TRL's trainers assume.

Only the small files are downloaded (config, tokenizer, chat template, processor, generation config). Prints one row
per check: PASS, FAIL (a TRL path breaks or trains the wrong thing), GAP (TRL lacks support it could add), or INFO.

    python probe_template.py <model_id_or_dir> [--revision <sha>]
"""

import argparse
import difflib
import json
import re
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig

import trl
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    has_generation_markers,
    is_chat_template_prefix_preserving,
    is_chat_template_stop_token_trained,
    parse_response,
    supports_tool_calling,
)
from trl.data_utils import apply_chat_template, prepare_multimodal_messages


# Processor outputs that TRL's collators and trainers know how to batch and slice
KNOWN_MM_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "image_position_ids",
    "spatial_shapes",
    "pixel_attention_mask",
    "image_sizes",
    "token_type_ids",
    "mm_token_type_ids",
    "num_patches",
}

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Hub model id or local directory")
parser.add_argument("--revision", default=None, help="Hub revision (commit sha) to pin")
args = parser.parse_args()

rows = []


def report(area, check, status, detail="", owner=""):
    rows.append((area, check, status, str(detail).replace("\n", "\\n").replace("|", "\\|"), owner))


def run(area, check, owner, fn):
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = "FAIL", f"{type(e).__name__}: {str(e)[:200]}"
    report(area, check, status, detail, owner if status in ("FAIL", "GAP") else "")
    return status


config = AutoConfig.from_pretrained(args.model, revision=args.revision)
is_vlm = "vision_config" in config.sub_configs  # same rule as TRL
processing_class = (AutoProcessor if is_vlm else AutoTokenizer).from_pretrained(args.model, revision=args.revision)
tokenizer = processing_class.tokenizer if is_vlm else processing_class
template = processing_class.chat_template
if isinstance(template, dict):
    report("CT", "named variants", "INFO", f"{sorted(template)}; TRL reads `default` in some helpers only")
    template = template["default"]
try:
    generation_config = GenerationConfig.from_pretrained(args.model, revision=args.revision)
    gen_eos = generation_config.eos_token_id
except OSError:  # transformers then derives the generation config from config.json
    generation_config, gen_eos = None, config.get_text_config().eos_token_id
    report("CF", "generation_config.json", "INFO", "missing: stop ids come from config.json eos_token_id")
gen_eos = [] if gen_eos is None else [gen_eos] if isinstance(gen_eos, int) else list(gen_eos)
stop_ids = set(gen_eos) | {tokenizer.eos_token_id}  # what GRPO/RLOO/Distillation stop on


def mm(messages, images=None):
    # TRL converts every message to the list-of-parts format when the processing class is a processor
    return prepare_multimodal_messages(messages, images=images) if is_vlm else messages


def ids(messages, **kwargs):
    out = processing_class.apply_chat_template(mm(messages), tokenize=True, return_dict=False, **kwargs)
    out = out[0] if is_vlm else out
    return out.tolist() if isinstance(out, torch.Tensor) else list(out)


def text(messages, **kwargs):
    return processing_class.apply_chat_template(mm(messages), tokenize=False, **kwargs)


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def tokens(token_ids):
    return tokenizer.convert_ids_to_tokens(token_ids)


print(f"model: {args.model}  revision: {args.revision}  class: {config.architectures}  vlm: {is_vlm}")
print(
    f"trl {trl.__version__} ({Path(trl.__file__).parent}), "
    f"transformers {transformers.__version__} ({Path(transformers.__file__).parent}), torch {torch.__version__}"
)
if template is None:
    raise SystemExit("No chat template: the repo must ship chat_template.jinja (or the tokenizer_config key).")

# --- Identity: TRL recognizes a template only by exact match with a stored copy ---
stored = {
    p.name: p.read_text(encoding="utf-8")
    for p in sorted((Path(trl.__file__).parent / "chat_templates").glob("*.jinja"))
}
match = [name for name, t in stored.items() if t == template]
if match:
    report("CT", "stored template", "PASS", ", ".join(match))
else:
    ratio, name = max((difflib.SequenceMatcher(None, t, template).ratio(), n) for n, t in stored.items())
    report("CT", "stored template", "GAP", f"no exact match; closest {name} (similarity {ratio:.2f})", "TRL")
if re.search(r"strftime_now|\b20\d\d\b", template):
    report("CT", "time-varying or hardcoded date", "INFO", "the rendered prompt changes with the date or pins a year")
if is_vlm and tokenizer.chat_template not in (None, template):
    report("CT", "processor vs tokenizer template", "INFO", "they differ; TRL renders with the processor's")

# Which reasoning field the template reads, and which thinking switches it has
probe = [{"role": "user", "content": "Q"}]
reasoning_key = next(
    (
        k
        for k in ("reasoning_content", "thinking", "reasoning")
        if "RSN7" in text(probe + [{"role": "assistant", "content": "A", k: "RSN7"}])
    ),
    None,
)
report("RS", "reasoning field", "INFO", reasoning_key or "none of reasoning_content/thinking/reasoning is rendered")
settings = [{}]
if "enable_thinking" in template:
    settings += [{"enable_thinking": False}, {"enable_thinking": True}]
if "reasoning_effort" in template:
    settings += [{"reasoning_effort": "low"}, {"reasoning_effort": "high"}]

# --- Generation prompt vs rendered turn (SFT/DPO/KTO split the prompt at the token level) ---
user = [{"role": "user", "content": "What color is the sky?"}]
# Settings where a turn without (plain) or with reasoning extends the generation prompt
plain_settings, reasoning_settings = [], []
for kw in settings:
    name = json.dumps(kw) if kw else "default"
    prompt_text = text(user, add_generation_prompt=True, **kw)
    report("GP", f"generation prompt ends with [{name}]", "INFO", repr(prompt_text[-40:]))
    variants = {"no reasoning": {"role": "assistant", "content": "It is blue."}}
    if reasoning_key and not prompt_text.rstrip().endswith("</think>"):
        variants["reasoning"] = {"role": "assistant", "content": "It is blue.", reasoning_key: "Rayleigh."}
    for label, assistant in variants.items():

        def check_prefix(kw=kw, assistant=assistant):
            prompt_ids, full_ids = ids(user, add_generation_prompt=True, **kw), ids(user + [assistant], **kw)
            n = common_prefix(prompt_ids, full_ids)
            if n == len(prompt_ids):
                (reasoning_settings if reasoning_key in assistant else plain_settings).append(kw)
                return "PASS", f"{len(prompt_ids)} prompt tokens"
            return (
                "FAIL",
                f"diverges at token {n}: prompt {tokens(prompt_ids[n : n + 3])} vs turn {tokens(full_ids[n : n + 3])}",
            )

        run("GP", f"prompt is token prefix of turn [{name}, {label}]", "provider", check_prefix)


def check_data_utils_split():
    example = {"prompt": mm(user), "completion": mm([{"role": "assistant", "content": "It is blue."}])}
    out = apply_chat_template(example, processing_class)
    expected = text(user, add_generation_prompt=True)
    return ("PASS", "") if out["prompt"] == expected else ("FAIL", f"prompt cut to {out['prompt'][-30:]!r}")


run("GP", "data_utils prompt-completion split keeps the generation prompt", "provider/TRL", check_data_utils_split)


def check_history():
    turn1 = user + [
        {"role": "assistant", "content": "It is blue.", **({reasoning_key: "Rayleigh."} if reasoning_key else {})}
    ]
    a, b = ids(turn1), ids(turn1 + [{"role": "user", "content": "Why?"}])
    n = common_prefix(a, b)
    if n == len(a):
        return "PASS", ""
    return "INFO", f"earlier turn re-rendered from token {n} (e.g. reasoning dropped from history)"


run("GP", "appending a user turn keeps earlier tokens", "provider", check_history)


def check_continue():
    rendered = text(user + [{"role": "assistant", "content": "It is"}], continue_final_message=True)
    return ("PASS", "") if rendered.rstrip().endswith("It is") else ("FAIL", repr(rendered[-30:]))


run("GP", "continue_final_message", "provider", check_continue)

# --- Assistant-only loss ---
report("AM", "native {% generation %} markers", "INFO", has_generation_markers(template))
sft_template = None


def check_sft_template():
    global sft_template
    sft_template = get_training_chat_template(processing_class, require_prefix_preservation=False)
    return "PASS", "native markers" if sft_template is None else "TRL stored training template"


if run("AM", "assistant_only_loss template (SFT)", "provider/TRL", check_sft_template) == "PASS":
    run(
        "AM",
        "end-of-turn token inside the mask",
        "provider/TRL",
        lambda: ("PASS", "")
        if is_chat_template_stop_token_trained(processing_class, sft_template)
        else ("FAIL", "model never learns to stop"),
    )
    for kw in settings:

        def check_mask(kw=kw):
            prompt_ids = ids(user, add_generation_prompt=True, **kw)
            assistant = {"role": "assistant", "content": "It is blue."}
            if kw in reasoning_settings:
                assistant[reasoning_key] = "Rayleigh."
            out = processing_class.apply_chat_template(
                mm(user + [assistant]),
                chat_template=sft_template,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **kw,
            )
            input_ids, mask = (
                (out["input_ids"][0], out["assistant_masks"][0])
                if is_vlm
                else (out["input_ids"], out["assistant_masks"])
            )
            input_ids, mask = list(input_ids), list(mask)
            if 1 not in mask:
                return "FAIL", "empty assistant mask"
            start, end = mask.index(1), len(mask) - mask[::-1].index(1)
            span = f"trained {tokens(input_ids[start : start + 3])}...{tokens(input_ids[end - 2 : end])}"
            n = common_prefix(prompt_ids, input_ids)
            if start != n:
                return "FAIL", f"mask starts at {start}, generation prompt ends at {n}; {span}"
            return "PASS", span

        run(
            "AM",
            f"mask starts after the generation prompt [{json.dumps(kw) if kw else 'default'}]",
            "TRL/provider",
            check_mask,
        )

# --- End of turn, EOS and pad ---
turn = ids(user + [{"role": "assistant", "content": "It is blue."}])
added = set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids)
# The turn terminator is the first special token after the content, in a turn followed by another user turn (some
# templates end a turn with the next role marker)
rendered = text(user + [{"role": "assistant", "content": "It is blue."}, {"role": "user", "content": "Thanks"}])
end = rendered.index("It is blue.") + len("It is blue.")
encoding = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
after = [t for t, (start, _) in zip(encoding.input_ids, encoding.offset_mapping, strict=True) if start >= end]
terminator = next((t for t in after if t in added), None)
report(
    "TK",
    "ids",
    "INFO",
    f"turn ends with {tokens([terminator]) if terminator is not None else None}; tokenizer eos "
    f"{tokenizer.eos_token!r}; generation_config eos {tokens(gen_eos)}; pad {tokenizer.pad_token!r}",
)
if terminator is None:
    report(
        "TK", "turn terminator is a stop id", "FAIL", "the template emits no special token after the turn", "provider"
    )
elif terminator not in stop_ids:
    report(
        "TK",
        "turn terminator is a stop id",
        "FAIL",
        "completions run past the turn; add it to generation_config eos_token_id",
        "provider",
    )
else:
    report("TK", "turn terminator is a stop id", "PASS")
if terminator is not None and terminator != tokenizer.eos_token_id:
    report(
        "TK",
        "tokenizer eos_token is the turn terminator",
        "INFO",
        "differs: text-format SFT appends eos_token, parse_response strips eos_token only",
    )
if tokenizer.pad_token is None or tokenizer.pad_token_id in stop_ids:
    report("TK", "distinct pad token", "INFO", f"pad={tokenizer.pad_token!r}: TRL falls back to/overlaps with eos")
if tokenizer.bos_token_id is not None:
    report(
        "TK", "single BOS on TRL's tokenization path", "PASS" if turn[:2].count(tokenizer.bos_token_id) < 2 else "FAIL"
    )
roundtrip = tokenizer(tokenizer.decode(turn), add_special_tokens=False).input_ids
report(
    "TK",
    "decode/encode round trip",
    "PASS" if roundtrip == turn else "INFO",
    "" if roundtrip == turn else "ids change",
)
if generation_config is not None:
    sampling = {k: getattr(generation_config, k) for k in ("temperature", "top_p", "top_k", "repetition_penalty")}
    report("CF", "generation_config sampling defaults", "INFO", {k: v for k, v in sampling.items() if v is not None})

# --- Content forms that trainers and datasets produce ---
run(
    "CT",
    "tools=[] renders like tools=None",
    "provider",
    lambda: ("PASS", "")
    if text(user, add_generation_prompt=True, tools=[]) == text(user, add_generation_prompt=True)
    else ("FAIL", "tool preamble for an empty list"),
)
if not is_vlm:

    def check_parts():
        convo = [{"role": "system", "content": "Be brief."}] + user + [{"role": "assistant", "content": "It is blue."}]
        return (
            ("PASS", "")
            if text(prepare_multimodal_messages(convo)) == text(convo)
            else ("INFO", "list-of-parts content renders differently")
        )

    run("CT", "string vs list-of-parts content", "provider", check_parts)
else:

    def check_system_parts():
        convo = [{"role": "system", "content": "Be brief."}] + user
        raw = processing_class.apply_chat_template(convo, tokenize=False)
        return ("PASS", "") if text(convo) == raw else ("FAIL", "system/user parts render differently from strings")

    run("CT", "string vs list-of-parts content", "provider", check_system_parts)

# --- Tools ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                },
                "required": ["city"],
            },
        },
    }
]
call = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": {"city": "Paris (FR)", "zip": "75001", "days": 3, "metric": True},
    },
}
call2 = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lyon"}}}
tool_user = [{"role": "user", "content": "Weather in Paris, then Lyon?"}]
tool_msg = {"role": "tool", "name": "get_weather", "content": "Sunny, 21C"}


def check_supports_tools():
    return ("PASS", "") if supports_tool_calling(processing_class) else ("FAIL", "a tool exchange does not render")


if run("TC", "supports_tool_calling", "provider", check_supports_tools) == "PASS":

    def check_tools_template():
        # Same rule as GRPO, AsyncGRPO and Distillation with tools
        if is_chat_template_prefix_preserving(processing_class):
            return "PASS", "prefix-preserving as shipped"
        get_training_chat_template(processing_class)
        return "PASS", "not prefix-preserving; TRL swaps in its stored training template"

    run("TC", "prefix-preserving template for tool training", "provider/TRL", check_tools_template)

    def check_content_forms():
        base = {"role": "assistant", "tool_calls": [call]}
        renders = {}
        for label, extra in (("missing", {}), ("None", {"content": None}), ('""', {"content": ""})):
            try:
                renders[label] = text(tool_user + [{**base, **extra}])
            except Exception as e:
                renders[label] = f"raises {type(e).__name__}"
        if len(set(renders.values())) == 1:
            return "PASS", ""
        n = min(common_prefix(renders["missing"], r) for r in renders.values())
        return "FAIL", ", ".join(f"{k}: {v[max(0, n - 20) : n + 30]!r}" for k, v in renders.items())

    run("TC", "tool-call turn with content missing/None/empty", "provider", check_content_forms)

    def parser_available():
        if getattr(tokenizer, "response_template", None) is not None:
            return "PASS", "native response_template"
        try:
            add_response_schema(processing_class)
        except ValueError as e:
            return "GAP", f"{e}: GRPO/AsyncGRPO/Distillation with tools raise at init"
        return "PASS", "TRL-provided response template"

    has_parser = run("RP", "response parser", "provider/TRL", parser_available) == "PASS"

    def completion_ids(messages, **kwargs):
        # What a model reproducing the template generates after the prompt, up to its first stop id. A template that
        # ends a turn with the next message's marker gets that marker appended, as the model generates it to stop.
        kwargs["tools"] = tools
        prompt_ids, full = ids(messages[:-1], add_generation_prompt=True, **kwargs), ids(messages, **kwargs)
        if common_prefix(prompt_ids, full) < len(prompt_ids):
            raise ValueError("the turn does not extend the generation prompt (see the GP rows)")
        rest = full[len(prompt_ids) :]
        cut = next((i for i, t in enumerate(rest) if t in stop_ids), None)
        if cut is not None:
            return prompt_ids, rest[: cut + 1]
        next_message = tool_msg if messages[-1].get("tool_calls") else {"role": "user", "content": "Thanks"}
        following = ids(messages + [next_message], **kwargs)
        return prompt_ids, rest + [next(t for t in following[len(prompt_ids) :] if t in stop_ids)]

    if has_parser:
        cases = {
            "content": {"role": "assistant", "content": "It is sunny."},
            "typed tool call": {"role": "assistant", "content": "", "tool_calls": [call]},
            "two tool calls": {"role": "assistant", "content": "", "tool_calls": [call, call2]},
        }
        if reasoning_settings:
            cases["reasoning"] = {"role": "assistant", "content": "It is sunny.", reasoning_key: "Check the sky."}
        for label, message in cases.items():

            def check_roundtrip(message=message):
                kw = (reasoning_settings if reasoning_key in message else plain_settings)[0]
                prompt_ids, comp = completion_ids(tool_user + [message], **kw)
                parsed = parse_response(tokenizer, comp, prefix=prompt_ids, tools=tools)
                got = [c["function"] for c in parsed.get("tool_calls") or []]
                want = [c["function"] for c in message.get("tool_calls") or []]
                if got != want or parsed["content"].strip() != message["content"]:
                    return "FAIL", f"parsed {json.dumps(parsed)[:180]}"
                if reasoning_key in message and parsed.get("reasoning_content") != message[reasoning_key]:
                    return "FAIL", f"reasoning not in reasoning_content: parsed {json.dumps(parsed)[:150]}"
                return "PASS", f"stops on {tokens(comp[-1:])}"

            run("RP", f"render -> parse round trip [{label}]", "TRL/provider", check_roundtrip)

        def check_truncated():
            message = {"role": "assistant", "content": "", "tool_calls": [call]}
            prompt_ids, comp = completion_ids(tool_user + [message], **plain_settings[0])
            bad = []
            for k in range(1, len(comp) - 1):
                calls = parse_response(tokenizer, comp[:k], prefix=prompt_ids, tools=tools).get("tool_calls") or []
                if calls and calls[0]["function"] != call["function"]:
                    bad.append(k)
            if bad:
                return (
                    "INFO",
                    f"{len(bad)}/{len(comp) - 2} truncated completions parse as a different, executable call",
                )
            return "PASS", f"{len(comp) - 2} truncated completions"

        run("RP", "truncated tool call is not executed", "TRL", check_truncated)

# --- Images ---
if is_vlm:

    def check_images():
        image = Image.new("RGB", (64, 48))
        out = processing_class.apply_chat_template(
            mm([{"role": "user", "content": "Describe both."}], images=[image, image]), tokenize=True, return_dict=True
        )
        unknown = set(out) - KNOWN_MM_KEYS
        image_token_id = getattr(config, "image_token_id", None)  # TRL places image features on this token
        processor_token_id = getattr(processing_class, "image_token_id", image_token_id)
        input_ids = list(out["input_ids"][0])
        runs = sum(1 for i, t in enumerate(input_ids) if t == image_token_id and (i == 0 or input_ids[i - 1] != t))
        problems = []
        if unknown:
            problems.append(f"processor keys TRL does not batch: {sorted(unknown)}")
        if image_token_id is None or processor_token_id != image_token_id:
            problems.append(f"config.image_token_id={image_token_id}, processor image_token_id={processor_token_id}")
        if runs != 2:
            problems.append(f"{runs} runs of image tokens for 2 images (vLLM server mode expects one per image)")
        return (
            ("GAP", "; ".join(problems))
            if problems
            else ("PASS", f"keys {sorted(set(out) - {'input_ids', 'attention_mask'})}")
        )

    run("VL", "two images render and match TRL's processor contract", "TRL/provider", check_images)

# --- Report ---
print("\n| Area | Check | Status | Detail | Owner |\n|---|---|---|---|---|")
for row in rows:
    print("| " + " | ".join(row) + " |")
counts = {s: sum(r[2] == s for r in rows) for s in ("PASS", "FAIL", "GAP", "INFO")}
print(f"\n{counts}")
