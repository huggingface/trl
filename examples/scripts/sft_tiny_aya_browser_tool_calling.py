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

# /// script
# dependencies = [
#     "trl[peft]",
#     "liger-kernel",
#     "trackio",
# ]
# ///

"""
Teach multi-turn, agentic tool calling to CohereLabs/tiny-aya-global (a ~3.3B "smol Aya") with SFT + LoRA on the
DataCreatorAI/tool-calling-browser-agent-tasks dataset (1,062 multi-turn browser-agent conversations, 46 tools).

Variant of `sft_tiny_aya_tool_calling.py`, tuned for a single 80 GB A100 and a real multi-turn dataset:

  * **No 4-bit quantization, bf16 throughout.** tiny-aya-global is ~3.3B params (native bfloat16); its bf16 weights are
    ~6.7 GB, so a bf16 LoRA fits an 80 GB A100 with room to spare. QLoRA is a small-GPU memory workaround that costs
    quality — which matters because this model is a *teacher* for a later distillation stage.
  * **Multi-turn masking by unrolling.** The dataset is genuinely multi-turn (user -> assistant tool_call -> tool
    result -> assistant ...). The tiny-aya chat template has no `{% generation %}` markers, so `assistant_only_loss` is
    unavailable. Instead each conversation is unrolled into one prompt/completion example per assistant turn: every
    user and tool-result turn lands in the (masked) prompt, and exactly one assistant turn (a tool call or a final
    answer) is supervised per example. `completion_only_loss` then masks the prompt. This is the same technique the
    SmolVLM Stage-1 script uses.
  * **Per-conversation tool schemas.** The 46 tools are polymorphic umbrella actions (e.g. `web_search` appears with
    200+ distinct argument keys), so a precise JSON Schema is neither available nor meaningful. Each tool is rendered
    with a short description and a permissive `{"type": "object"}` parameter object; only the tools a given
    conversation actually calls are exposed in its prompt.
  * **Tool calls inlined into assistant text.** Each assistant turn's structured `tool_calls` are rendered to the
    template's `<tool_call>...</tool_call>` text and folded into that turn's `content`, and the structured field is
    dropped. This is deliberate: the tool calls carry heterogeneous argument keys, and a HF `Dataset` (Arrow) would
    unify `tool_calls.arguments` into a single struct with the union of every key seen, null-filling the rest —
    training the model to emit spurious `<parameter=x>null</parameter>` lines. With all-string `content`, the stored
    messages stay conversational (so SFT handles BOS/EOS correctly) yet render byte-identically to the structured
    form. The tool *schemas* are still passed via a `tools` column so the template renders them in the system preamble.

The tool-aware chat template is saved with the tokenizer, so inference only requires loading the tokenizer from the
output directory and calling `apply_chat_template(..., tools=...)`.

Example:

    python examples/scripts/sft_tiny_aya_browser_tool_calling.py
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer


# The 46 umbrella tools in the dataset are polymorphic (free-form arguments), so each gets a concise description and a
# permissive parameter object. The description teaches the model which tool to pick; the arguments are learned from the
# supervised completions.
TOOL_DESCRIPTIONS = {
    "web_search": "Search the web for information matching the given criteria.",
    "manage_booking": "Create, modify, or cancel a booking such as travel, hotel, or an appointment.",
    "app_open": "Open an application, file, workspace, or page.",
    "create_document": "Create a new document, spreadsheet, chart, or file.",
    "financial_services": "Perform a financial operation such as a payment, transfer, loan, or insurance action.",
    "resource_find": "Find a resource (file, board, card, document) matching search criteria.",
    "form_fill": "Fill in the fields of a form.",
    "content_processing": "Process content: convert, extract, compress, or transform files and documents.",
    "browser_open": "Open a URL or page in the browser.",
    "form_submit": "Submit a completed form.",
    "form_validate": "Validate a form's fields before submission.",
    "content_update": "Update the content of an existing document or resource.",
    "file_management": "Manage files: move, rename, organize, back up, or delete.",
    "action_execute": "Execute an action on a resource (comment, move, commit, delete, and similar).",
    "summarize_text": "Summarize one or more pieces of text.",
    "data_lookup": "Look up data or records from a data source.",
    "resource_edit": "Edit an existing resource such as a document, design, or repository file.",
    "schedule_management": "Create or manage schedules, reminders, and calendar events.",
    "form_select": "Select values for choice fields in a form.",
    "form_upload": "Upload files to a form field.",
    "messaging_communication": "Send or manage messages, emails, and other communications.",
    "data_analysis": "Analyze data and produce metrics or a report.",
    "resource_create": "Create a new resource such as a board, card, repository, or design.",
    "save_to_digilocker": "Save a document to DigiLocker.",
    "social_media": "Post to or manage social media accounts.",
    "upload_file": "Upload a file to a service or destination.",
    "form_automation": "Automatically fill and submit a form from provided details.",
    "shopping_ecommerce": "Search, compare, or purchase products from online stores.",
    "file_download": "Download or export a file.",
    "file_upload": "Upload a file to a target application or resource.",
    "account_management": "Manage account settings and profile information.",
    "security_privacy": "Manage security and privacy settings.",
    "recommend_item": "Recommend items matching the user's preferences.",
    "send_message": "Send a message to a recipient or channel.",
    "document_creation": "Create a document, project, or collection.",
    "book_service": "Book a service such as travel, a subscription, or an appointment.",
    "share_send": "Share or send content to recipients.",
    "data_extract": "Extract structured data from a source.",
    "manage_account": "Manage account items such as cards, documents, or profile details.",
    "resource_open": "Open a resource in an application.",
    "process_upi_payment": "Process a UPI payment.",
    "device_control": "Control a device or set device preferences.",
    "schedule_event": "Schedule an event, task, or reminder.",
    "download_file": "Download a file from a URL.",
    "file_save": "Save a file to a destination folder.",
    "fill_passenger_form": "Fill a passenger details form.",
}


def _humanize(name):
    return name.replace("_", " ").capitalize() + "."


def _tool_schema(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS.get(name, _humanize(name)),
            # Polymorphic arguments: keep the parameter object permissive rather than inventing a fixed schema.
            "parameters": {"type": "object"},
        },
    }


def _conversation_tools(messages):
    """Ordered, de-duplicated schemas for the tools a single conversation actually calls."""
    names = []
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            name = tool_call["function"]["name"]
            if name not in names:
                names.append(name)
    return [_tool_schema(name) for name in names]


def normalize_messages(messages):
    """Make a conversation render under the tiny-aya template, which requires strictly alternating turns.

    Agentic data often splits one assistant turn across several messages (reasoning text, then the tool call —
    ~290 `assistant -> assistant` pairs here), which trips the template's alternation guard. Merge consecutive
    assistant messages into a single turn (contents joined, `tool_calls` concatenated) — the same content/then/calls
    order the template renders anyway. Returns `None` for the rare conversation the template still can't render (a
    stray `tool -> user`), so the caller can skip it. Consecutive `tool` messages are left as-is; the template
    already allows them.
    """
    merged = []
    for message in messages:
        if message["role"] == "assistant" and merged and merged[-1]["role"] == "assistant":
            prev = merged[-1]
            contents = [c for c in (prev.get("content"), message.get("content")) if c]
            prev["content"] = "\n\n".join(contents)
            calls = (prev.get("tool_calls") or []) + (message.get("tool_calls") or [])
            if calls:
                prev["tool_calls"] = calls
        else:
            merged.append(dict(message))

    roles = [m["role"] for m in merged]
    if any(a == "tool" and b == "user" for a, b in zip(roles, roles[1:], strict=False)):
        return None
    return merged


def inline_tool_calls(messages, tokenizer):
    """Fold each assistant turn's structured `tool_calls` into its `content` as the template's rendered tool-call text,
    dropping the structured field. The result is an all-string-`content` conversation that renders identically but does
    not carry a heterogeneous `arguments` struct into the (Arrow-backed) `Dataset`.
    """
    inlined = []
    for message in messages:
        if message["role"] != "assistant" or not message.get("tool_calls"):
            inlined.append({"role": message["role"], "content": message.get("content") or ""})
            continue
        # Render just this assistant turn (behind a dummy user turn) and lift out the response body between the
        # template's response delimiters — that body is exactly `content` followed by the rendered tool calls.
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}, message], tokenize=False, add_generation_prompt=False
        )
        body = rendered.split("<|START_RESPONSE|>", 1)[1].rsplit("<|END_RESPONSE|>", 1)[0]
        inlined.append({"role": "assistant", "content": body})
    return inlined


def load_conversations(dataset_name):
    """Read the raw JSONL as native Python objects so each tool call keeps its own clean argument dict (see the
    module docstring: going through a HF `Dataset` here would null-pad the arguments)."""
    path = hf_hub_download(dataset_name, "browser_tasks_final.jsonl", repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_split(conversations, tokenizer):
    """Unroll each multi-turn conversation into one prompt/completion example per assistant turn.

    For an assistant turn at index `i`, the prompt is `messages[:i]` (every prior user, assistant, and tool-result
    turn — all masked) and the completion is the single assistant turn `messages[i]` (supervised). Every assistant turn
    is therefore supervised exactly once, and tool-result tokens never enter the loss.
    """
    examples = []
    for row in conversations:
        messages = normalize_messages(row["messages"])
        if messages is None:
            continue
        tools = _conversation_tools(messages)  # from the structured tool_calls, before they are inlined
        messages = inline_tool_calls(messages, tokenizer)
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            examples.append({"prompt": messages[:i], "completion": [message], "tools": tools})
    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_to", type=str, default="trackio", help="Experiment tracker, e.g. 'wandb' or 'trackio'.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the trained model to the Hub when done.")
    cli_args = parser.parse_args()

    model_id = "CohereLabs/tiny-aya-global"
    dataset_name = "DataCreatorAI/tool-calling-browser-agent-tasks"
    output_dir = "tiny-aya-global-browser-tool-calling-SFT"

    # Tokenizer carries the tool-aware chat template (used to inline the tool calls and saved for inference).
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = (Path(__file__).parent / "tiny_aya_chat_template.jinja").read_text()

    # Split at the conversation level (so turns from one conversation never straddle the train/eval boundary), then
    # unroll each side into per-assistant-turn examples.
    conversations = load_conversations(dataset_name)
    n_eval = max(1, round(len(conversations) * 0.05))
    rng = random.Random(42)
    rng.shuffle(conversations)
    eval_dataset = build_split(conversations[:n_eval], tokenizer)
    train_dataset = build_split(conversations[n_eval:], tokenizer)

    # Load model in bf16, no quantization (fits an 80 GB A100 comfortably; better quality than QLoRA for a teacher).
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )

    # LoRA on all attention + MLP projections. r=32 / alpha=64 (alpha = 2*r) is a solid default; LoRA also acts as a
    # regularizer, which suits a ~6k-example fine-tune better than full fine-tuning would.
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        # Mask the prompt (user + tool results + history); supervise only the single assistant completion.
        completion_only_loss=True,
        max_length=4096,
        # 80 GB A100 budget: bf16, no activation offloading (that trades speed for memory we have to spare).
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        use_liger_kernel=True,
        num_train_epochs=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=cli_args.report_to,
        trackio_space_id=output_dir if cli_args.report_to == "trackio" else None,
        push_to_hub=cli_args.push_to_hub,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # carries the tool-aware chat template; saved for inference
        peft_config=peft_config,
    )
    trainer.train()

    # Save model and tokenizer (tokenizer carries the tool-aware chat template).
    trainer.save_model(output_dir)
    if cli_args.push_to_hub:
        trainer.push_to_hub(dataset_name=dataset_name)


if __name__ == "__main__":
    main()
