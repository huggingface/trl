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
#     "trl[peft] @ git+https://github.com/huggingface/trl.git",
#     "trackio",
#     "vllm",
#     "liger-kernel",
# ]
# ///

# docstyle-ignore
"""
# Tool-calling distillation on a browser-agent dataset (Qwen3-8B -> Qwen3-1.7B).
# Same family: student and teacher share the tokenizer and Qwen3's native tool-calling template, so distillation
# works directly with no custom template and no stage-1 SFT. Pure on-policy (lmbda=1.0): the student generates and
# *actually executes* the browser tools, and the teacher scores the student's own rollouts. Runs generation through
# vLLM (colocate), so a GPU + vLLM is required.
accelerate launch trl/experimental/gold/gold_tool_calling.py \
    --student_model_name Qwen/Qwen3-1.7B \
    --teacher_model_name Qwen/Qwen3-8B
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import get_json_schema

from trl.experimental.gold import GOLDConfig, GOLDTrainer


BROWSER_DATASET = "DataCreatorAI/tool-calling-browser-agent-tasks"


# The six browser-form tools the student may call. GOLD introspects each callable's type hints + docstring to build
# the tool schema, so both are required; the tool *results* are masked out of the loss (only tool calls are trained).
def build_browser_tools():
    def browser_open(url: str) -> str:
        """
        Open a URL in the browser.

        Args:
            url: The URL to open.

        Returns:
            A short status message.
        """
        return f"Opened {url}"

    def form_fill(fields: list) -> str:
        """
        Fill one or more form fields with values.

        Args:
            fields: List of field entries (each with a name and a value) to fill in.

        Returns:
            A short status message.
        """
        return f"Filled {len(fields)} field(s)."

    def form_select(field_name: str, value: str) -> str:
        """
        Select a value for a single choice or dropdown form field.

        Args:
            field_name: The name of the field to set.
            value: The value to select.

        Returns:
            A short status message.
        """
        return f"Selected {value!r} for {field_name}."

    def form_upload(field_name: str, file_path: str) -> str:
        """
        Attach a file to a form upload field.

        Args:
            field_name: The upload field to attach to.
            file_path: Path of the file to upload.

        Returns:
            A short status message.
        """
        return f"Uploaded {file_path} to {field_name}."

    def form_submit() -> str:
        """
        Submit the currently filled form.

        Returns:
            A short status message.
        """
        return "Form submitted successfully."

    def form_validate() -> str:
        """
        Validate the currently filled form before submission.

        Returns:
            A short status message.
        """
        return "Form is valid."

    return [browser_open, form_fill, form_select, form_upload, form_submit, form_validate]


def normalize_messages(messages):
    """Merge consecutive assistant turns into one; return `None` for the rare `tool -> user` conversation to skip it."""
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


def inline_tool_calls(messages):
    """Fold each assistant turn's structured `tool_calls` into its `content` as Qwen3's native `<tool_call>` text.
    This renders byte-identically but keeps `content` a plain string, avoiding the Arrow null-padding a heterogeneous
    `arguments` struct would otherwise incur when stored in a `Dataset`."""
    inlined = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            parts = [message.get("content") or ""]
            for tool_call in message["tool_calls"]:
                function = tool_call["function"]
                call = {"name": function["name"], "arguments": function["arguments"]}
                parts.append("<tool_call>\n" + json.dumps(call) + "\n</tool_call>")
            inlined.append({"role": "assistant", "content": "\n".join(part for part in parts if part)})
        else:
            inlined.append({"role": message["role"], "content": message.get("content") or ""})
    return inlined


def build_browser_dataset(max_conversations=None):
    """Browser-agent tasks -> GOLD prompt/completion + tools, kept to conversations that only use our six tools.

    Split at each tool-bearing user turn: the prompt is the *full* conversation history up to and including that user
    turn (so a mid-conversation reference like "I'd give it a 4" stays resolvable), and the completion is the assistant
    tool call plus its result and follow-up up to the next user turn. `tools` is stored as a JSON string (again to
    avoid Arrow null-padding), and `enable_thinking=False` is stored per row for the off-policy rendering path.
    """
    from huggingface_hub import hf_hub_download

    browser_tools = build_browser_tools()
    allowed = {tool.__name__ for tool in browser_tools}
    tools_schema = json.dumps([get_json_schema(tool) for tool in browser_tools])
    path = hf_hub_download(BROWSER_DATASET, "browser_tasks_final.jsonl", repo_type="dataset")

    rows, kept = [], 0
    with open(path) as f:
        for line in f:
            if max_conversations is not None and kept >= max_conversations:
                break
            messages = json.loads(line)["messages"]
            used = {tc["function"]["name"] for m in messages for tc in (m.get("tool_calls") or [])}
            if not used or not used <= allowed:
                continue
            messages = normalize_messages(messages)
            if messages is None:
                continue
            messages = inline_tool_calls(messages)
            user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
            for ui, next_ui in zip(user_indices, user_indices[1:] + [len(messages)], strict=True):
                completion = messages[ui + 1 : next_ui]
                if not any(m["role"] == "tool" for m in completion):
                    continue  # no tool result in this turn -> nothing tool-related to distil
                rows.append(
                    {
                        "prompt": messages[: ui + 1],
                        "completion": completion,
                        "tools": tools_schema,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
            kept += 1
    return Dataset.from_list(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="trackio")
    cli_args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cli_args.student_model_name, padding_side="left")
    student_model = AutoModelForCausalLM.from_pretrained(cli_args.student_model_name, dtype=torch.bfloat16)
    teacher_model = AutoModelForCausalLM.from_pretrained(cli_args.teacher_model_name, dtype=torch.bfloat16)

    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj"])

    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
    dataset = build_browser_dataset(cli_args.max_conversations).train_test_split(test_size=0.05, seed=42)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # ──────────────────────────────────────────────
    # Training config
    # ──────────────────────────────────────────────
    student_short = cli_args.student_model_name.split("/")[-1]
    teacher_short = cli_args.teacher_model_name.split("/")[-1]
    run_name = cli_args.run_name or f"gold-{student_short}-from-{teacher_short}-tools"

    args = GOLDConfig(
        output_dir=run_name,
        run_name=run_name,
        # GOLD-specific
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.6,
        max_completion_length=1024,
        max_grad_norm=1.0,
        teacher_model_name_or_path=cli_args.teacher_model_name,
        num_generations=1,
        max_tool_calling_iterations=10,  # cap the on-policy tool loop so a stuck rollout can't run to the token limit
        # vLLM + fused loss
        use_vllm=True,
        use_liger_kernel=True,  # fused JSD avoids materializing full-vocab logits (Qwen3 vocab is ~150k)
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=8192,
        max_length=8192,
        # Training schedule
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=1e-4,
        warmup_steps=10,
        # Evaluation
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=25,
        # Precision / logging
        bf16=True,
        logging_steps=10,
        log_completions=True,
        log_completions_steps=1,
        report_to=cli_args.report_to,
    )

    # ──────────────────────────────────────────────
    # Trainer
    # ──────────────────────────────────────────────
    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        tools=build_browser_tools(),
    )
    # Qwen3 reasons by default; disable it so the model emits the tool call directly. On-policy generation reads
    # `chat_template_kwargs` (student and teacher share the tokenized prompt); off-policy reads the per-row column.
    trainer.chat_template_kwargs = {"enable_thinking": False}
    trainer.train()
    trainer.save_model(args.output_dir)
