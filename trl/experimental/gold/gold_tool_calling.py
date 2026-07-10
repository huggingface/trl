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
#     "smolagents[toolkit]",
#     "docling",
#     "trackio",
#     "vllm",
#     "liger-kernel",
# ]
# ///

# docstyle-ignore
"""
Direct tool-calling distillation with GOLD (no separate SFT-teacher stage): distil a small student from an
off-the-shelf instruct teacher of the same family. Student and teacher share a tokenizer and the model's native
tool-calling chat template, so same-family tool distillation works out of the box — no custom template and no step-1
SFT are needed. Combined on/off-policy training (`lmbda=0.5`): off-policy slices consume the dataset's recorded tool
trajectories, on-policy slices let the student generate and *actually execute* the tools. Requires transformers >=
5.2.0.

Both modes generate through vLLM in colocate mode, so both need a GPU + vLLM. Two modes, selected with `--mode`:

  # Text (Qwen3): student Qwen/Qwen3-1.7B, teacher Qwen/Qwen3-8B, browser-agent tool dataset.
  accelerate launch trl/experimental/gold/gold_tool_calling.py --mode text

  # VLM (Qwen3-VL): student Qwen/Qwen3-VL-2B-Instruct, teacher Qwen/Qwen3-VL-8B-Instruct, Search-VL dataset. Tools
  # are genuine web/wiki search plus docling `layout_parsing` (an environment method, so it can resolve image refs).
  accelerate launch trl/experimental/gold/gold_tool_calling.py --mode vlm

NOTE: Qwen3's chat template is not prefix-preserving (it drops historical `<think>` blocks). `GOLDTrainer` detects
this and renders all tool-flow spans with a training-safe prefix-preserving template, so the tool-result masking stays
aligned; nothing extra is needed here.
"""

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.utils import get_json_schema

from trl.experimental.gold import GOLDConfig, GOLDTrainer


BROWSER_DATASET = "DataCreatorAI/tool-calling-browser-agent-tasks"

SEARCH_VL_DATASET = "OpenSearch-VL/Search-VL-SFT-36K"
SEARCH_VL_DOMAINS = {
    "fvqa": "fvqa/fvqa_llama_factory_clean.json",
    "livevqa": "livevqa/livevqa_llama_factory_filtered.json",
    "webqa": "webqa/webqa_llama_factory_filtered.json",
}
# Search-VL tools whose results are text (GOLD-compatible). Conversations using image-returning tools
# (`crop`, `super_resolution`, reverse-image `image_search`) are dropped.
VLM_ALLOWED_TOOLS = {"text_search", "web_search", "layout_parsing"}


# Text tools: a coherent, well-typed subset of the browser dataset's 46 tools that the on-policy student can call.
# GOLD introspects each callable's signature + docstring to render the schema; results are masked out of the loss.
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

    return [
        browser_open,
        form_fill,
        form_select,
        form_upload,
        form_submit,
        form_validate,
    ]


# VLM tools: real, key-free web/wikipedia search (smolagents), wrapped as type-hinted callables for schema
# introspection. `layout_parsing` is an environment method (see LayoutParsingEnv) so it can resolve image references.
def build_search_tools():
    from smolagents import DuckDuckGoSearchTool, WikipediaSearchTool

    _web = DuckDuckGoSearchTool()
    _wiki = WikipediaSearchTool()

    def web_search(q: str, hl: str = "en") -> str:
        """
        Perform a web search and return the top results as text.

        Args:
            q: The search query keywords.
            hl: Language code for the results (e.g. 'en').

        Returns:
            The top web results, concatenated as text.
        """
        return str(_web(q))

    def text_search(q: str, hl: str = "en", top_k: int = 5) -> str:
        """
        Search encyclopedic knowledge (Wikipedia) for a query and return a text summary.

        Args:
            q: The search query keywords.
            hl: Language code for the results (e.g. 'en').
            top_k: Number of passages to retrieve.

        Returns:
            A text summary of the most relevant article(s).
        """
        return str(_wiki(q))

    return [web_search, text_search]


class LayoutParsingEnv:
    """Environment exposing docling-based `layout_parsing` as a tool.

    Registered via `environment_factory`, so GOLD calls `reset(**example)` per rollout with the raw example dict (which
    carries the prompt images). `layout_parsing` resolves the model's symbolic `"img_1"` reference to the actual PIL
    image and runs docling's smallest document pipeline on it.
    """

    def reset(self, **example):
        self._images = example.get("images") or []
        self._converter = None
        # Per-instance scratch dir: environment instances are pooled and run concurrently within a batch, so a shared
        # fixed path (e.g. /tmp/layout_0.png) would race between rollouts resolving the same image index.
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="gold_layout_"))

    def layout_parsing(self, image: str) -> str:
        """
        Parse document layout from an image reference and return the extracted structured text.

        Args:
            image: Image reference such as 'img_1' or 'img_2'.

        Returns:
            The extracted text (titles, paragraphs, footnotes).
        """
        from docling.document_converter import DocumentConverter

        idx = int(re.search(r"\d+", image).group()) - 1 if re.search(r"\d+", image) else 0
        idx = max(0, min(idx, len(self._images) - 1))
        tmp = self._tmp_dir / f"layout_{idx}.png"
        self._images[idx].save(tmp)
        if self._converter is None:
            self._converter = DocumentConverter()  # TODO: configure the smallest docling pipeline for speed
        return self._converter.convert(str(tmp)).document.export_to_markdown()


def normalize_messages(messages):
    """Merge consecutive assistant turns into one (contents joined, `tool_calls` concatenated); return `None` for the
    rare `tool -> user` conversation so the caller can skip it."""
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
    """Fold each assistant turn's structured `tool_calls` into its `content` as Qwen3's native `<tool_call>` text. The
    all-string-`content` result renders byte-identically under Qwen3's template but avoids the Arrow null-padding a
    heterogeneous `arguments` struct would incur in a `Dataset`."""
    inlined = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            parts = [message.get("content") or ""]
            for tool_call in message["tool_calls"]:
                function = tool_call["function"]
                call = {"name": function["name"], "arguments": function["arguments"]}
                parts.append("<tool_call>\n" + json.dumps(call) + "\n</tool_call>")
            inlined.append(
                {
                    "role": "assistant",
                    "content": "\n".join(part for part in parts if part),
                }
            )
        else:
            inlined.append({"role": message["role"], "content": message.get("content") or ""})
    return inlined


def build_browser_dataset(max_conversations=None):
    """DataCreatorAI/tool-calling-browser-agent-tasks (browser-form subset) -> GOLD prompt/completion + tools.

    Kept to conversations whose tool calls are all `build_browser_tools`, then split at each tool-bearing user turn:
    the prompt is the *full* conversation history up to and including that user turn, and the completion is the
    assistant tool-call turn (plus its tool result and follow-up) up to the next user turn. Carrying the whole prefix
    (rather than slicing at the user turn) keeps the context a mid-conversation turn like "I'd give it a 4" refers to;
    these conversations are short (<1.3k tokens), so length is not a concern. Tool schemas are stored as a JSON string
    to avoid Arrow null-padding.
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
            # Split at each user turn: prompt = full history up to and including this user turn, completion = the
            # assistant response (tool call + result + follow-up) up to the next user turn.
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


def build_vlm_dataset(domains, max_conversations=None):
    """OpenSearch-VL/Search-VL-SFT-36K (text-returning-tool subset) -> GOLD prompt/completion with images + tools.

    Kept to conversations whose tools all return text (`text_search`, `web_search`, `layout_parsing`) and whose images
    live only in the prompt. Search-VL already records its tool calls in Qwen3's native `<tool_call>` format, so no
    reformatting is needed. Tool schemas are stored as a JSON string to avoid Arrow null-padding.
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image as PILImage

    rows = []
    for domain in domains:
        json_path = hf_hub_download(SEARCH_VL_DATASET, SEARCH_VL_DOMAINS[domain], repo_type="dataset")
        zip_path = hf_hub_download(SEARCH_VL_DATASET, f"{domain}/images.zip", repo_type="dataset")
        extract_dir = Path(zip_path).parent / f"{domain}_images"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        kept = 0
        for rec in json.load(open(json_path)):
            if max_conversations is not None and kept >= max_conversations:
                break
            conv = rec["conversations"]
            names = [n for m in conv if m["from"] == "gpt" for n in re.findall(r'"name":\s*"([^"]+)"', m["value"])]
            if not names or not set(names) <= VLM_ALLOWED_TOOLS:
                continue
            if any(m["from"] == "observation" and "<image>" in m["value"] for m in conv):
                continue

            tools = rec["tools"]
            tools = json.loads(tools) if isinstance(tools, str) else tools
            tools = [t for t in tools if t.get("function", t)["name"] in VLM_ALLOWED_TOOLS]

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": rec["system"].strip()}],
                }
            ]
            completion, n_images = [], 0
            for m in conv:
                if m["from"] == "human":
                    n = m["value"].count("<image>")
                    n_images += n
                    blocks = [{"type": "image"}] * n
                    text = m["value"].replace("<image>", "").strip()
                    if text:
                        blocks.append({"type": "text", "text": text})
                    prompt.append({"role": "user", "content": blocks})
                elif m["from"] == "gpt":
                    completion.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": m["value"].strip()}],
                        }
                    )
                elif m["from"] == "observation":
                    obs = re.sub(r"</?observation>", "", m["value"]).strip()
                    completion.append({"role": "tool", "content": [{"type": "text", "text": obs}]})

            image_paths = [extract_dir / p for p in rec["images"]]
            if len(image_paths) != n_images or not all(p.exists() for p in image_paths):
                continue

            rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "images": [PILImage.open(p).convert("RGB") for p in image_paths],
                    "tools": json.dumps(tools),
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )
            kept += 1
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vlm"], default="text")
    parser.add_argument("--student_model_name", type=str, default=None)
    parser.add_argument("--teacher_model_name", type=str, default=None)
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="trackio")
    cli_args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Models, tools and dataset
    # ──────────────────────────────────────────────
    if cli_args.mode == "text":
        student_id = cli_args.student_model_name or "Qwen/Qwen3-1.7B"
        teacher_id = cli_args.teacher_model_name or "Qwen/Qwen3-8B"
        processing_class = AutoTokenizer.from_pretrained(student_id, padding_side="left")
        student_model = AutoModelForCausalLM.from_pretrained(student_id, dtype=torch.bfloat16)
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_id, dtype=torch.bfloat16)
        train_dataset = build_browser_dataset(cli_args.max_conversations)
        tools, environment_factory = build_browser_tools(), None
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj"])
    else:
        student_id = cli_args.student_model_name or "Qwen/Qwen3-VL-2B-Instruct"
        teacher_id = cli_args.teacher_model_name or "Qwen/Qwen3-VL-8B-Instruct"
        processing_class = AutoProcessor.from_pretrained(student_id, padding_side="left")
        student_model = AutoModelForImageTextToText.from_pretrained(student_id, dtype=torch.bfloat16)
        teacher_model = AutoModelForImageTextToText.from_pretrained(teacher_id, dtype=torch.bfloat16)
        for name, param in student_model.named_parameters():
            if "language_model" not in name:
                param.requires_grad = False
        train_dataset = build_vlm_dataset(list(SEARCH_VL_DOMAINS), cli_args.max_conversations)
        tools, environment_factory = build_search_tools(), LayoutParsingEnv
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=r"^.*language_model.*\.(q_proj|k_proj)$",
        )

    dataset = train_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # ──────────────────────────────────────────────
    # Training config
    # ──────────────────────────────────────────────
    student_short = student_id.split("/")[-1]
    teacher_short = teacher_id.split("/")[-1]
    run_name = cli_args.run_name or f"gold-{student_short}-from-{teacher_short}-{cli_args.mode}-tools"
    output_dir = cli_args.output_dir or run_name

    args = GOLDConfig(
        output_dir=output_dir,
        run_name=run_name,
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.6,
        max_completion_length=2048 * 2,
        max_grad_norm=1.0,
        teacher_model_name_or_path=teacher_id,
        num_generations=1,
        use_vllm=True,
        use_liger_kernel=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=8192,
        max_length=8192,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=1e-4,
        warmup_steps=10,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=25,
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
        processing_class=processing_class,
        peft_config=peft_config,
        tools=tools,
        environment_factory=environment_factory,
    )
    # Qwen3 reasons by default; `enable_thinking=False` makes its template emit an empty `<think></think>` block so
    # the model goes straight to the tool call. On-policy generation renders with `self.chat_template_kwargs`
    # (teacher and student share the tokenized prompt, so this covers both). For off-policy slices the collator reads
    # a per-example `chat_template_kwargs` column instead, set in the dataset builders.
    trainer.chat_template_kwargs = {"enable_thinking": False}
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
