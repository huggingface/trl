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

Two modes, selected with `--mode`:

  # Text (Qwen3): student Qwen/Qwen3-1.7B, teacher Qwen/Qwen3-8B, browser-agent tool dataset. Generation runs
  # through transformers (no vLLM).
  accelerate launch trl/experimental/gold/gold_tool_calling.py --mode text

  # VLM (Qwen3-VL): student Qwen/Qwen3-VL-2B-Instruct, teacher Qwen/Qwen3-VL-8B-Instruct, Search-VL dataset. Tools
  # are genuine web/wiki search plus docling `layout_parsing` (an environment method, so it can resolve image refs).
  # The VLM tool loop generates through vLLM, so this mode needs a GPU + vLLM.
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


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# Text tools: the browser-form subset. The full browser dataset uses 46 mostly polymorphic tools (free-form
# arguments) that cannot be given genuine typed signatures; this coherent subset can, so the on-policy student can
# actually call them. GOLD introspects each callable's signature + docstring to render the tool schema. Tool
# *results* are masked out of the GOLD loss, so the stub bodies only need to keep the multi-turn rollout moving.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# VLM tools (executed on on-policy slices). smolagents ships real, key-free web/wikipedia search; we wrap each
# smolagents Tool in a plain type-hinted callable because GOLD introspects the callable's signature + docstring to
# render the tool schema. `layout_parsing` is an environment method (see LayoutParsingEnv) so it can resolve image
# references. Tool *results* are masked out of the GOLD loss, so genuine-but-noisy results are fine.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
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

    Registered via `environment_factory`, so GOLD calls `reset(**example)` per rollout with the raw example dict
    (which carries the prompt images). `layout_parsing` resolves the model's symbolic `"img_1"` reference to the
    actual PIL image and runs docling's smallest document pipeline on it.
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
    """Merge consecutive assistant turns into a single turn (contents joined, `tool_calls` concatenated). Agentic
    data often splits one assistant turn across several messages (reasoning text, then the tool call); merging keeps
    each rendered turn clean. Returns `None` for the rare `tool -> user` conversation, so the caller can skip it.
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


def inline_tool_calls(messages):
    """Fold each assistant turn's structured `tool_calls` into its `content` as Qwen3's native `<tool_call>` text,
    dropping the structured field. The result is an all-string-`content` conversation that renders byte-identically
    under Qwen3's template but does not carry a heterogeneous `arguments` struct into the (Arrow-backed) `Dataset`
    (which would null-pad the union of every tool's argument keys).
    """
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
    """DataCreatorAI/tool-calling-browser-agent-tasks (browser-form subset) -> GOLD `messages` + tools.

    Restricted to conversations whose every tool call is one of the six `build_browser_tools`, so the on-policy
    student can execute them. GOLD (like GRPO) supports the single-user-turn agentic shape: one user prompt, then an
    assistant/tool trajectory. The browser conversations are genuinely multi-turn (the user comes back with
    follow-ups), so each conversation is **segmented at its user turns**: every `[user, assistant, tool, ...]` slice up
    to the next user turn that contains a tool call becomes its own record. `GOLDTrainer` then splits each `messages`
    record at the first assistant turn (prompt vs. completion) and masks the tool-result tokens; the assistant turns
    (including the tool call) are supervised. A later segment loses the earlier conversation as context — the price of
    fitting GOLD's single-user-turn shape without a trainer change, and cheaper than carrying a long masked prefix.

    The raw JSONL is read directly (a HF `Dataset` would null-pad the heterogeneous tool arguments), consecutive
    assistant turns are merged, and each assistant turn's structured `tool_calls` are inlined into Qwen3 `<tool_call>`
    text — so no heterogeneous struct reaches the Arrow-backed dataset. The tool schemas are stored as a JSON string
    for the same reason (GOLD `json.loads` a string `tools` column).
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
            # Segment at user turns: each [user, assistant, tool, ...] slice up to the next user turn is one record.
            user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
            for start, end in zip(user_indices, user_indices[1:] + [len(messages)], strict=True):
                segment = messages[start:end]
                if not any(m["role"] == "tool" for m in segment):
                    continue  # no tool result in this turn -> nothing tool-related to distil
                rows.append({"messages": segment, "tools": tools_schema})
            kept += 1
    return Dataset.from_list(rows)


def build_vlm_dataset(domains, max_conversations=None):
    """OpenSearch-VL/Search-VL-SFT-36K (text-returning-tool subset) -> GOLD prompt/completion with images + tools.

    Search-VL already records its tool calls in Qwen3's native `<tool_call>{...}</tool_call>` format, so no
    reformatting is needed. We keep conversations whose tools all return text (`text_search`, `web_search`,
    `layout_parsing`, executed on-policy) and whose images live only in the prompt. Each conversation is a single user
    turn followed by the assistant/tool trajectory, so the prompt is the system + image question and the completion is
    the trajectory; `GOLDTrainer` masks the tool-result tokens. Tool schemas are stored as a JSON string to avoid
    Arrow null-padding the heterogeneous argument keys.
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

            prompt = [{"role": "system", "content": [{"type": "text", "text": rec["system"].strip()}]}]
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
                    completion.append({"role": "assistant", "content": [{"type": "text", "text": m["value"].strip()}]})
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
                }
            )
            kept += 1
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "vlm"], default="text")
    parser.add_argument("--student_model_name", type=str, default=None, help="Defaults to the base model per mode.")
    parser.add_argument("--teacher_model_name", type=str, default=None, help="Defaults to the instruct teacher per mode.")
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="trackio", help="Experiment tracker, e.g. 'wandb' or 'trackio'.")
    cli_args = parser.parse_args()

    # LoRA on all attention + MLP projections (student only; the teacher is frozen). Both models load in bf16 with no
    # 4-bit quant — the student/teacher pairs are small enough to fit an 80 GB A100 alongside generation.
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if cli_args.mode == "text":
        student_id = cli_args.student_model_name or "Qwen/Qwen3-1.7B"
        teacher_id = cli_args.teacher_model_name or "Qwen/Qwen3-8B"
        processing_class = AutoTokenizer.from_pretrained(student_id)
        student = AutoModelForCausalLM.from_pretrained(student_id, dtype=torch.bfloat16)
        teacher = AutoModelForCausalLM.from_pretrained(teacher_id, dtype=torch.bfloat16)
        train_dataset = build_browser_dataset(cli_args.max_conversations)
        tools, environment_factory = build_browser_tools(), None
        use_vllm = False
        # Text tool trajectories: 8192-token window, 2048-token on-policy generation budget per rollout.
        max_length, max_completion_length = 8192, 2048
    else:
        student_id = cli_args.student_model_name or "Qwen/Qwen3-VL-2B-Instruct"
        teacher_id = cli_args.teacher_model_name or "Qwen/Qwen3-VL-8B-Instruct"
        processing_class = AutoProcessor.from_pretrained(student_id)
        student = AutoModelForImageTextToText.from_pretrained(student_id, dtype=torch.bfloat16)
        teacher = AutoModelForImageTextToText.from_pretrained(teacher_id, dtype=torch.bfloat16)
        train_dataset = build_vlm_dataset(list(SEARCH_VL_DOMAINS), cli_args.max_conversations)
        tools, environment_factory = build_search_tools(), LayoutParsingEnv
        use_vllm = True
        # Image placeholder tokens dominate the sequence, so the window is far larger: 16384 in, 4096 generated.
        max_length, max_completion_length = 16384, 4096

    args = GOLDConfig(
        output_dir=cli_args.output_dir or f"gold-qwen3-{cli_args.mode}-tools",
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.7,
        max_completion_length=max_completion_length,
        max_length=max_length,
        num_generations=2,
        # 80 GB A100 budget: bf16, effective batch 16 (4 x 4). If you OOM, drop per_device_train_batch_size first
        # (raise gradient_accumulation_steps to keep the effective batch at 16).
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=200,
        logging_steps=10,
        bf16=True,
        use_vllm=use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        report_to=cli_args.report_to,
    )

    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=train_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
        tools=tools,
        environment_factory=environment_factory,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
