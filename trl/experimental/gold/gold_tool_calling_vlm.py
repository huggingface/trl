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
# Multimodal tool-calling distillation on the Search-VL dataset (Qwen3-VL-8B -> Qwen3-VL-2B).
# Same family: student and teacher share the processor and Qwen3-VL's native tool-calling template. Pure on-policy
# (lmbda=1.0): the student generates and *actually executes* the tools (DuckDuckGo web search + docling
# `layout_parsing`), and the teacher scores the student's own rollouts. Runs generation through vLLM (colocate), so a
# GPU + vLLM is required.
#
# NOTE: this is a runnable demonstration, not a strong agent. Search-VL's competence comes from a proprietary
# search+summarize backend and a reasoning teacher; our stand-in tools (DuckDuckGo/docling) and the off-the-shelf
# instruct teacher are weaker, so expect modest tool-use quality. A high-value result needs a teacher specialized on
# the exact toolset, a faithful tool backend, and matching data.
accelerate launch trl/experimental/gold/gold_tool_calling_vlm.py \
    --student_model_name Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model_name Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import logging
import re
import tempfile
import zipfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from trl.experimental.gold import GOLDConfig, GOLDTrainer


SEARCH_VL_DATASET = "OpenSearch-VL/Search-VL-SFT-36K"
SEARCH_VL_DOMAINS = {
    "fvqa": "fvqa/fvqa_llama_factory_clean.json",
    "livevqa": "livevqa/livevqa_llama_factory_filtered.json",
    "webqa": "webqa/webqa_llama_factory_filtered.json",
}
# The dataset's tools whose results are text (GOLD-compatible). Conversations using image-returning tools
# (`crop`, `super_resolution`, reverse-image `image_search`) are dropped.
VLM_ALLOWED_TOOLS = {"text_search", "web_search", "layout_parsing"}

# Cap each tool result so a single search/parse can't flood the context (and blow the completion budget).
MAX_TOOL_RESULT_CHARS = 2000

# Search-VL ships an elaborate system prompt advertising tools we don't provide and a mandatory `<think>` protocol,
# which mismatches our three text tools and the non-reasoning Qwen3-VL-Instruct and sends the model into retry loops.
# Replace it with one describing exactly what the model has. Tool signatures are rendered from the `tools=` schema, so
# this is guidance only.
VLM_SYSTEM_PROMPT = (
    "You are a visual question answering agent with web-search tools. Look at the image and decide what the question "
    "needs:\n"
    "- If the answer is directly visible — objects, colors, counts, legible text, a value plotted on a chart — read "
    "it off the image and answer. Do not call a tool for something you can already see.\n"
    "- If the answer needs facts that are NOT in the image — identifying a place, person, or work, an event or date, "
    "or details of the paper or article the image is from — use `text_search` or `web_search`. Search for it rather "
    "than guessing from memory or replying that you lack information.\n"
    "- Use `layout_parsing` only to OCR dense printed text in a document image (paragraphs, tables, footnotes); it "
    "cannot read chart data or describe pictures. Pass the image reference `img_1`.\n"
    "All tool results are plain text, truncated to a few thousand characters. Call one tool per turn and wait for its "
    "result; if a result is empty or off-topic, refine the query once or answer with what you have — never repeat the "
    "same call or the same sentence. When ready, give a short final answer as plain text."
)


# Standalone tools: real, key-free web search (smolagents DuckDuckGo), wrapped as type-hinted callables so GOLD can
# build the schema. `web_search` and `text_search` are the two names the dataset uses; both route to DuckDuckGo
# because Wikipedia-only lookup dead-ended on the dataset's descriptive queries.
def build_search_tools():
    from smolagents import DuckDuckGoSearchTool

    _web = DuckDuckGoSearchTool()

    def web_search(q: str, hl: str = "en") -> str:
        """
        Perform a web search and return the top results as text.

        Args:
            q: The search query keywords.
            hl: Language code for the results (e.g. 'en').

        Returns:
            The top web results, concatenated as text.
        """
        return str(_web(q))[:MAX_TOOL_RESULT_CHARS]

    def text_search(q: str, hl: str = "en", top_k: int = 5) -> str:
        """
        Search the web for a query and return a text summary of the top results.

        Args:
            q: The search query keywords.
            hl: Language code for the results (e.g. 'en').
            top_k: Number of passages to retrieve.

        Returns:
            A text summary of the most relevant results.
        """
        return str(_web(q))[:MAX_TOOL_RESULT_CHARS]

    return [web_search, text_search]


class LayoutParsingEnv:
    """Environment exposing docling-based `layout_parsing` as a tool.

    Registered via `environment_factory`, so GOLD calls `reset(**example)` per rollout with the raw example dict (which
    carries the prompt images). `layout_parsing` resolves the model's symbolic `"img_1"` reference to the actual PIL
    image and runs docling on it.
    """

    def __init__(self):
        # Built lazily on first use and kept for the life of this (pooled, reused) instance. Rebuilding it per reset
        # would reload docling's OCR models on every rollout.
        self._converter = None

    def reset(self, **example):
        self._images = example.get("images") or []
        # Per-instance scratch dir: environment instances are pooled and run concurrently, so a shared fixed path
        # would race between rollouts saving the same image index.
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
            self._converter = DocumentConverter()
        return self._converter.convert(str(tmp)).document.export_to_markdown()[:MAX_TOOL_RESULT_CHARS]


def build_vlm_dataset(domains, max_conversations=None):
    """Search-VL (text-returning-tool subset) -> GOLD prompt/completion with images + tools.

    Kept to conversations whose tools all return text and whose images live only in the prompt. Search-VL already
    records its tool calls in Qwen3's native `<tool_call>` format, so no reformatting is needed. The dataset's own
    system prompt is replaced with `VLM_SYSTEM_PROMPT`; `tools` is stored as a JSON string to avoid Arrow null-padding.
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

            prompt = [{"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}]}]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--teacher_model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--lmbda", type=float, default=1.0)
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="trackio")
    cli_args = parser.parse_args()

    # docling's OCR backend (RapidOCR) logs an INFO line per model on every call; quiet it to keep the logs readable.
    for noisy in ("RapidOCR", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ──────────────────────────────────────────────
    # Models
    # ──────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(cli_args.student_model_name, padding_side="left")
    student_model = AutoModelForImageTextToText.from_pretrained(cli_args.student_model_name, dtype=torch.bfloat16)
    teacher_model = AutoModelForImageTextToText.from_pretrained(cli_args.teacher_model_name, dtype=torch.bfloat16)

    # Train only the language model; keep the vision encoder frozen.
    for name, param in student_model.named_parameters():
        if "language_model" not in name:
            param.requires_grad = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r"^.*language_model.*\.(q_proj|k_proj)$",
    )

    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
    dataset = build_vlm_dataset(list(SEARCH_VL_DOMAINS), cli_args.max_conversations).train_test_split(
        test_size=0.05, seed=42
    )
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # ──────────────────────────────────────────────
    # Training config
    # ──────────────────────────────────────────────
    student_short = cli_args.student_model_name.split("/")[-1]
    teacher_short = cli_args.teacher_model_name.split("/")[-1]
    run_name = cli_args.run_name or f"gold-{student_short}-from-{teacher_short}-vlm-tools"

    args = GOLDConfig(
        output_dir=run_name,
        run_name=run_name,
        # GOLD-specific
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.6,
        max_completion_length=2048 * 2,
        max_grad_norm=1.0,
        teacher_model_name_or_path=cli_args.teacher_model_name,
        num_generations=1,
        max_tool_calling_iterations=10,  # cap the on-policy tool loop so a stuck rollout can't run to the token limit
        # vLLM + fused loss
        use_vllm=True,
        use_liger_kernel=True,  # fused JSD avoids materializing full-vocab logits (Qwen3 vocab is ~150k)
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=16384,  # expanded image tokens make VLM prompts long; leave room for the completion
        max_length=16384,
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
    # `tools` are the standalone search callables; `layout_parsing` is added per rollout as a method of the
    # environment (so it can resolve image references), instantiated via `environment_factory`.
    trainer = GOLDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        tools=build_search_tools(),
        environment_factory=LayoutParsingEnv,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
