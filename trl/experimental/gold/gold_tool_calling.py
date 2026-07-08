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
Step 2 of task-specific tool distillation with GOLD: distil a base model (student) from a *task-specific
teacher* that was tool-SFT'd in step 1 and pushed to the Hub. Combined on/off-policy training (`lmbda=0.5`):
off-policy slices consume the dataset's tool trajectories, on-policy slices let the student generate and
*actually execute* the tools. Requires transformers >= 5.2.0 (tool + environment-factory support).

Two modes, selected with `--mode`:

  # Text (Aya): teacher = the step-1 SFT model on bebechien/SimpleToolCalling. Tools are genuine web/wiki search.
  accelerate launch trl/experimental/gold/gold_tool_calling.py --mode aya \
      --teacher_model_name <user>/tiny-aya-tools-SFT

  # VLM (SmolVLM): teacher = the step-1 SFT model on OpenSearch-VL/Search-VL-SFT-36K. Tools are genuine
  # web/wiki search plus docling `layout_parsing` (an environment method, so it can resolve image refs).
  accelerate launch trl/experimental/gold/gold_tool_calling.py --mode smolvlm \
      --teacher_model_name <user>/SmolVLM-search-tools-SFT

NOTE: the VLM mode needs a GPU + vLLM. The SmolVLM/Idefics3 unexpanded-prompt vLLM fix from
https://github.com/huggingface/trl/issues/6294 is now applied inside `GOLDTrainer` itself (it feeds vLLM the
tokenizer-only, single-`<image>` prompt IDs), so no trainer subclass is needed here.
"""

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from trl.experimental.gold import GOLDConfig, GOLDTrainer


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "scripts"
AYA_TEMPLATE = EXAMPLES_DIR / "tiny_aya_chat_template.jinja"
SMOLVLM_TEMPLATE = EXAMPLES_DIR / "smolvlm_tool_chat_template.jinja"

ALLOWED_TOOLS = {"text_search", "web_search", "layout_parsing"}
SEARCH_VL_DOMAINS = {
    "fvqa": "fvqa/fvqa_llama_factory_clean.json",
    "livevqa": "livevqa/livevqa_llama_factory_filtered.json",
    "webqa": "webqa/webqa_llama_factory_filtered.json",
}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# Genuine tools (executed on on-policy slices). smolagents ships real, key-free web/wikipedia search; we wrap
# each smolagents Tool in a plain type-hinted callable because GOLD introspects the callable's signature +
# docstring to render the tool schema. Tool *results* are masked out of the GOLD loss, so genuine-but-noisy
# results are fine — they only need to keep the multi-turn rollout moving.
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


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# Datasets: both are converted to GOLD's conversational format with tool schemas baked into the system message
# (the chat templates are schema-agnostic). Off-policy slices use the multi-turn completion; on-policy slices
# use the prompt and regenerate with live tools.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
AYA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search public information on the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search internal/company documents, policies and project data.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]


def render_tools_block(tools):
    lines = ["# Tools", "", "You have access to the following functions:", "", "<tools>"]
    for tool in tools:
        t = tool.get("function", tool)
        lines += ["<function>", f"<name>{t['name']}</name>"]
        if t.get("description"):
            lines.append(f"<description>{t['description'].strip()}</description>")
        if t.get("parameters"):
            lines.append(f"<parameters>{json.dumps(t['parameters'])}</parameters>")
        lines.append("</function>")
    lines += ["</tools>", "", 'To call a function, reply with: <tool_call>{"name": <name>, "arguments": <args-json>}</tool_call>']
    return "\n".join(lines)


def build_aya_dataset():
    """bebechien/SimpleToolCalling -> GOLD prompt/completion with structured tool_calls + tools column."""
    ds = load_dataset("bebechien/SimpleToolCalling", split="train")
    tools_block = render_tools_block(AYA_TOOLS_SCHEMA)

    def to_gold(sample):
        return {
            "prompt": [
                {"role": "system", "content": tools_block},
                {"role": "user", "content": sample["user_content"]},
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": sample["tool_name"],
                                "arguments": json.loads(sample["tool_arguments"]),
                            },
                        }
                    ],
                }
            ],
            "tools": AYA_TOOLS_SCHEMA,
        }

    return ds.map(to_gold, remove_columns=ds.column_names)


def build_smolvlm_dataset(domains, max_conversations):
    """OpenSearch-VL/Search-VL-SFT-36K (clean subset) -> GOLD conversational format with images + tools baked in."""
    from huggingface_hub import hf_hub_download

    rows = []
    for domain in domains:
        json_path = hf_hub_download("OpenSearch-VL/Search-VL-SFT-36K", SEARCH_VL_DOMAINS[domain], repo_type="dataset")
        zip_path = hf_hub_download("OpenSearch-VL/Search-VL-SFT-36K", f"{domain}/images.zip", repo_type="dataset")
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
            if not names or not set(names) <= ALLOWED_TOOLS:
                continue
            if any(m["from"] == "observation" and "<image>" in m["value"] for m in conv):
                continue

            tools = rec["tools"]
            tools = json.loads(tools) if isinstance(tools, str) else tools
            tools = [t for t in tools if t.get("function", t)["name"] in ALLOWED_TOOLS]
            system_text = rec["system"].strip() + "\n\n" + render_tools_block(tools)

            prompt = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
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
            from PIL import Image as PILImage

            rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "images": [PILImage.open(p).convert("RGB") for p in image_paths],
                    "tools": tools,
                }
            )
            kept += 1
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["aya", "smolvlm"], required=True)
    parser.add_argument("--teacher_model_name", type=str, required=True, help="Step-1 tool-SFT model on the Hub.")
    parser.add_argument("--student_model_name", type=str, default=None, help="Defaults to the base model per mode.")
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="trackio", help="Experiment tracker, e.g. 'wandb' or 'trackio'.")
    cli_args = parser.parse_args()

    # LoRA on all attention + MLP projections. r=32 / alpha=64 (alpha = 2*r) mirrors the step-1 SFT teacher; LoRA also
    # regularizes the student. Both student and teacher are small (~2-3B), so we load them in bf16 with no 4-bit quant
    # (bf16 trains better than QLoRA). Their weights are cheap; the 80 GB A100 budget is spent on the long-context
    # activations and KV cache driven by the max_length / max_completion_length set per mode below.
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    if cli_args.mode == "aya":
        student_id = cli_args.student_model_name or "CohereLabs/tiny-aya-global"
        processing_class = AutoTokenizer.from_pretrained(student_id)
        processing_class.chat_template = AYA_TEMPLATE.read_text()
        student = AutoModelForCausalLM.from_pretrained(student_id, dtype=torch.bfloat16)
        teacher = AutoModelForCausalLM.from_pretrained(cli_args.teacher_model_name, dtype=torch.bfloat16)
        train_dataset = build_aya_dataset()
        tools, environment_factory = build_search_tools_aya(), None
        use_vllm = False
        # Text tool trajectories: 8192-token window, 2048-token on-policy generation budget per rollout.
        max_length, max_completion_length = 8192, 2048
    else:
        student_id = cli_args.student_model_name or "HuggingFaceTB/SmolVLM-Instruct"
        processing_class = AutoProcessor.from_pretrained(student_id)
        processing_class.chat_template = SMOLVLM_TEMPLATE.read_text()
        student = AutoModelForImageTextToText.from_pretrained(student_id, dtype=torch.bfloat16)
        teacher = AutoModelForImageTextToText.from_pretrained(cli_args.teacher_model_name, dtype=torch.bfloat16)
        train_dataset = build_smolvlm_dataset(list(SEARCH_VL_DOMAINS), cli_args.max_conversations)
        tools, environment_factory = build_search_tools(), LayoutParsingEnv
        use_vllm = True
        # Image placeholder tokens dominate the sequence, so the window is far larger: 16384 in, 4096 generated.
        max_length, max_completion_length = 16384, 4096

    args = GOLDConfig(
        output_dir=cli_args.output_dir or f"gold-{cli_args.mode}-tools",
        lmbda=cli_args.lmbda,
        beta=0.5,
        temperature=0.7,
        max_completion_length=max_completion_length,
        max_length=max_length,
        num_generations=2,
        # Effective batch 16 (4 x 4) matches the step-1 SFT. On 80 GB the pressure is the long context windows above
        # (and the VLM KV cache), not the weights — if you OOM, drop per_device_train_batch_size first (raise
        # gradient_accumulation_steps to keep the effective batch at 16).
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


def build_search_tools_aya():
    """Aya's dataset uses `search_google` / `search_knowledge_base`; back them with the same genuine search tools."""
    from smolagents import DuckDuckGoSearchTool, WikipediaSearchTool

    _web = DuckDuckGoSearchTool()
    _wiki = WikipediaSearchTool()

    def search_google(query: str) -> str:
        """
        Search public information on the web.

        Args:
            query: The search query.

        Returns:
            The top web results as text.
        """
        return str(_web(query))

    def search_knowledge_base(query: str) -> str:
        """
        Search internal/company documents, policies and project data.

        Args:
            query: The search query.

        Returns:
            The most relevant knowledge-base text.
        """
        return str(_wiki(query))

    return [search_google, search_knowledge_base]


if __name__ == "__main__":
    main()
