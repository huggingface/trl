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
#     "bitsandbytes",
#     "trackio",
# ]
# ///

"""
Teach visual tool calling to HuggingFaceTB/SmolVLM-Instruct with SFT + QLoRA on the
OpenSearch-VL/Search-VL-SFT-36K dataset (multimodal search-agent trajectories).

This is the VLM counterpart of `sft_tiny_aya_tool_calling.py`: SmolVLM has no native tool-calling
support, so we attach a tool-aware chat template (`smolvlm_tool_chat_template.jinja`) that renders
tool schemas into the system message, assistant `<tool_call>` turns, and `tool`-role results as
`<observation>` blocks. The modified template is saved with the processor.

Two adaptations are required for the VLM + multi-turn tool setting:

1. Only the GOLD-compatible slice of the dataset is used: conversations whose tools all return TEXT
   (`text_search`, `web_search`, `layout_parsing`) and whose images live only in the prompt. Trajectories
   that use image-returning tools (`crop`, `super_resolution`, `image_search` reverse lookup) are dropped.

2. Assistant-only loss is not supported for VLMs in TRL, so instead of `{% generation %}` markers we unroll
   each multi-turn conversation into one prompt/completion example per assistant turn (each tool observation
   therefore always lands in the prompt) and mask with `DataCollatorForVisionLanguageModeling(completion_only_loss=True)`.

The resulting model is meant to be pushed to the Hub and reused as the *teacher* for on-policy GOLD
distillation (see the companion GOLD tool-calling example).

Example:

    python examples/scripts/sft_smolvlm_tool_calling.py --push_to_hub --hub_model_id <user>/SmolVLM-search-tools-SFT
"""

import argparse
import json
import re
import zipfile
from pathlib import Path

import torch
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


DATASET_ID = "OpenSearch-VL/Search-VL-SFT-36K"
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
TEMPLATE_PATH = Path(__file__).parent / "smolvlm_tool_chat_template.jinja"

# Tools whose results are text (GOLD-compatible). Conversations using any other tool (image-returning
# `crop`/`super_resolution`, reverse-image `image_search`) are dropped.
ALLOWED_TOOLS = {"text_search", "web_search", "layout_parsing"}

# JSON + image archive for each domain subset.
DOMAINS = {
    "fvqa": "fvqa/fvqa_llama_factory_clean.json",
    "livevqa": "livevqa/livevqa_llama_factory_filtered.json",
    "webqa": "webqa/webqa_llama_factory_filtered.json",
    "wiki_en": "wiki_en/wiki_en_llama_factory_filtered.json",
    "wiki_zh": "wiki_zh/wiki_zh_llama_factory_filtered.json",
    "wiki_art": "wiki_art/wikiart_llama_factory_filtered.json",
    "palace": "palace/palace_llama_factory_filtered.json",
}


def render_tools_block(tools: list[dict]) -> str:
    """Serialize tool schemas into the XML block that gets appended to the system message."""
    lines = ["# Tools", "", "You have access to the following functions:", "", "<tools>"]
    for tool in tools:
        t = tool.get("function", tool)
        lines.append("<function>")
        lines.append(f"<name>{t['name']}</name>")
        if t.get("description"):
            lines.append(f"<description>{t['description'].strip()}</description>")
        if t.get("parameters"):
            lines.append(f"<parameters>{json.dumps(t['parameters'])}</parameters>")
        lines.append("</function>")
    lines += ["</tools>", "", 'To call a function, reply with: <tool_call>{"name": <name>, "arguments": <args-json>}</tool_call>']
    return "\n".join(lines)


def _keep(conv: list[dict]) -> bool:
    """Keep only conversations whose tool calls are all text-returning and whose observations carry no images."""
    names = [nm for m in conv if m["from"] == "gpt" for nm in re.findall(r'"name":\s*"([^"]+)"', m["value"])]
    if not names or not set(names) <= ALLOWED_TOOLS:
        return False
    return not any(m["from"] == "observation" and "<image>" in m["value"] for m in conv)


def _to_messages(record: dict) -> tuple[list[dict], int]:
    """Convert a ShareGPT record into a normalized message list (content = list of blocks) plus its image count."""
    tools = record["tools"]
    tools = json.loads(tools) if isinstance(tools, str) else tools
    tools = [t for t in tools if (t.get("function", t)).get("name") in ALLOWED_TOOLS]
    system_text = record["system"].strip() + "\n\n" + render_tools_block(tools)

    messages = [{"role": "system", "content": [{"type": "text", "text": system_text}]}]
    n_images = 0
    for m in record["conversations"]:
        if m["from"] == "human":
            n = m["value"].count("<image>")
            n_images += n
            blocks = [{"type": "image"}] * n
            text = m["value"].replace("<image>", "").strip()
            if text:
                blocks.append({"type": "text", "text": text})
            messages.append({"role": "user", "content": blocks})
        elif m["from"] == "gpt":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": m["value"].strip()}]})
        elif m["from"] == "observation":
            obs = re.sub(r"</?observation>", "", m["value"]).strip()
            messages.append({"role": "tool", "content": [{"type": "text", "text": obs}]})
    return messages, n_images


def _unroll(messages: list[dict], image_paths: list[str]) -> list[dict]:
    """Split a multi-turn conversation into one prompt/completion example per assistant turn.

    For each assistant turn at index i, the prompt is `messages[:i]` (which always contains every prior tool
    observation, so it is masked) and the completion is `[messages[i]]`. Every example carries the full image
    list because the image-bearing user turn is part of every prompt.
    """
    examples = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        examples.append({"prompt": messages[:i], "completion": [msg], "images": list(image_paths)})
    return examples


def build_dataset(domains: list[str], max_conversations: int | None) -> Dataset:
    rows: list[dict] = []
    for domain in domains:
        json_path = hf_hub_download(DATASET_ID, DOMAINS[domain], repo_type="dataset")
        zip_path = hf_hub_download(DATASET_ID, f"{domain}/images.zip", repo_type="dataset")
        extract_dir = Path(zip_path).parent / f"{domain}_images"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        data = json.load(open(json_path))
        kept = 0
        for record in data:
            if max_conversations is not None and kept >= max_conversations:
                break
            if not _keep(record["conversations"]):
                continue
            messages, n_images = _to_messages(record)
            image_paths = [str(extract_dir / p) for p in record["images"]]
            if len(image_paths) != n_images or not all(Path(p).exists() for p in image_paths):
                continue  # skip rows whose image references don't resolve on disk
            rows.extend(_unroll(messages, image_paths))
            kept += 1

    features = Features(
        {
            "prompt": [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}],
            "completion": [{"role": Value("string"), "content": [{"type": Value("string"), "text": Value("string")}]}],
            "images": Sequence(ImageFeature()),
        }
    )
    return Dataset.from_list(rows, features=features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="SmolVLM-search-tools-SFT")
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument("--max_conversations", type=int, default=None, help="Cap kept conversations per domain.")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    cli_args = parser.parse_args()

    # ── Processor + tool-aware chat template ──────────────────────────────────
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.chat_template = TEMPLATE_PATH.read_text()
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Model (4-bit QLoRA, LoRA on the language model only) ───────────────────
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=r"^.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = build_dataset(cli_args.domains, cli_args.max_conversations)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # ── Training ────────────────────────────────────────────────────────────────
    # Masking lives on the collator (prompt masked, completion supervised); assistant_only_loss is unsupported
    # for VLMs. skip_prepare_dataset + remove_unused_columns=False keep our prompt/completion/images rows intact.
    training_args = SFTConfig(
        output_dir=cli_args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=10,
        max_length=2048,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="trackio",
        push_to_hub=cli_args.push_to_hub,
        hub_model_id=cli_args.hub_model_id,
    )

    collator = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=training_args.max_length,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor,
        data_collator=collator,
        peft_config=peft_config,
    )
    trainer.train()

    # Save model + processor (the processor carries the tool-aware chat template).
    trainer.save_model(cli_args.output_dir)
    if cli_args.push_to_hub:
        trainer.push_to_hub(dataset_name=DATASET_ID)


if __name__ == "__main__":
    main()
