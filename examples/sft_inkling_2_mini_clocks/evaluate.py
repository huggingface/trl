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
#     "transformers>=5.24.0",
#     "peft>=0.21.0",
# ]
# ///

# docstyle-ignore
"""
Share of held-out analog clocks a model reads right: to the second, to the minute, and the hour alone.

    python evaluate.py thinkingmachines/Inkling-2-Mini
    python evaluate.py thinkingmachines/Inkling-2-Mini --adapter Inkling-2-Mini-clocks
"""

import argparse
import re

import torch
from datasets import Image, load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--adapter", help="LoRA adapter to load on top of the model")
parser.add_argument("--reasoning_effort", default="none")
parser.add_argument("--num_clocks", type=int, default=200)
parser.add_argument("--max_new_tokens", type=int, default=256)
args = parser.parse_args()

dataset = load_dataset("migonsa/analog_clocks_combinations_for_finetuning", split="test")
# 3 of the images are empty files
dataset = dataset.cast_column("image", Image(decode=False))
dataset = dataset.filter(lambda image: image["bytes"], input_columns="image").cast_column("image", Image())
dataset = dataset.shuffle(seed=0).select(range(args.num_clocks))
processor = AutoProcessor.from_pretrained(args.model)
processor.tokenizer.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)


def read_time(text):
    """The first time written in the answer, as a list of integers: hours, minutes and possibly seconds."""
    match = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    return [int(x) for x in match.groups() if x is not None] if match else []


scores = {"to the second": 0, "to the minute": 0, "the hour": 0}
for start in range(0, len(dataset), 16):
    batch = dataset.select(range(start, min(start + 16, len(dataset))))
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}],
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort=args.reasoning_effort,
        )
        for example in batch
    ]
    images = [[example["image"]] for example in batch]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    prompts_ids = [
        ids[mask.bool()].tolist() for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"], strict=True)
    ]
    completions = outputs[:, inputs["input_ids"].shape[1] :].tolist()
    for example, prompt_ids, completion in zip(batch, prompts_ids, completions, strict=True):
        answer = processor.parse_response(completion, prefix=prompt_ids).get("content", "")
        predicted, truth = read_time(answer), read_time(example["answer"])
        scores["to the second"] += predicted == truth
        scores["to the minute"] += predicted[:2] == truth[:2]
        scores["the hour"] += predicted[:1] == truth[:1]

name = args.adapter or args.model
print(f"{name}: " + ", ".join(f"{value / len(dataset):.1%} {key}" for key, value in scores.items()))
