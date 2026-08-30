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
#     "trl",
#     "kernels",
# ]
# ///

"""
Zero-sync GRPO on a multi-hop search agent: Qwen3.5 learns to use a retrieval tool.

The task is HotpotQA: questions that cannot be answered from one document ("Which magazine was
started first, Arthur's Magazine or First for Women?"). The model gets a `search` tool over a
Wikipedia corpus built from the dataset's own paragraphs, so answering takes several turns: search,
read, search again, answer. Only the final answer is rewarded, so the search behavior in between is
learned rather than demonstrated.

Two things make this cheap enough to run on one node, and both come from the same place: the KV
cache is the scarce resource in zero-sync training, because generation and training share the GPU.

- Training and generation share ONE copy of the weights. There is no inference server and no weight
  sync, so the model the tool loop queries is always the one the optimizer just updated.
- Qwen3.5 is a hybrid: 3 layers in 4 are linear attention, which keeps a fixed-size recurrent state
  instead of a growing KV cache. Qwen3.5-9B stores 32 KiB of KV per token where a dense Qwen3-14B
  stores 160 KiB, so a 32k-token rollout costs 1 GiB instead of 5. That headroom is what pays for a
  a large `rollouts_in_flight`.

Each turn resubmits the conversation so far, and the continuous batching engine serves the repeated
prefix from its cache (~91% of the second turn's prompt), so a tool loop costs far less than the
same tokens generated fresh.

torchrun --nproc-per-node 4 examples/zero_sync_grpo_search/zero_sync_grpo_search.py

Qwen3.5-9B takes 4 GPUs (its 4 key-value heads set the tensor parallel width). For a single GPU,
set `MODEL` to Qwen/Qwen3.5-2B.

As a smoke test, a few steps on 4xH100 run the loop end to end: about 2 search calls per question
and 0.73 answer F1 on Qwen3.5-9B. That is the starting point, not a training curve.
"""

import math
import os
import re
import string
from collections import Counter, defaultdict

from datasets import load_dataset

from trl.experimental.zero_sync_grpo import ZeroSyncGRPOConfig, ZeroSyncGRPOTrainer


MODEL = "Qwen/Qwen3.5-9B"

# Telling the model it does not know the answer is what gets it to search: left to itself, it
# answers the question straight away from memory, and there is no tool use to reinforce.
SYSTEM_PROMPT = (
    "You answer multi-hop questions with the `search` tool. You do not know the answer yet: always "
    "call `search` first, one entity at a time, and read the article before answering. Search again "
    "for whatever the first article points to. Once the articles give you the answer, reply with the "
    "answer alone: a name, a date or a short phrase, with no explanation."
)

# The corpus the `search` tool reads. Tools are plain functions called by the trainer, so the index
# lives at module level, built once in `main`. Every process must retrieve the same articles for the
# same query: under tensor parallelism the ranks run one conversation together, so a tool result that
# differs between them makes their requests diverge. Hence sorted postings rather than sets, whose
# iteration order over strings changes from one process to the next.
DOCUMENTS: dict[str, str] = {}
INVERTED_INDEX: dict[str, list[str]] = {}
INVERSE_DOCUMENT_FREQUENCY: dict[str, float] = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(dataset) -> None:
    """Index every paragraph of the dataset, so each question's gold paragraphs sit among the others."""
    for example in dataset:
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"], strict=True):
            DOCUMENTS.setdefault(title, "".join(sentences))
    postings = defaultdict(set)
    for title, text in DOCUMENTS.items():
        for token in set(tokenize(title + " " + text)):
            postings[token].add(title)
    for token, titles in postings.items():
        INVERTED_INDEX[token] = sorted(titles)
        INVERSE_DOCUMENT_FREQUENCY[token] = math.log(len(DOCUMENTS) / len(titles))


def search(query: str) -> str:
    """Search an encyclopedia and return the most relevant articles.

    Args:
        query: What to look for, for example the name of a person, work or place.
    """
    scores = Counter()
    for token in sorted(set(tokenize(query))):
        weight = INVERSE_DOCUMENT_FREQUENCY.get(token, 0.0)
        for title in INVERTED_INDEX.get(token, ()):
            # A match in the title is what identifies an article, so it counts double
            scores[title] += weight * (2.0 if token in tokenize(title) else 1.0)
    if not scores:
        return "No article found."
    # Ranked by score, ties broken by title, so the ranking never depends on iteration order
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:3]
    return "\n\n".join(f"# {title}\n{DOCUMENTS[title]}" for title, _ in ranked)


def normalize(text: str) -> list[str]:
    """The HotpotQA answer normalization: casing, articles and punctuation carry no meaning here."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [word for word in text.split() if word not in ("a", "an", "the")]


def answer_reward(completions, answer, **kwargs):
    """Token F1 between the model's last message and the reference answer."""
    rewards = []
    for completion, reference in zip(completions, answer, strict=True):
        predicted = Counter(normalize(completion[-1]["content"] or ""))
        expected = Counter(normalize(reference))
        common = sum((predicted & expected).values())
        if common == 0:
            rewards.append(0.0)
            continue
        precision = common / sum(predicted.values())
        recall = common / sum(expected.values())
        rewards.append(2 * precision * recall / (precision + recall))
    return rewards


def main():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train[:8192]")
    build_index(dataset)

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    dataset = dataset.map(make_conversation, remove_columns=dataset.column_names)

    config = ZeroSyncGRPOConfig(
        output_dir="zero-sync-grpo-hotpotqa",
        save_strategy="no",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        # Per turn, not per rollout: a full search loop is several of these
        max_completion_length=1024,
        max_tool_calling_iterations=4,
        # The tool loop is the reasoning here, so the model answers from what it read
        chat_template_kwargs={"enable_thinking": False},
        rollouts_in_flight=64,
        # Fraction of free VRAM for the KV cache; the rest is for activations and gradients
        continuous_batching_config={"max_memory_percent": 0.25},
        packed_training=True,
        tp_size=int(os.environ.get("WORLD_SIZE", "1")),
        max_steps=200,
    )
    trainer = ZeroSyncGRPOTrainer(
        model=MODEL,
        args=config,
        train_dataset=dataset,
        reward_funcs=answer_reward,
        tools=[search],
    )
    trainer.train()


if __name__ == "__main__":
    main()
