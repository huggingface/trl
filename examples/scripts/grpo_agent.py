# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python grpo_agent.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir grpo_biogrid_qwen_3g-1.7b \
    --push_to_hub True \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_enable_sleep_mode False \
    --max_completion_length 1024 \
    --report_to trackio \
    --log_completions True \
    --max_steps 200
```
"""

import os
import signal
import sqlite3
import textwrap
from contextlib import contextmanager

from datasets import load_dataset

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# ------------------------
# Reward functions
# ------------------------


def correctness_reward(completions, answer, **kwargs):
    """
    Reward Yes/No correctness.
    Robust to extra spaces, punctuation, or surrounding stars.
    """
    rewards = []
    for completion, ans in zip(completions, answer, strict=False):
        guess = completion[-1]["content"].strip().lower()
        guess_clean = guess.replace("*", "").replace("`", "").strip()
        reward = 0.0

        if guess_clean not in ["yes", "no"]:
            reward -= 0.2  # didn't produce a valid Yes/No
        elif guess_clean == ans.lower():
            reward += 0.5  # correct answer
        else:
            reward -= 0.2  # incorrect answer

        rewards.append(reward)
    return rewards


def tool_usage_reward(completions, **kwargs):
    """
    Reward proper tool usage.
    Looks for assistant tool_calls and corresponding tool responses.
    """
    rewards = []
    for completion in completions:
        tool_called = False
        tool_response_ok = False
        reward = 0.0

        for turn in completion:
            if turn.get("role") == "assistant" and turn.get("tool_calls"):
                tool_called = True
            if turn.get("role") == "tool" and turn.get("content"):
                tool_response_ok = True
                if "error" in turn["content"].lower():
                    reward -= 0.3  # penalize errors

        if tool_called and tool_response_ok:
            reward += 0.25  # reward correct tool usage
        elif not tool_called:
            reward -= 0.3  # penalize missing tool call
        elif tool_called and not tool_response_ok:
            reward -= 0.2  # called tool but no response

        rewards.append(reward)
    return rewards


def structure_reward(completions, **kwargs):
    """
    Reward proper assistant structure.
    Encourages a logical sequence: tool call + response + optional extra content.
    """
    rewards = []

    for completion in completions:
        has_call = False
        has_response = False
        has_other = False

        for turn in completion:
            role = turn.get("role")
            if role == "assistant" and turn.get("tool_calls"):
                has_call = True
            elif role == "tool":
                has_response = True
            else:
                content = turn.get("content")
                if content and content.strip() not in ["", "<think>"]:
                    has_other = True

        # Reward sequences
        if has_call and has_response:
            if has_other:
                reward = 0.25
            else:
                reward = 0.1  # still positive even without extra text
        elif has_call and not has_response:
            reward = -0.15
        else:
            reward = 0.0  # neutral if no call

        rewards.append(reward)

    return rewards


# ------------------------
# Database tool function
# ------------------------
class TimeoutError(Exception):
    """Raised when a function call times out."""

    pass


@contextmanager
def timeout(seconds):
    """Context manager that raises TimeoutError if execution exceeds time limit."""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def query_biogrid(sql_command: str) -> list[tuple]:
    """
    Execute a read-only SQL command on the BioGRID database.

    BioGRID is a curated biological database that compiles protein, genetic, and chemical interactions from multiple organisms. It provides researchers with experimentally verified interaction data to support studies in systems biology and functional genomics.

    Args:
        sql_command: The SQL command to execute.

    Returns:
        A list of tuples containing the query results.
    """
    with timeout(5):
        conn = sqlite3.connect("file:biogrid.db?mode=ro", uri=True)
        cursor = conn.cursor()
        try:
            cursor.execute(sql_command)
            results = cursor.fetchall()
        finally:
            conn.close()
    return results


# ------------------------
# Dataset formatting
# ------------------------
def format_example(example):
    question = example["question"]
    preamble = textwrap.dedent("""\
    You may use the BioGRID database to answer the question. Feel free to run exploratory SQL queries to familiarize yourself with the database structure if needed (e.g., `SELECT * FROM interactions LIMIT 1;` or `PRAGMA table_info(interactions);`).
    Provide your final answer enclosed in stars, such as `*Yes*` or `*No*`.
    Facts:
      - The NCBI Taxonomy identifier for humans is taxid:9606
    """)
    content = f"{preamble}\nQuestion: {question}"
    prompt = [{"role": "user", "content": content}]
    return {"prompt": prompt}


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # ------------------------
    # Create DB
    # ------------------------
    print("Creating biogrid.db...")
    biogrid_dataset = load_dataset("qgallouedec/biogrid", split="train")
    biogrid_dataset.to_sql("interactions", "sqlite:///biogrid.db", if_exists="replace")
    print("biogrid.db created.")

    # ------------------------
    # Load and format dataset
    # ------------------------
    dataset = load_dataset("qgallouedec/biogrid_qa", split="train")
    dataset = dataset.map(format_example, remove_columns=["question"])

    train_dataset = dataset
    eval_dataset = None  # No eval by default, can be added if needed

    training_args.chat_template_kwargs = {"enable_thinking": False}

    # ------------------------
    # Initialize trainer
    # ------------------------
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tools=[query_biogrid],
        reward_funcs=[correctness_reward, tool_usage_reward, structure_reward],
        args=training_args,
    )

    # ------------------------
    # Train
    # ------------------------
    trainer.train()

    # ------------------------
    # Save and push
    # ------------------------
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
