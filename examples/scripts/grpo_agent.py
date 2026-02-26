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
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python examples/scripts/grpo_agent.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --output_dir grpo_biogrid_qwen_3g-1.7b \
    --push_to_hub True \
    --use_vllm True \
    --vllm_mode colocate \
    --max_completion_length 1024 \
    --report_to trackio \
    --log_completions True \
    --max_steps 400
```
"""

import os
import re
import signal
import sqlite3
import textwrap
from contextlib import contextmanager

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def query_reward(completions, answer, **kwargs):
    """
    Reward query strategy:
    - Penalize more than 2 queries
    - Penalize generic queries (LIMIT 1 / PRAGMA)
    - Reward usage of WHERE
    - Reward evidence supporting the final answer
    """
    rewards = []

    for completion, ans in zip(completions, answer, strict=False):
        reward = 0.0
        sql_queries = []
        tool_results = []

        # collect all SQL queries and tool results
        for turn in completion:
            if turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    sql = call["function"]["arguments"].get("sql_command", "").lower()
                    sql_queries.append(sql)
            if turn.get("role") == "tool" and turn.get("content"):
                tool_results.append(turn["content"])

        # --- penalize too many queries ---
        if len(sql_queries) > 3:
            reward -= 1.5

        # --- check query quality ---
        where_count = 0
        for q in sql_queries:
            if "limit 1" in q:
                reward -= 1.0
            if " where " not in q:
                reward -= 0.5
            else:
                where_count += 1
        reward += min(where_count, 3) * 0.4  # small bonus for WHERE usage

        # --- evidence check: do queries support the answer? ---
        combined_results = []
        error_detected = False

        for res in tool_results:
            if isinstance(res, dict) and "error" in res:
                error_detected = True
            elif isinstance(res, list):
                combined_results.extend(res)

        # if error detected, penalize heavily
        if error_detected:
            reward -= 2.0
        elif len(sql_queries) == 0:
            reward -= 1.5
        else:
            has_hits = len(combined_results) > 0
            correct_answer = ans.lower()
            if (has_hits and correct_answer == "yes") or (not has_hits and correct_answer == "no"):
                reward += 2.0
            else:
                reward -= 1.5

        rewards.append(reward)

    return rewards


def correctness_reward(completions, answer, **kwargs):
    """
    Reward Yes/No correctness.
    Model must provide final answer enclosed in stars — *yes* or *no*.
    Does not reward informal yes/no buried in text.
    """
    rewards = []
    for completion, ans in zip(completions, answer, strict=False):
        raw = completion[-1]["content"].lower()

        # detect form *yes* or *no*
        match = re.search(r"\*(yes|no)\*", raw)
        guess = match.group(1) if match else None

        reward = 0.0

        if guess is None:
            reward -= 0.5  # invalid format
        elif guess == ans.lower():
            reward += 0.6  # correct under required format
        else:
            reward -= 1.0  # wrong answer

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
                reward = 0.1
            else:
                reward = 0.05  # still positive even without extra text
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
    You have access to the BioGRID SQLite database.
    Use SQL queries to retrieve only the information needed to answer the question.

    Genes may appear in the database in columns `Alt_IDs_Interactor_A` `Alt_IDs_Interactor_B`, `Aliases_Interactor_A` and `Aliases_Interactor_B`,
    and each entry can contain multiple gene names or synonyms separated by '|', for example:
    'entrez gene/locuslink:JNKK(gene name synonym)|entrez gene/locuslink:MAPKK4(gene name synonym)|...'
    So a gene like 'JNKK' or 'MAPKK4' may appear inside one of these strings.

    If the database schema is unclear or you are unsure about column names:
    - First inspect the schema with `PRAGMA table_info(interactions);`
    - Or preview a few rows with `SELECT * FROM interactions LIMIT 1;`

    Otherwise, directly query the required data.

    Final answer must be enclosed in stars, e.g. *Yes* or *No*.
    Facts:
    - The NCBI Taxonomy identifier for humans is taxid:9606.
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
    # Load dataset
    biogrid_dataset = load_dataset("qgallouedec/biogrid", split="train")
    df = biogrid_dataset.to_pandas()

    # Normalize column names: remove spaces, replace with underscores
    df.columns = [c.replace(" ", "_") for c in df.columns]
    conn = sqlite3.connect("biogrid.db")
    try:
        df.to_sql("interactions", conn, if_exists="replace", index=False)
        print(f"biogrid.db created. Rows stored: {len(df)}")
    finally:
        conn.close()

    # ------------------------
    # Load and format dataset
    # ------------------------
    dataset = load_dataset("qgallouedec/biogrid_qa", split="train")
    dataset = dataset.filter(
        lambda example: example["question"].startswith("Does the gene ")
    )  # keep only simple questions for example
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
        reward_funcs=[correctness_reward, structure_reward, query_reward],
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
