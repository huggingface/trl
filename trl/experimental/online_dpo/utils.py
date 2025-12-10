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

import math
import random
from collections.abc import Sequence
from typing import Any

from transformers.utils import is_rich_available


if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


def _to_list(data: Any) -> list[Any] | None:
    """Convert sequences like deques or tuples to lists without splitting strings."""
    if data is None:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        return [data]
    if isinstance(data, Sequence):
        return list(data)
    return [data]


def _format_scalar(value: Any) -> str:
    """Format a scalar value for display."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.2f}"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def print_pairwise_prompt_completions_sample(
    prompts: list | Sequence,
    completions: list | Sequence | None,
    rewards: dict[str, Sequence] | None,
    advantages: Sequence | None,
    step: int,
    num_samples: int = None,
    rejected_completions: list | Sequence | None = None,
    rejected_rewards: dict[str, Sequence] | None = None,
) -> None:
    """
    Print a sample of model completions to the console.

    Supports both single-completion (GRPO-style) logs and pairwise (DPO-style) logs with chosen/rejected completions.
    Reward columns are shown if provided; when both chosen and rejected rewards are given, values are rendered as
    `<chosen> / <rejected>`. Advantages are optional and shown only when provided.
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )

    prompts = _to_list(prompts) or []
    completions = _to_list(completions) or []
    rejected_completions = _to_list(rejected_completions)
    rewards = {k: _to_list(v) or [] for k, v in (rewards or {}).items()}
    rejected_rewards = {k: _to_list(v) or [] for k, v in (rejected_rewards or {}).items()}
    advantages = _to_list(advantages) if advantages is not None else None

    if rejected_completions is not None:
        row_count = min(len(prompts), len(completions), len(rejected_completions))
    else:
        row_count = min(len(prompts), len(completions))

    if row_count == 0:
        return

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= row_count:
            num_samples = None
        elif num_samples <= 0:
            return

    if num_samples is not None:
        indices = random.sample(range(row_count), num_samples)
    else:
        indices = list(range(row_count))

    reward_keys = list(rewards.keys())
    for key in rejected_rewards.keys():
        if key not in reward_keys:
            reward_keys.append(key)

    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    table.add_column("Prompt", style="bright_yellow")
    if rejected_completions is None:
        table.add_column("Completion", style="bright_green")
    else:
        table.add_column("Chosen", style="bright_green")
        table.add_column("Rejected", style="bright_red")

    for reward_name in reward_keys:
        column_title = f"{reward_name} (C/R)" if rejected_completions is not None else reward_name
        table.add_column(column_title, style="bold cyan", justify="right")

    if advantages is not None:
        table.add_column("Advantage", style="bold magenta", justify="right")

    def format_entry(entry) -> Text:
        t = Text()
        if isinstance(entry, list) and all(isinstance(m, dict) for m in entry):
            for j, msg in enumerate(entry):
                role = msg.get("role", "")
                if "content" in msg:
                    t.append(f"{role.upper()}\n", style="bold red")
                    t.append(msg["content"])
                elif "name" in msg and "args" in msg:
                    t.append(f"{role.upper()}\n", style="bold red")
                    t.append(f"{msg['name']}({msg['args']})")
                else:
                    t.append(str(msg))
                if j < len(entry) - 1:
                    t.append("\n\n")
        else:
            t.append(str(entry))
        return t

    for idx in indices:
        reward_cells = []
        for reward_name in reward_keys:
            chosen_val = rewards.get(reward_name, [])
            chosen_val = chosen_val[idx] if idx < len(chosen_val) else None
            if rejected_completions is None:
                reward_cells.append(_format_scalar(chosen_val))
            else:
                rejected_val = rejected_rewards.get(reward_name, [])
                rejected_val = rejected_val[idx] if idx < len(rejected_val) else None
                reward_cells.append(f"{_format_scalar(chosen_val)} / {_format_scalar(rejected_val)}")

        row_values = [
            format_entry(prompts[idx]),
            format_entry(completions[idx]),
        ]
        if rejected_completions is not None:
            row_values.append(format_entry(rejected_completions[idx]))

        row_values.extend(reward_cells)

        if advantages is not None and idx < len(advantages):
            row_values.append(_format_scalar(advantages[idx]))

        table.add_row(*row_values)
        table.add_section()

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)
