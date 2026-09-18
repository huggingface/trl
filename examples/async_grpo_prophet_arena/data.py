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

"""Download the Prophet Arena subset and build a leakage-resistant temporal split.

Prophet Arena (https://www.prophetarena.co) scores forecasters on real prediction-market questions. Each event
bundles one or more binary markets (e.g. "PSG vs Real Madrid" bundles the markets "PSG", "Real Madrid", "Tie"), and
every market is graded as its own binary question against its own resolution. The pinned subset is published at
https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200, one CSV row per recorded snapshot.
"""

from __future__ import annotations

import ast
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from datasets import load_dataset


DATASET_REPOSITORY = "prophetarena/Prophet-Arena-Subset-1200"
DEFAULT_REVISION = "main"
# Events that closed before this date form the training set; events first observed on or after it form validation.
# Every market of an event is kept on the same side, and events already open at the boundary are excluded, so no
# question's future is visible in another question's training example.
DEFAULT_SPLIT_DATE = "2025-10-20"
DEFAULT_MAX_TRAIN_QUESTIONS = 1024
DEFAULT_MAX_VALIDATION_QUESTIONS = 256

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastExample:
    """One binary market, at the earliest snapshot recorded for it."""

    event_ticker: str
    event_title: str
    market: str
    reference_material: str
    resolution_criteria: str
    snapshot_time: datetime
    close_time: datetime
    outcome: int


def _parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _format_sources(raw_sources: list[dict]) -> str:
    sources = sorted(
        ((source.get("ranking", i + 1), source["title"].strip(), source["summary"].strip()))
        for i, source in enumerate(raw_sources)
    )
    return "\n\n".join(f"{i}. {title}\n{summary}" for i, (_, title, summary) in enumerate(sources, start=1))


def _parse_structured(value: str) -> object:
    # `sources` and `market_outcome` are valid JSON, but `markets` is a Python list repr (single-quoted) --
    # tolerate both rather than special-casing one column.
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def load_prophet_arena_examples(revision: str = DEFAULT_REVISION) -> list[ForecastExample]:
    """Load the earliest snapshot of every event/market pair, oldest first."""
    by_market: dict[tuple[str, str], list[ForecastExample]] = defaultdict(list)
    for row in load_dataset(DATASET_REPOSITORY, split="train", revision=revision):
        reference_material = _format_sources(_parse_structured(row["sources"]))
        snapshot_time = _parse_utc_datetime(row["snapshot_time"])
        close_time = _parse_utc_datetime(row["close_time"])
        outcomes = _parse_structured(row["market_outcome"])
        for market in _parse_structured(row["markets"]):
            by_market[(row["event_ticker"], market)].append(
                ForecastExample(
                    event_ticker=row["event_ticker"],
                    event_title=row["title"],
                    market=market,
                    reference_material=reference_material,
                    resolution_criteria=f"The market resolves YES if this condition occurs: {market}.",
                    snapshot_time=snapshot_time,
                    close_time=close_time,
                    outcome=int(outcomes[market]),
                )
            )
    # Keep only the earliest (longest-horizon, least-informed) snapshot of each market: the question never changes
    # between snapshots, only the attached news and market price, which this example never shows the model anyway.
    examples = [min(snapshots, key=lambda example: example.snapshot_time) for snapshots in by_market.values()]
    examples.sort(key=lambda example: example.snapshot_time)
    return examples


@dataclass(frozen=True)
class ForecastSplit:
    """Training and validation questions separated by a chronological event-level boundary."""

    train: tuple[ForecastExample, ...]
    validation: tuple[ForecastExample, ...]


def load_prophet_arena_split(
    *,
    revision: str = DEFAULT_REVISION,
    split_date: str = DEFAULT_SPLIT_DATE,
    max_train_questions: int | None = DEFAULT_MAX_TRAIN_QUESTIONS,
    max_validation_questions: int | None = DEFAULT_MAX_VALIDATION_QUESTIONS,
    seed: int = 0,
) -> ForecastSplit:
    """Split events around `split_date` without sharing an event, or its future outcome, across the boundary."""
    split_time = _parse_utc_datetime(split_date)
    examples = load_prophet_arena_examples(revision)

    event_starts: dict[str, datetime] = {}
    event_closes: dict[str, datetime] = {}
    for example in examples:
        event_starts[example.event_ticker] = min(
            event_starts.get(example.event_ticker, example.snapshot_time), example.snapshot_time
        )
        event_closes[example.event_ticker] = max(
            event_closes.get(example.event_ticker, example.close_time), example.close_time
        )

    # An event whose markets all closed before the boundary is training data; one first observed at or after it is
    # validation. An event straddling the boundary (already open, not yet closed) is excluded from both.
    train_tickers = {ticker for ticker, closes in event_closes.items() if closes < split_time}
    validation_tickers = {ticker for ticker, starts in event_starts.items() if starts >= split_time}
    train = [example for example in examples if example.event_ticker in train_tickers]
    validation = [example for example in examples if example.event_ticker in validation_tickers]

    rng = random.Random(seed)
    if max_train_questions is not None and len(train) > max_train_questions:
        train = rng.sample(train, max_train_questions)
    if max_validation_questions is not None and len(validation) > max_validation_questions:
        validation = rng.sample(validation, max_validation_questions)
    logger.info(
        "Prophet Arena split: %d train, %d validation, %d events excluded at the boundary",
        len(train),
        len(validation),
        len(event_starts) - len(train_tickers) - len(validation_tickers),
    )
    return ForecastSplit(tuple(train), tuple(validation))


def render_prompt(example: ForecastExample) -> str:
    """Render the forecasting fields attached to the recorded snapshot. Never the resolved outcome or market price."""
    return f"""Forecast whether this market will resolve YES using information available through {example.snapshot_time.isoformat()}.

Event:
{example.event_title}

Market:
{example.market}

Reference material:
{example.reference_material}

Resolution criteria:
{example.resolution_criteria}

Market close time: {example.close_time.isoformat()}

Output only the probability of YES as a number between 0 and 1."""


def parse_probability(text: str) -> float | None:
    """Parse a probability from the completion's last non-empty line.

    Stdlib-only (no `trl`/`arena` import) so both the training script and the live serving agent can share it.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1].strip("*_. ")
    if ":" in last:
        last = last.rsplit(":", 1)[-1].strip()
    is_percent = last.endswith("%")
    if is_percent:
        last = last[:-1].strip()
    try:
        value = float(last)
    except ValueError:
        return None
    if is_percent:
        value /= 100.0
    return value if 0.0 <= value <= 1.0 else None
