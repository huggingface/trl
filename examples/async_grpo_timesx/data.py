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

"""Load TimesX's train/test splits from the Hub.

TimesX (https://huggingface.co/papers/2607.06973) is a context-enriched, multimodal time-series forecasting
benchmark: each variable is a long numeric series with a rolling window of "prediction origins", and every origin
comes with textual context (a domain description, calendar effects, related-series statistics, and time-stamped
news events) that is not available to a plain numeric forecaster.

Data is loaded from https://huggingface.co/datasets/kashif/timesx, a flattened, typed mirror of
https://github.com/haoxin1998/TimesX-project's per-variable JSON files -- see that dataset's card for field
descriptions, how its `train`/`test` splits are built, and why they don't reproduce the original paper's official
split (the paper's 2018-2022 training period isn't present in the upstream repository this mirror is built from).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import load_dataset


DATASET_REPOSITORY = "kashif/timesx"
DEFAULT_REVISION = "main"
# CommodityPrice/EnergyAndFuels alone has 10 variables: enough to see reward move without training on the full
# 190-variable benchmark. Pass a wider prefix (e.g. "CommodityPrice") to train on more.
DEFAULT_DOMAIN = "CommodityPrice/EnergyAndFuels"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastExample:
    """One rolling-window prediction origin for one TimesX variable."""

    variable: str
    idx: int
    background: str
    scenario: str
    holiday_info: str
    covariates_info: str
    past_timestamps: tuple[str, ...]
    past_values: tuple[float, ...]
    future_timestamps: tuple[str, ...]
    future_values: tuple[float, ...]


def _row_to_example(row: dict) -> ForecastExample:
    return ForecastExample(
        variable=row["variable"],
        idx=row["idx"],
        background=row["background"],
        scenario=row["scenario"],
        holiday_info=row["holiday_info"],
        covariates_info=row["covariates_info"],
        past_timestamps=tuple(row["past_timestamps"]),
        past_values=tuple(row["past_values"]),
        future_timestamps=tuple(row["future_timestamps"]),
        future_values=tuple(row["future_values"]),
    )


@dataclass(frozen=True)
class ForecastSplit:
    """Training and test windows, as split on the Hub."""

    train: tuple[ForecastExample, ...]
    test: tuple[ForecastExample, ...]


def load_timesx_split(domain: str = DEFAULT_DOMAIN, *, revision: str = DEFAULT_REVISION) -> ForecastSplit:
    """Load every variable under `domain` from both of the dataset's `train`/`test` splits."""
    domain_dir, _, subdomain_prefix = domain.partition("/")

    def _matches_domain(row: dict) -> bool:
        return row["domain"] == domain_dir and (not subdomain_prefix or row["subdomain"] == subdomain_prefix)

    split = {}
    for split_name in ("train", "test"):
        rows = load_dataset(DATASET_REPOSITORY, split=split_name, revision=revision)
        rows = rows.filter(_matches_domain)
        if len(rows) == 0:
            raise ValueError(f"no TimesX rows found under domain {domain!r} in the {split_name!r} split")
        examples = sorted((_row_to_example(row) for row in rows), key=lambda example: (example.variable, example.idx))
        split[split_name] = tuple(examples)
    logger.info(
        "TimesX %s: %d train windows, %d test windows across %d variables",
        domain,
        len(split["train"]),
        len(split["test"]),
        len({example.variable for example in split["train"]}),
    )
    return ForecastSplit(split["train"], split["test"])


def render_prompt(example: ForecastExample) -> str:
    """Render one rolling window's textual context and history into a forecast prompt."""
    history = "\n".join(
        f"{timestamp}: {value}" for timestamp, value in zip(example.past_timestamps, example.past_values, strict=True)
    )
    targets = "\n".join(example.future_timestamps)
    return f"""{example.background}

{example.scenario}

Upcoming calendar effects: {example.holiday_info}

Related indicators: {example.covariates_info}

Historical values, oldest first:
{history}

Forecast the value at each of the following timestamps, in order:
{targets}

On the last line, output only the {len(example.future_timestamps)} forecast values, one per timestamp above, in the
same order, separated by spaces. Do not include units, timestamps, or any other text on that line."""
