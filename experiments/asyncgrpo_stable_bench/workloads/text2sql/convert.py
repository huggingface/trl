"""miles' convert_dataset.py for TRL: the recipe's data release as jsonl, one row per task.

    {"id": <task id>, "prompt": <opening messages>, "task": <the task row as JSON, database_path absolute>}

Splits, loaded with the recipe's own loader (every file and database checked against its pinned SHA-256):
    train       BIRD-Platinum training tasks, without the 31 the recipe's hint mask leaves out (2,033), as miles
    validation  BIRD-Platinum validation tasks (398), as miles
    arcwise     Arcwise-Plat-SQL (498), graded with the ReViSQL authors' 1% single-value tolerance, as miles
    hard        BIRD-Platinum-Hard (689 training tasks), hint mask applied like train (not used by miles)
Also writes probe-<split>.jsonl: 200 tasks per split drawn with seed 0, for the base pass-rate probe.

Usage: python convert.py <data dir> <output dir>
"""

import json
import random
import sys
from pathlib import Path

from text_to_sql_recipe import data, prompts


def rows(split, data_dir):
    match split:
        case "train":
            return data.apply_hint_mask(data.load_dataset("revisql-verified", data_dir))
        case "validation":
            return data.load_dataset("revisql-verified-validation", data_dir)
        case "arcwise":
            return [{**row, "numeric_tolerance": True} for row in data.load_dataset(data.ARCWISE, data_dir)]
        case "hard":
            return data.apply_hint_mask(data.load_dataset("bird-platinum-hard", data_dir))


def main(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "arcwise", "hard"):
        records = [
            {"id": str(row["id"]), "prompt": prompts.opening_messages(row, few_shot=True), "task": json.dumps(row)}
            for row in rows(split, data_dir)
        ]
        # the TRL loop finds a rollout's task by its last prompt message, so it must be unique
        problems = [r["prompt"][-1]["content"] for r in records]
        assert len(set(problems)) == len(problems), f"{split}: {len(problems) - len(set(problems))} duplicate prompts"
        for name, subset in ((split, records), (f"probe-{split}", random.Random(0).sample(records, 200))):
            with (output_dir / f"{name}.jsonl").open("w") as f:
                for record in subset:
                    f.write(json.dumps(record) + "\n")
        print(f"{split}: {len(records)} tasks")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
