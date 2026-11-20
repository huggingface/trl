"""SkyRL parquet for the text-to-SQL workload + token/reward parity check against the TRL port.

Rows: prompt = miles' opening messages with the `execute_sql_query` tool block folded into the system message
(SkyRL renders prompts without `tools=`; the folded system message renders to the same ids, asserted for every row),
env_class = t2sql, task = the task row (JSON string) for the env.

Parity: for N tasks, a scripted model (tool call with the gold SQL, then the gold SQL as the answer) is played through
TRL's `run_episode` and through SkyRL's path (prompt render, `Text2SQLEnv.step`, `SkyRLGymGenerator.get_obs_ids_from_obs`);
the full token sequences and rewards must match.

Usage: python t2sql_data.py <in train.jsonl> <out dir> [n_parity]
"""

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

import skyrl_t2sql_env  # noqa: F401  (installs the trl stub before t2sql_episode)
import t2sql_episode as ep
from text_to_sql_recipe import prompts

MODEL = "Qwen/Qwen3.5-35B-A3B"


def folded_messages(tokenizer, messages):
    tools = [prompts.sql_tool_schema({})]
    with_tools = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)
    head = "<|im_start|>system\n"
    assert messages[0]["role"] == "system" and with_tools.startswith(head)
    system = with_tools[len(head) : with_tools.index("<|im_end|>")]
    return [{"role": "system", "content": system}, *messages[1:]]


def main(src, out, n_parity):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(line) for line in open(src)]
    records = []
    for row in rows:
        task = json.loads(row["task"])
        prompt = folded_messages(tokenizer, row["prompt"])
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        assert ids == ep.opening_prompt_ids(tokenizer, task), row["id"]
        records.append({"prompt": prompt, "env_class": "t2sql", "task": row["task"], "id": row["id"]})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out / "train.parquet")
    print(f"{len(records)} rows, prompt ids identical to TRL's for all")
    if n_parity:
        parity(tokenizer, records[:n_parity])


def parity(tokenizer, records):
    from omegaconf import OmegaConf
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

    cfg = SkyRLTrainConfig.from_cli_overrides(
        [f"trainer.policy.model.path={MODEL}", "generator.batched=false", "generator.max_turns=6"]
    )
    gen = SkyRLGymGenerator(cfg.generator, cfg.environment.skyrl_gym, None, tokenizer)
    eos = tokenizer.eos_token_id
    ok = 0
    for rec in records:
        task = json.loads(rec["task"])
        sql = task["gold_sql"]
        script = [
            f"Let me check.\n</think>\n\n<tool_call>\n<function=execute_sql_query>\n<parameter=query>\n{sql}\n</parameter>\n</function>\n</tool_call>",
            f"It works.\n</think>\n\nThe query restates the evidence.\n<solution>{sql}</solution>",
        ]

        async def trl_side():
            turn = iter(script)

            async def sample(prompt_ids, max_tokens):
                text = next(turn)
                return tokenizer.encode(text, add_special_tokens=False) + [eos], [0.0], text, "stop"

            return await ep.run_episode(sample, tokenizer, task, evaluation=False)

        trl = asyncio.run(trl_side())
        trl_ids = trl["turns"][-1].prompt_ids + trl["turns"][-1].output_ids

        env = skyrl_t2sql_env.Text2SQLEnv(None, {"task": rec["task"]})
        ids = tokenizer.apply_chat_template(rec["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False)
        for text in script:
            ids = ids + tokenizer.encode(text, add_special_tokens=False) + [eos]
            out = env.step(text)
            if out["done"]:
                break
            ids = ids + gen.get_obs_ids_from_obs(out["observations"], False)
        same = ids == trl_ids and abs(out["reward"] - trl["reward"]) < 1e-9
        ok += same
        print(f"{rec['id']}: tokens {'==' if ids == trl_ids else '!='} ({len(ids)} vs {len(trl_ids)}), reward skyrl {out['reward']:.3f} trl {trl['reward']:.3f}")
    print(f"PARITY {ok}/{len(records)}")
    assert ok == len(records)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
