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

"""
AsyncGRPO checkpoint round-trip: train a few steps, push the checkpoint to the Hub, resume in a second job.

Launched by `hfjob_async_grpo_smoke.sh` (see it for the vLLM side and the two `hf jobs run` invocations). The two
stages differ only in environment:

    MAX_STEPS=4                 python3 async_grpo_checkpoint_smoke.py   # fresh: saves + pushes checkpoint-4
    MAX_STEPS=8 RESUME_FROM_HUB=1 python3 async_grpo_checkpoint_smoke.py # resume: pulls it back, trains 5..8

What the resume is expected to restore, and where each piece comes from:

* weights, optimizer, LR scheduler, RNG, `global_step` — `Trainer._load_from_checkpoint` /
  `_load_optimizer_and_scheduler`, i.e. plain `transformers` machinery, same as `SFTTrainer` and `GRPOTrainer`;
* the position in the prompt dataset — `rollout_state.json`, written next to each checkpoint by
  `AsyncGRPOTrainer._save_checkpoint`. `GRPOTrainer` gets this for free (its dataloader is a map-style dataset, so
  `Trainer` fast-forwards the sampler through `skip_first_batches`); here the prompt stream lives in the rollout
  worker's child process, which starts from scratch in the new job, so the position has to be checkpointed
  explicitly;
* the vLLM weights — `_InitialWeightSyncCallback` broadcasts the restored trainer weights on train begin, so the
  server's copy of the policy is the checkpoint's, not the base model's.

Not restored, and not restorable: rollouts that were in flight or sitting in the queue when the first job ended die
with its process. Those prompts are skipped rather than regenerated — see the `rows_consumed` note in the trainer.

The trackio run name is identical in both stages, and the HF trackio callback inits with `resume="allow"`, so the
second job appends to the first job's run instead of starting a new one.

The dataset is the sanity check from "Defeating the Training-Inference Mismatch via FP16"
(https://huggingface.co/papers/2510.26788): 1460 MATH problems filtered so DeepSeek-R1-Distill-Qwen-1.5B solves each
between 20% and 80% of the time. It is used here for its prompts and its rule-based ground truth, not for its pass
mark — a handful of steps says nothing about training accuracy. Hyperparameters follow the reference
`oat/scripts/sanity/bf16_grpo.sh` (lr 1e-6 constant, no KL penalty, 8 rollouts/prompt, temperature 1.0), except for
the smoke-test budget: 8 completions per optimizer step instead of 128, and 2048-token completions instead of 8192.
"""

import json
import logging
import os
import re
import threading

from datasets import load_dataset
from huggingface_hub import snapshot_download
from math_verify import parse, verify
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Same repo and same run name in both stages: the first job pushes `last-checkpoint/` here, the second pulls it back
# and continues the same trackio run.
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "aminediroHF/async-grpo-ckpt-smoke-r1d-1.5b")
RUN_NAME = os.environ.get("RUN_NAME", "async-grpo-ckpt-smoke")
OUTPUT_DIR = "/tmp/async-grpo-ckpt-smoke"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "4"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "4"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "2048"))

# Response template for the rollout worker. `add_response_schema` only knows a fixed allowlist of chat templates and
# R1-Distill's is not one of them, so one has to be set up front (the worker skips its own lookup when one is present).
#
# It has to be a new-style `response_template`, not the legacy `response_schema`: on transformers >= 5.13
# `tokenizer.parse_response` raises `AttributeError: This tokenizer does not have a response_template` when only the
# schema is set, so the legacy path is already dead, not merely deprecated as the worker's FutureWarning suggests.
#
# `start_anchor` is the full generation prompt suffix — R1-Distill's chat template pre-writes the opening `<think>\n`,
# so anchoring on `<｜Assistant｜>` alone would fold that tag into the parsed content. There is deliberately no
# `reasoning_content` field: `content` stays the whole completion, `</think>` included, because that is what the reward
# below splits on.
R1_DISTILL_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<｜Assistant｜><think>\n",
    "fields": {"content": {"close_pattern": r"<｜end▁of▁sentence｜>\s*", "content": "text"}},
}

# Port of the reference reward: oat's `r1_distill_qwen_math_reward_fn` -> `boxed_reward_fn`
# (https://github.com/sail-sg/oat, selected by the recipe's `--prompt_template r1_distill_qwen`).
# `trl.rewards.accuracy_reward` is not equivalent: it returns `None` when the gold fails to parse (160 of this
# dataset's 1460 golds don't parse under math_verify, and trl drops those samples), it extracts the *first*
# `\boxed{}` where oat takes the last, and it does not require `</think>`.
_SUBS = [
    (r"\left", ""),
    (r"\right", ""),
    (r"\!", ""),
    (r"\,", ""),
    (r"\;", ""),
    (r"\ ", ""),
    (r"\$", ""),
    ("$", ""),
    (r"\%", ""),
    ("%", ""),
    (r"\dfrac", r"\frac"),
    (r"\tfrac", r"\frac"),
    (r"^\circ", ""),
    (r"\circ", ""),
    (r"\text{", "{"),
    (r"\mbox{", "{"),
]


def _normalize(s: str) -> str:
    """Normalize LaTeX the way oat's `grade_answer_mathd` does before comparing."""
    s = str(s).strip()
    for old, new in _SUBS:
        s = s.replace(old, new)
    s = re.sub(r"\\frac\s+", r"\\frac", s)  # `\frac {40}7` -> `\frac{40}7`
    s = re.sub(r"\\frac\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", s)  # `\frac{40}7` -> `\frac{40}{7}`
    s = re.sub(r"\\sqrt\s+", r"\\sqrt", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separator: `1,657` -> `1657`
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s.removesuffix(".0")


def _last_boxed(text: str) -> str | None:
    """Contents of the LAST `\\boxed{...}`, matching oat's `extract_boxed_answer` (which uses rfind)."""
    start = text.rfind(r"\boxed")
    if start == -1:
        return None
    open_brace = text.find("{", start)
    if open_brace == -1:
        return None
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i]
    return None


def r1_distill_math_reward(completions, solution, completion_ids, **kwargs):
    """1.0 if the completion terminates, closes its reasoning and boxes the right answer, else 0.0."""
    # math_verify implements its timeouts with `signal.alarm`, which only works on the main thread.
    # AsyncGRPOTrainer scores rollouts via `asyncio.to_thread`, so timeouts must be disabled there or every symbolic
    # comparison raises. `trl.rewards.accuracy_reward` does the same dance.
    on_main_thread = threading.current_thread() is threading.main_thread()
    parsing_timeout = 10 if on_main_thread else None
    verify_timeout = 5 if on_main_thread else None
    if not on_main_thread:
        logging.getLogger("math_verify.parser").setLevel(logging.ERROR)
        logging.getLogger("math_verify.grader").setLevel(logging.ERROR)

    rewards = []
    for completion, gold, ids in zip(completions, solution, completion_ids, strict=True):
        # oat overrides the grade to 0 for any rollout that hit the length budget without emitting EOS
        # (`if no_eos[i][j]: reward = 0`), regardless of what it had written by then.
        if len(ids) >= MAX_COMPLETION_LENGTH:
            rewards.append(0.0)
            continue
        # oat requires exactly one `</think>`: an unterminated or doubled reasoning block is unformatted, hence 0.
        parts = completion[0]["content"].split("</think>")
        answer = _last_boxed(parts[1]) if len(parts) == 2 else None
        if answer is None:
            rewards.append(0.0)
        elif _normalize(answer) == _normalize(gold):
            rewards.append(1.0)
        else:
            # Symbolic fallback for answers that are equivalent but written differently.
            parsed_gold = parse(gold, parsing_timeout=parsing_timeout)
            parsed_answer = parse(answer, parsing_timeout=parsing_timeout)
            rewards.append(
                float(
                    bool(parsed_gold)
                    and bool(parsed_answer)
                    and bool(verify(parsed_gold, parsed_answer, timeout_seconds=verify_timeout))
                )
            )
    return rewards


def format_sample(sample):
    # `prompt` is already conversational and already ends with the "put your final answer within \boxed{}"
    # instruction, so it is passed through untouched. Only the rule-based ground truth has to be lifted out of the
    # `reward_model` struct into the `solution` column the reward function reads.
    return {"prompt": sample["prompt"], "solution": sample["reward_model"]["ground_truth"]}


def pull_last_checkpoint() -> str:
    """Download the `last-checkpoint/` folder pushed by the previous job, and return its local path.

    `hub_strategy="checkpoint"` uploads the whole checkpoint folder — weights, optimizer, scheduler, RNG,
    `trainer_state.json` and this trainer's `rollout_state.json` — under that single path in the model repo, so a
    job that starts on a bare filesystem can pick it up. Nothing in `Trainer` does this download for you.

    Downloaded outside `output_dir` on purpose: every push also mirrors `output_dir` itself to the repo root, and a
    copy of the old checkpoint sitting in there would be uploaded alongside — racing the new one.
    """
    path = snapshot_download(repo_id=HUB_MODEL_ID, allow_patterns="last-checkpoint/*", local_dir="/tmp/resume-from")
    checkpoint = os.path.join(path, "last-checkpoint")
    with open(os.path.join(checkpoint, "trainer_state.json")) as f:
        global_step = json.load(f)["global_step"]
    with open(os.path.join(checkpoint, "rollout_state.json")) as f:
        rows_consumed = json.load(f)["rows_consumed"]
    print(f"[resume] {HUB_MODEL_ID}/last-checkpoint: global_step={global_step}, rows_consumed={rows_consumed}")
    return checkpoint


def main() -> None:
    dataset = load_dataset("sail/Sanity-Test-R1D-1.5B", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.response_template = R1_DISTILL_RESPONSE_TEMPLATE

    config = AsyncGRPOConfig(
        output_dir=OUTPUT_DIR,
        # The trainer defaults to float32; bfloat16 matches the dtype the launcher serves vLLM in, so the
        # precision-mismatch warning stays silent, and it halves what every checkpoint push has to upload. Note this
        # is *pure* bf16, not fp32 master weights — fine for exercising a checkpoint round-trip, wrong for measuring
        # training quality at lr 1e-6. `bf16=True` only keeps accelerate's mixed precision consistent with the
        # parameter dtype; with bf16 parameters its autocast is a formality.
        dtype="bfloat16",
        bf16=True,
        num_generations=8,
        temperature=1.0,
        max_completion_length=MAX_COMPLETION_LENGTH,
        learning_rate=1e-6,
        # oat's `PPOArgs` defaults to adam_beta_2=0.95; HF `TrainingArguments` defaults to 0.999.
        adam_beta2=0.95,
        # Constant LR (the reference recipe's) is what makes the two stages comparable at all: with the default linear
        # decay, the resumed job's larger `max_steps` would rescale the schedule under the restored optimizer.
        lr_scheduler_type="constant",
        # `token_budget=0` selects `FixedCountBatcher`, so the step size is an exact completion count rather than a
        # token target: 1 x 8 accumulation x 1 process = 8 completions = 1 prompt group per optimizer step. Small on
        # purpose — it keeps the dataset position small enough to eyeball in the logs across the two jobs.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        token_budget=0,
        max_staleness=4,
        max_steps=MAX_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        # `hub_strategy="checkpoint"` is the load-bearing part: it mirrors the full checkpoint folder to
        # `last-checkpoint/` in the repo. `hub_always_push` stops a save from being skipped because the previous
        # ~9 GB upload is still in flight.
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="checkpoint",
        hub_always_push=True,
        logging_steps=1,
        vllm_server_base_url=os.environ.get("SERVE_URL", "http://localhost:8000"),
        report_to="trackio",
        run_name=RUN_NAME,
        project="async-grpo-ckpt-smoke",
        trackio_space_id="async-grpo-ckpt-smoke",
    )
    trainer = AsyncGRPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=r1_distill_math_reward,
    )

    checkpoint = pull_last_checkpoint() if os.environ.get("RESUME_FROM_HUB") == "1" else None
    trainer.train(resume_from_checkpoint=checkpoint)

    print(f"[done] global_step={trainer.state.global_step}, rows_consumed={trainer.rollout_worker.rows_consumed}")


if __name__ == "__main__":
    main()
