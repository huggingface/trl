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
AsyncGRPO sanity run on `sail/Sanity-Test-R1D-1.5B`, checkpointed to the Hub so it can span several jobs.

Launched by `hfjob_async_grpo_smoke.sh` (see it for the GPU split and the vLLM side). Each job trains
`STEPS_PER_JOB` optimizer steps, pushes its checkpoint to `<HUB_MODEL_ID>/last-checkpoint`, and the next job picks it
up — so a 2400-step run that no single job could finish becomes a chain of jobs on one trackio run:

    ./hfjob_async_grpo_smoke.sh fresh     # steps 1..300
    ./hfjob_async_grpo_smoke.sh resume    # steps 301..600   (repeat until TOTAL_STEPS)

The recipe is the one that worked on Slurm (run 7280, final train reward 0.841 against the synchronous reference's
0.864): oat's `scripts/sanity/bf16_grpo.sh` batch shape and optimizer, with `AsyncGRPOTrainer`'s own loss. Four things
in here are load-bearing, each of them a run that failed before it was fixed:

* `top_p=1.0`. R1-Distill's `generation_config.json` ships `top_p: 0.95`, and vLLM adopts that as a server-side
  sampling default. Every async run that sampled at 0.95 collapsed (0.539 final reward, entropy 4.5-6.2); the same
  recipe at 1.0 reached 0.861. `AsyncGRPOConfig` sends `top_p` explicitly on every request, so the default of 1.0 is
  already correct here — the launcher also passes `--generation-config vllm` so the server's own defaults are out of
  the picture either way.
* `dtype="float32"` with bf16 *mixed precision* (fp32 master weights, bf16 autocast), not bf16 parameters. With bf16
  parameters a 1e-6 Adam step rounds to zero and the reward simply does not move (+0.031 over 2400 steps).
  Consequence: the trainer holds fp32 weights while vLLM serves bf16, so the trainer logs a precision-mismatch
  warning at train begin. That warning is expected here and does not describe a problem — it is what every one of the
  good runs did.
* 128 completions per optimizer step (`per_device_train_batch_size x gradient_accumulation_steps x num_processes`),
  with `token_budget=0` so the step size is an exact completion count and not a token target.
* `adam_beta2=0.95` (oat's default; HF's is 0.999) and a constant 1e-6 LR. The constant schedule is also what makes
  the job chain safe: there is no decay horizon for a per-job `max_steps` to distort.

Not the reference recipe: `AsyncGRPOTrainer` hard-codes std-normalized advantages, a token-mean loss normalizer and a
vLLM-logprob PPO denominator, so this is vanilla async GRPO at the reference's batch size and lag, not Dr. GRPO.
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
# Same repo and same run name in every job of the chain: each pushes `last-checkpoint`, the next pulls it, and trackio
# appends to the run instead of starting a new one (the HF callback inits with `resume="allow"`).
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "aminediroHF/async-grpo-sanity-r1d-1.5b")
RUN_NAME = os.environ.get("RUN_NAME", "async-grpo-hfjobs")
PROJECT = os.environ.get("PROJECT", "async-grpo-sanity-r1d-1.5b")
OUTPUT_DIR = "/tmp/async-grpo-sanity"

# How far this job trains, and how far the chain goes in total. `max_steps` is derived from the checkpoint's
# `global_step`, so `resume` is the same command every time.
STEPS_PER_JOB = int(os.environ.get("STEPS_PER_JOB", "300"))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "2400"))
# Twice per job by default. The upload runs in a background thread, so an extra save costs a local write and not much
# else — worth it, because a job that dies between saves loses everything since the last one.
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", str(max(STEPS_PER_JOB // 2, 1))))

# The reference's `--generate_max_length 8192`. Do not lower it for a quick run: at 2048 every completion is truncated,
# oat's no-EOS rule scores all of them 0, and a group of equal rewards has zero advantage — the run trains on nothing.
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "8192"))

# Batch shape. The product `PDTB x ACCUM x num_processes` must stay 128 (the recipe's `--train_batch_size`); the
# launcher sets ACCUM from the number of trainer GPUs it gives this job. PDTB=2 (over 7127's 1) was measured at 1.6x
# the throughput for the same final reward, because per-iteration overhead scales with the microbatch count.
PDTB = int(os.environ.get("PDTB", "2"))
ACCUM = int(os.environ.get("ACCUM", "16"))
# The reference generates 512 completions per rollout batch and consumes them in 4 minibatches, so its samples are up
# to ~3 optimizer steps stale; 5 with `weight_sync_steps=4` reproduces that lag.
MAX_STALENESS = int(os.environ.get("MAX_STALENESS", "5"))
WEIGHT_SYNC_STEPS = int(os.environ.get("WEIGHT_SYNC_STEPS", "4"))

# `bf16` and `fp16` here mean mixed precision on top of fp32 master weights, which is what the paper's two arms are.
# They finished 0.841 and 0.867, i.e. indistinguishable once `top_p` is correct. The launcher derives vLLM's `--dtype`
# from the same variable so the two sides cannot drift apart.
PRECISION = os.environ.get("PRECISION", "bf16")
if PRECISION not in ("bf16", "fp16"):
    raise ValueError(f"PRECISION must be 'bf16' or 'fp16', got {PRECISION!r}")
# The dtype the *parameters* are held in. Leave at float32; `bfloat16` reproduces the run that went nowhere.
DTYPE = os.environ.get("DTYPE", "float32")

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


def pull_last_checkpoint() -> tuple[str, int]:
    """Download the `last-checkpoint/` folder pushed by the previous job; return its path and its `global_step`.

    `hub_strategy="checkpoint"` uploads the whole checkpoint folder — weights, optimizer, scheduler, RNG,
    `trainer_state.json` and this trainer's `rollout_state.json` — under that single path in the model repo, so a job
    that starts on a bare filesystem can pick it up. Nothing in `Trainer` does this download for you.

    Downloaded outside `output_dir` on purpose: every push also mirrors `output_dir` to the repo root, and a copy of
    the old checkpoint sitting in there would be uploaded alongside — racing the new one.
    """
    path = snapshot_download(repo_id=HUB_MODEL_ID, allow_patterns="last-checkpoint/*", local_dir="/tmp/resume-from")
    checkpoint = os.path.join(path, "last-checkpoint")
    with open(os.path.join(checkpoint, "trainer_state.json")) as f:
        global_step = json.load(f)["global_step"]
    with open(os.path.join(checkpoint, "rollout_state.json")) as f:
        rollout_state = json.load(f)
    print(f"[resume] {HUB_MODEL_ID}/last-checkpoint: global_step={global_step}, {rollout_state}")
    return checkpoint, global_step


def main() -> None:
    if os.environ.get("RESUME_FROM_HUB") == "1":
        checkpoint, done_steps = pull_last_checkpoint()
    else:
        checkpoint, done_steps = None, 0

    max_steps = min(done_steps + STEPS_PER_JOB, TOTAL_STEPS)
    if max_steps <= done_steps:
        print(f"[skip] the chain is already at step {done_steps} of {TOTAL_STEPS}; nothing to do")
        return
    print(f"[plan] training steps {done_steps + 1}..{max_steps} of {TOTAL_STEPS}, saving every {SAVE_STEPS}")

    dataset = load_dataset("sail/Sanity-Test-R1D-1.5B", split="train")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.response_template = R1_DISTILL_RESPONSE_TEMPLATE

    config = AsyncGRPOConfig(
        output_dir=OUTPUT_DIR,
        dtype=DTYPE,
        bf16=PRECISION == "bf16",
        fp16=PRECISION == "fp16",
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # `learning_rate=1e-6`, `lr_scheduler_type="constant"`, `gradient_checkpointing=True` and `logging_steps=1` are
        # already `AsyncGRPOConfig`'s defaults, and all four are what the recipe asks for.
        adam_beta2=0.95,
        per_device_train_batch_size=PDTB,
        gradient_accumulation_steps=ACCUM,
        token_budget=0,
        max_staleness=MAX_STALENESS,
        weight_sync_steps=WEIGHT_SYNC_STEPS,
        # `max_inflight_tasks` is left to auto: `max_staleness x 128` = 640, matching the reference's rollout batch of
        # 64 prompts x 8 samples times its staleness.
        max_steps=max_steps,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        # `hub_strategy="checkpoint"` is what mirrors the full checkpoint folder to `last-checkpoint/` in the repo, and
        # `hub_always_push` stops a save from being skipped because the previous ~21 GB upload is still in flight.
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_strategy="checkpoint",
        hub_always_push=True,
        vllm_server_base_url=os.environ.get("SERVE_URL", "http://localhost:8000"),
        report_to="trackio",
        run_name=RUN_NAME,
        project=PROJECT,
        trackio_space_id=PROJECT,
    )
    trainer = AsyncGRPOTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=r1_distill_math_reward,
    )
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.rollout_worker is not None:
        print(f"[done] global_step={trainer.state.global_step}, rows_consumed={trainer.rollout_worker.rows_consumed}")


if __name__ == "__main__":
    main()
