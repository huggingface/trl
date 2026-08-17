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
`STEPS_PER_JOB` optimizer steps into a Storage Bucket mounted read-write at `CKPT_DIR`, and the next job finds the
checkpoint already sitting there — so a 2400-step run that no single job could finish becomes a chain of jobs on one
trackio run:

    ./hfjob_async_grpo_smoke.sh fresh     # steps 1..STEPS_PER_JOB
    ./hfjob_async_grpo_smoke.sh resume    # the next STEPS_PER_JOB   (repeat until TOTAL_STEPS)

The bucket is what makes this cheap. Pushing each checkpoint to a model repo and pulling it back in the next job moved
21 GB (7 GB fp32 weights + 14 GB optimizer) in each direction per save; a mounted bucket is just a path, so
`get_last_checkpoint` works exactly as it does on a local disk and there is nothing to upload or download. The final
model still goes to the Hub, once, at the end of the chain.

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
from math_verify import parse, verify
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Same run name in every job of the chain, so trackio appends to the run instead of starting a new one (the HF callback
# inits with `resume="allow"`). `HUB_MODEL_ID` only receives the finished model, at the end of the chain.
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "aminediroHF/async-grpo-sanity-r1d-1.5b")
RUN_NAME = os.environ.get("RUN_NAME", "async-grpo-hfjobs")
PROJECT = os.environ.get("PROJECT", "async-grpo-sanity-r1d-1.5b")
# A Storage Bucket, mounted read-write at the same path in every job of the chain (`-v hf://buckets/...:/ckpt`).
OUTPUT_DIR = os.environ.get("CKPT_DIR", "/ckpt/async-grpo-sanity")
# Unset keeps every checkpoint of the chain, which is the point of a bucket: 10 saves of a 1.5B fp32 run is ~210 GB of
# object storage, next to nothing against the GPU hours, and it makes any step of the run inspectable after the fact.
SAVE_TOTAL_LIMIT = int(os.environ["SAVE_TOTAL_LIMIT"]) if os.environ.get("SAVE_TOTAL_LIMIT") else None

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


def find_last_checkpoint() -> tuple[str | None, int]:
    """Locate the checkpoint a previous job of the chain left behind, and read how far it got.

    Checkpoints are *written* to the mounted bucket, but they are *read* from `RESUME_DIR`, a local-disk copy the
    launcher stages before starting the trainer. Loading straight off the mount killed a rank with SIGBUS: safetensors
    mmaps `model.safetensors`, and mmap over the bucket's FUSE layer does not survive two ranks reading 7 GB
    concurrently. A sequential copy does, which is also how the write path works.
    """
    resume_dir = os.environ.get("RESUME_DIR") or OUTPUT_DIR
    checkpoint = get_last_checkpoint(resume_dir) if os.path.isdir(resume_dir) else None
    if checkpoint is None:
        return None, 0
    with open(os.path.join(checkpoint, "trainer_state.json")) as f:
        global_step = json.load(f)["global_step"]
    with open(os.path.join(checkpoint, "rollout_state.json")) as f:
        rollout_state = json.load(f)
    print(f"[found] {checkpoint}: global_step={global_step}, {rollout_state}")
    return checkpoint, global_step


def main() -> None:
    # trl logs through `accelerate.logging`, i.e. the stdlib root logger, which has no handler in a bare script — so
    # INFO records fall to `logging.lastResort`, a stderr handler pinned at WARNING, and vanish. That silently hides the
    # lines a chained run is checked against: "Resuming the prompt stream at dataset row N", the weight-sync timings and
    # the stale-sample drops. Scoped to `trl` rather than `basicConfig`, which would also unleash urllib3 and aiohttp.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    trl_logger = logging.getLogger("trl")
    trl_logger.addHandler(handler)
    trl_logger.setLevel(logging.INFO)

    checkpoint, done_steps = find_last_checkpoint()
    if os.environ.get("RESUME") == "1":
        if checkpoint is None:
            raise RuntimeError(f"nothing to resume from in {OUTPUT_DIR}; start the chain with the `fresh` stage")
    elif checkpoint is not None:
        # A `fresh` job into a bucket that already holds a chain would either clobber it or silently continue it.
        # Neither is worth a GPU-day, so refuse and make the caller choose.
        raise RuntimeError(f"{OUTPUT_DIR} already holds {checkpoint}; use `resume`, or point CKPT_DIR somewhere new")
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
        save_total_limit=SAVE_TOTAL_LIMIT,
        # No `push_to_hub`: checkpoints land in the mounted bucket, and pushing them would move 21 GB per save for
        # nothing. `hub_model_id` is only used by the one-off push at the end of the chain.
        hub_model_id=HUB_MODEL_ID,
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
    # Only the job that finishes the chain publishes a model; the intermediate ones leave everything in the bucket.
    if trainer.state.global_step >= TOTAL_STEPS:
        print(f"[publish] chain complete at step {trainer.state.global_step}; pushing the model to {HUB_MODEL_ID}")
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
