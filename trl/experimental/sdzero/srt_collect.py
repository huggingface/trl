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
#     "math-verify>=0.5.2",
# ]
# ///

"""Example pipeline to build a self-revision dataset for [`SRTTrainer`].

For each `(problem, gold_answer)` pair in the seed dataset:

    1. Sample one initial response `y_init` from the model.
    2. Verify `y_init` (correct vs. incorrect).
    3. Select a control prompt based on the outcome:
       - correct   → "Let me rephrase the above solution."
       - incorrect → "Wait, this response is not correct, let me start over."
    4. Sample `num_revisions` revised responses conditioned on `[problem, y_init, control_prompt]`.
    5. Keep only revisions that verify as correct.

The resulting dataset has one row per kept revision with columns:

- `problem`        (str): the original problem statement.
- `y_init`         (str): the model's initial response.
- `r_init`         (int): 1 if `y_init` was correct, 0 otherwise.
- `control_prompt` (str): the rephrase/restart nudge used to elicit the revision.
- `y_revised`      (str): a verified-correct revised response.

Saved via `datasets.save_to_disk` for direct consumption by [`SRTTrainer`].

Generation backend is selectable at the CLI:

- default (transformers): loads the model with `AutoModelForCausalLM` and calls `.generate()` in batches.
- `--use_vllm`: loads the model via `vllm.LLM` (requires the `vllm` optional dep).

Example:

    uv run python trl/experimental/sdzero/srt_collect.py \\
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \\
        --num_problems 256 --num_revisions 3 \\
        --output_dir srt_revision_data
"""

import argparse
import os

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.import_utils import is_vllm_available
from trl.rewards import accuracy_reward


REPHRASE_PROMPT = "Let me rephrase the above solution."
RESTART_PROMPT = "Wait, this response is not correct, let me start over."


def build_control_prompt(is_correct: bool) -> str:
    return REPHRASE_PROMPT if is_correct else RESTART_PROMPT


def render_initial_prompt(tokenizer, problem: str) -> str:
    """Render the initial-response prompt. Matches what `SRTTrainer` tokenizes at train time."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_revision_prompt(tokenizer, problem: str, y_init: str, control_prompt: str, separator: str) -> str:
    return render_initial_prompt(tokenizer, problem) + y_init + separator + control_prompt + separator


def verify_batch(completions: list[str], references: list[str]) -> list[bool]:
    """Binary verifier: wraps `trl.rewards.accuracy_reward` (math_verify + LaTeX boxed parsing)."""
    chat_completions = [[{"role": "assistant", "content": c}] for c in completions]
    rewards = accuracy_reward(chat_completions, solution=references)
    return [r == 1.0 for r in rewards]


def load_seed(dataset_name: str, dataset_split: str, num_problems: int) -> list[dict]:
    split = f"{dataset_split}[:{num_problems}]" if num_problems > 0 else dataset_split
    ds = load_dataset(dataset_name, split=split)
    return [{"problem": ex["problem"], "reference": ex["answer"]} for ex in ds]


class Generator:
    """Backend-agnostic generator for data collection.

    Exposes a single `.generate(prompts, num_return_sequences, max_new_tokens, temperature, top_p, seed)` method
    that returns a `list[list[str]]` (outer: one per prompt; inner: `num_return_sequences` decoded completions),
    regardless of whether the backing engine is `transformers` or `vllm`.
    """

    def __init__(self, model_name_or_path: str, *, use_vllm: bool, batch_size: int = 16):
        self.use_vllm = use_vllm
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_vllm:
            if not is_vllm_available():
                raise ImportError("vLLM is not installed; install `trl[vllm]` or drop `--use_vllm`.")
            from vllm import LLM

            self.llm = LLM(model=model_name_or_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to("cuda").eval()

    def generate(
        self,
        prompts: list[str],
        *,
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        if self.use_vllm:
            return self._generate_vllm(prompts, num_return_sequences, max_new_tokens, temperature, top_p, seed)
        return self._generate_hf(prompts, num_return_sequences, max_new_tokens, temperature, top_p, seed)

    def _generate_vllm(self, prompts, n, max_tokens, temperature, top_p, seed):
        from vllm import SamplingParams

        params = SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
        outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
        # vLLM returns outputs in the same order as input prompts; each has `n` CompletionOutputs.
        return [[o.text for o in r.outputs] for r in outputs]

    def _generate_hf(self, prompts, n, max_tokens, temperature, top_p, seed):
        torch.manual_seed(seed)
        device = next(self.model.parameters()).device
        all_completions: list[list[str]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).to(device)
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
            for j in range(len(batch_prompts)):
                all_completions.append(decoded[j * n : (j + 1) * n])
        return all_completions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a self-revision dataset for SRTTrainer.")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="HF repo or local path of the model used to generate both initial and revised responses.",
    )
    parser.add_argument(
        "--dataset_name",
        default="open-r1/OpenR1-Math-220k",
        help="Seed dataset. Must expose a `problem` column (problem statement) and an `answer` column "
        "(gold final answer, LaTeX-compatible for the default verifier).",
    )
    parser.add_argument(
        "--dataset_split",
        default="train",
        help="Split to load from `--dataset_name`.",
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=256,
        help="Number of seed problems to load from the dataset split. Use a small value for quick runs; "
        "scale up for real training data. `<= 0` loads the entire split.",
    )
    parser.add_argument(
        "--num_revisions",
        type=int,
        default=3,
        help="Number of revised responses to sample per problem. More revisions raise the chance that at least "
        "one verifies as correct (and thus survives filtering) at the cost of more generations.",
    )
    parser.add_argument(
        "--max_init_tokens",
        type=int,
        default=512,
        help="Max new tokens when sampling the initial response `y_init`. Low defaults are for smoke tests; "
        "raise this for non-trivial problems.",
    )
    parser.add_argument(
        "--max_revised_tokens",
        type=int,
        default=512,
        help="Max new tokens when sampling each revised response `y_revised`. Same guidance as `--max_init_tokens`.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used for both initial and revision generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling cutoff used for both initial and revision generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed. Initial-response generation uses `seed`, revision generation uses `seed + 1`.",
    )
    parser.add_argument(
        "--separator",
        default="\n\n",
        help="Plain-text separator inserted between `y_init`, the control prompt, and the revision slot when "
        "composing the revision prompt. Should match the `separator` used by SRTTrainer at train time.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory for the collected dataset. Written via `datasets.save_to_disk`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Prompts per generation batch. Only used by the transformers backend; vLLM batches internally.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation (requires the `vllm` optional dependency). Defaults to transformers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    generator = Generator(args.model_name_or_path, use_vllm=args.use_vllm, batch_size=args.batch_size)
    tokenizer = generator.tokenizer

    seed_rows = load_seed(args.dataset_name, args.dataset_split, args.num_problems)

    # Step 1: sample one initial response per problem.
    init_prompts = [render_initial_prompt(tokenizer, row["problem"]) for row in seed_rows]
    init_completions = generator.generate(
        init_prompts,
        num_return_sequences=1,
        max_new_tokens=args.max_init_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    y_inits = [c[0] for c in init_completions]

    # Step 2: verify each initial response and pick the control prompt (rephrase vs. restart).
    init_correct = verify_batch(y_inits, [row["reference"] for row in seed_rows])
    rows_for_revision = [
        {
            "problem": row["problem"],
            "reference": row["reference"],
            "y_init": y_init,
            "r_init": int(is_correct),
            "control_prompt": build_control_prompt(is_correct),
        }
        for row, y_init, is_correct in zip(seed_rows, y_inits, init_correct, strict=True)
    ]

    # Step 3: sample `num_revisions` revised responses per problem.
    revision_prompts = [
        render_revision_prompt(tokenizer, r["problem"], r["y_init"], r["control_prompt"], args.separator)
        for r in rows_for_revision
    ]
    revision_completions = generator.generate(
        revision_prompts,
        num_return_sequences=args.num_revisions,
        max_new_tokens=args.max_revised_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed + 1,
    )

    # Step 4: keep only revisions that verify as correct.
    flat_completions: list[str] = []
    flat_references: list[str] = []
    flat_source_row: list[dict] = []
    for row, completions in zip(rows_for_revision, revision_completions, strict=True):
        for y_revised in completions:
            flat_completions.append(y_revised)
            flat_references.append(row["reference"])
            flat_source_row.append(row)
    revised_correct = verify_batch(flat_completions, flat_references)

    collected = [
        {
            "problem": row["problem"],
            "y_init": row["y_init"],
            "r_init": row["r_init"],
            "control_prompt": row["control_prompt"],
            "y_revised": y_revised,
        }
        for row, y_revised, is_correct in zip(flat_source_row, flat_completions, revised_correct, strict=True)
        if is_correct
    ]

    if not collected:
        raise RuntimeError("No verified revisions were produced; try increasing `num_problems` or sampling budget.")

    Dataset.from_list(collected).save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
