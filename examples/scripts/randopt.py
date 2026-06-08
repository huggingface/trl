#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
import os
import random
import re
import time

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.data_utils import maybe_apply_chat_template
from trl.experimental.randopt import RandOptConfig, majority_vote
from trl.experimental.sdpo.sdpo import _extract_predicted_answer, _make_conversation

ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")
_WORKER_STATE: dict = {}


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K RandOpt full experiment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--population_size", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--sigma_values", type=str, default="0.0001,0.0005,0.001,0.002")
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--test_log_every", type=int, default=2)
    parser.add_argument("--prompt_style", type=str, choices=["plain", "sdpo"], default="plain")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel GPU workers for perturbation eval.")
    return parser.parse_args()


def build_prompt_texts(tokenizer, prompts: list[list[dict]], questions: list[str], prompt_style: str) -> list[str]:
    if prompt_style == "sdpo":
        return [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]

    # Plain prompt style keeps only user question and tends to be a stronger base-eval setting for instruct models.
    prompt_texts: list[str] = []
    for question in questions:
        prompt = [{"role": "user", "content": question}]
        prompt_texts.append(maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"])
    return prompt_texts


def generate_answers(model, tokenizer, prompt_texts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    out_texts: list[str] = []
    model.eval()
    for i in range(0, len(prompt_texts), batch_size):
        batch_prompts = prompt_texts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        out_texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return out_texts


def _normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().replace("$", "").replace(",", "")
    if not cleaned:
        return ""
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value.is_integer():
        return str(int(value))
    return str(value)


def _extract_predicted_answer_robust(completion_text: str) -> str:
    # First try TRL's native GSM8K extractor.
    pred = _extract_predicted_answer(completion_text)
    normalized = _normalize_numeric_text(pred)
    if normalized:
        return normalized

    # If model outputs XML-like tags, prefer the last <answer>...</answer>.
    tag_matches = ANSWER_TAG_RE.findall(completion_text)
    if tag_matches:
        nums = NUMBER_RE.findall(tag_matches[-1])
        if nums:
            return _normalize_numeric_text(nums[-1])
        return _normalize_numeric_text(tag_matches[-1])

    # If multiple "#### ..." segments exist, take the last one.
    hash_matches = HASH_ANSWER_RE.findall(completion_text)
    if hash_matches:
        nums = NUMBER_RE.findall(hash_matches[-1])
        if nums:
            return _normalize_numeric_text(nums[-1])
        return _normalize_numeric_text(hash_matches[-1])

    # Final fallback: last number anywhere in completion.
    nums = NUMBER_RE.findall(completion_text)
    if nums:
        return _normalize_numeric_text(nums[-1])
    return ""


def score_gsm8k_completions(completions: list[str], solutions: list[str]) -> tuple[float, int, list[str]]:
    preds = [_extract_predicted_answer_robust(completion) for completion in completions]
    normalized_solutions = [_normalize_numeric_text(solution) for solution in solutions]
    correct = sum(
        1
        for pred, gold in zip(preds, normalized_solutions, strict=True)
        if pred != "" and gold != "" and pred == gold
    )
    total = max(len(solutions), 1)
    return correct / total, correct, preds


@torch.no_grad()
def _apply_noise_inplace(params: list[torch.nn.Parameter], seed: int, sigma: float):
    generators: dict[torch.device, torch.Generator] = {}
    for param in params:
        device = param.device
        if device not in generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            generators[device] = generator
        noise = torch.randn(param.shape, generator=generators[device], device=device, dtype=param.dtype)
        param.add_(noise, alpha=sigma)


@torch.no_grad()
def _restore_base_inplace(params: list[torch.nn.Parameter], base_state: list[torch.Tensor]):
    for param, base in zip(params, base_state, strict=True):
        param.copy_(base)


def _worker_init(
    model_name: str,
    dtype_name: str,
    visible_devices: list[str],
    train_prompts: list[str],
    train_solutions: list[str],
    test_prompts: list[str],
    test_solutions: list[str],
    batch_size: int,
    max_new_tokens: int,
):
    rank = max(0, mp.current_process()._identity[0] - 1) if mp.current_process()._identity else 0
    local_device = rank % max(1, len(visible_devices))
    torch.cuda.set_device(local_device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map={"": f"cuda:{local_device}"},
    )
    params = [param for param in model.parameters() if param.requires_grad and torch.is_floating_point(param)]
    base_state = [param.detach().clone() for param in params]

    _WORKER_STATE.update(
        {
            "tokenizer": tokenizer,
            "model": model,
            "params": params,
            "base_state": base_state,
            "train_prompts": train_prompts,
            "train_solutions": train_solutions,
            "test_prompts": test_prompts,
            "test_solutions": test_solutions,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "local_device": local_device,
        }
    )


def _worker_generate(prompt_texts: list[str]) -> list[str]:
    tokenizer = _WORKER_STATE["tokenizer"]
    model = _WORKER_STATE["model"]
    batch_size = _WORKER_STATE["batch_size"]
    max_new_tokens = _WORKER_STATE["max_new_tokens"]
    return generate_answers(model, tokenizer, prompt_texts, batch_size=batch_size, max_new_tokens=max_new_tokens)


def _worker_eval_train(task: tuple[int, float]) -> tuple[int, float, float]:
    seed, sigma = task
    params = _WORKER_STATE["params"]
    base_state = _WORKER_STATE["base_state"]
    with torch.no_grad():
        _apply_noise_inplace(params, seed=seed, sigma=sigma)
    outputs = _worker_generate(_WORKER_STATE["train_prompts"])
    acc, _, _ = score_gsm8k_completions(outputs, _WORKER_STATE["train_solutions"])
    with torch.no_grad():
        _restore_base_inplace(params, base_state)
    return seed, sigma, acc


def _worker_eval_test(task: tuple[int, float]) -> tuple[int, float, float, list[str]]:
    seed, sigma = task
    params = _WORKER_STATE["params"]
    base_state = _WORKER_STATE["base_state"]
    with torch.no_grad():
        _apply_noise_inplace(params, seed=seed, sigma=sigma)
    outputs = _worker_generate(_WORKER_STATE["test_prompts"])
    acc, _, preds = score_gsm8k_completions(outputs, _WORKER_STATE["test_solutions"])
    with torch.no_grad():
        _restore_base_inplace(params, base_state)
    return seed, sigma, acc, preds


def _worker_eval_base() -> tuple[float, int]:
    outputs = _worker_generate(_WORKER_STATE["test_prompts"])
    acc, correct, _ = score_gsm8k_completions(outputs, _WORKER_STATE["test_solutions"])
    return acc, correct


def _get_visible_devices() -> list[str]:
    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        devices = [item.strip() for item in env_devices.split(",") if item.strip() != ""]
        if devices:
            return devices
    return [str(i) for i in range(torch.cuda.device_count())]


def run_experiment(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"gsm8k_pop{args.population_size}_top{args.top_k}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[INFO] loading gsm8k dataset (trl/sdpo format)...", flush=True)
    train_raw = load_dataset("openai/gsm8k", "main", split=f"train[:{args.train_samples}]")
    test_raw = load_dataset("openai/gsm8k", "main", split=f"test[:{args.test_samples}]")
    train_data = [dict(example) for example in train_raw]
    test_data = [dict(example) for example in test_raw]
    train_conv = [_make_conversation(example, feedback_column=None, feedback_from_solution="final_answer") for example in train_data]
    test_conv = [_make_conversation(example, feedback_column=None, feedback_from_solution="final_answer") for example in test_data]
    train_solutions = [example["solution"] for example in train_conv]
    test_solutions = [example["solution"] for example in test_conv]
    train_questions = [example["question"] for example in train_data]
    test_questions = [example["question"] for example in test_data]
    train_prompts = build_prompt_texts(
        tokenizer, [example["prompt"] for example in train_conv], train_questions, args.prompt_style
    )
    test_prompts = build_prompt_texts(tokenizer, [example["prompt"] for example in test_conv], test_questions, args.prompt_style)
    print(
        f"[INFO] eval config: prompt_style={args.prompt_style} max_new_tokens={args.max_new_tokens} "
        f"train_samples={args.train_samples} test_samples={args.test_samples}",
        flush=True,
    )

    sigma_values = [float(x.strip()) for x in args.sigma_values.split(",") if x.strip()]
    config = RandOptConfig(population_size=args.population_size, sigma_values=sigma_values, top_k=args.top_k, base_seed=args.seed)
    rng = random.Random(args.seed)
    candidate_tasks = [(args.seed + i, rng.choice(sigma_values)) for i in range(args.population_size)]

    visible_devices = _get_visible_devices()
    worker_count = min(args.num_workers, len(visible_devices), args.population_size)
    if worker_count < 1:
        raise RuntimeError("No CUDA devices available for RandOpt parallel execution.")

    print(
        f"[INFO] running RandOpt search in parallel: workers={worker_count} visible_devices={visible_devices}",
        flush=True,
    )
    start_time = time.time()
    perf: dict[tuple[int, float], float] = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(
            args.model_name,
            "bfloat16" if torch.cuda.is_available() else "float32",
            visible_devices,
            train_prompts,
            train_solutions,
            test_prompts,
            test_solutions,
            args.batch_size,
            args.max_new_tokens,
        ),
    ) as pool:
        print("[INFO] evaluating base model...", flush=True)
        base_acc, base_correct = pool.submit(_worker_eval_base).result()
        print(f"[BASE] test_acc={base_acc:.4f} ({base_correct}/{len(test_solutions)})", flush=True)

        futures = [pool.submit(_worker_eval_train, task) for task in candidate_tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            seed, sigma, acc = future.result()
            perf[(seed, sigma)] = acc
            best_score = max(perf.values()) if perf else 0.0
            if args.log_every > 0 and (idx % args.log_every == 0 or idx == 1 or idx == args.population_size):
                elapsed = time.time() - start_time
                speed = idx / elapsed if elapsed > 0 else 0.0
                remain = (args.population_size - idx) / speed if speed > 0 else float("inf")
                remain_str = f"{remain/60:.1f}m" if remain != float("inf") else "inf"
                print(
                    f"[RANDOPT] {idx}/{args.population_size} seed={seed} sigma={sigma} "
                    f"train_score={acc:.4f} best={best_score:.4f} eta={remain_str}",
                    flush=True,
                )

        sorted_items = sorted(perf.items(), key=lambda item: item[1], reverse=True)
        top_candidate_pairs = [(seed, sigma) for (seed, sigma), _ in sorted_items[: config.top_k]]
        top_candidate_scores = [score for _, score in sorted_items[: config.top_k]]

        print(f"[INFO] search complete. kept top={len(top_candidate_pairs)}", flush=True)
        test_futures = [pool.submit(_worker_eval_test, task) for task in top_candidate_pairs]
        test_map: dict[tuple[int, float], tuple[float, list[str]]] = {}
        test_start_time = time.time()
        total_test = len(top_candidate_pairs)
        for idx, future in enumerate(as_completed(test_futures), start=1):
            seed, sigma, acc, preds = future.result()
            test_map[(seed, sigma)] = (acc, preds)
            if args.test_log_every > 0 and (idx % args.test_log_every == 0 or idx == 1 or idx == total_test):
                elapsed = time.time() - test_start_time
                speed = idx / elapsed if elapsed > 0 else 0.0
                remain = (total_test - idx) / speed if speed > 0 else float("inf")
                remain_str = f"{remain/60:.1f}m" if remain != float("inf") else "inf"
                print(
                    f"[TOPK-EVAL] {idx}/{total_test} seed={seed} sigma={sigma} "
                    f"test_acc={acc:.4f} eta={remain_str}",
                    flush=True,
                )

    # Evaluate top-k candidates on test split and majority-vote ensembles at K={1,5,10,20,50}
    model_answers: list[list[str]] = []
    top_candidates = []
    for rank, ((seed, sigma), train_score) in enumerate(zip(top_candidate_pairs, top_candidate_scores, strict=True), start=1):
        test_acc, preds = test_map[(seed, sigma)]
        model_answers.append(preds)
        top_candidates.append({"rank": rank, "seed": seed, "sigma": sigma, "train_score": train_score})
        print(
            f"[CANDIDATE {rank:03d}] seed={seed} sigma={sigma} train_score={train_score:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )

    k_values = list(dict.fromkeys([1, 5, 10, 20, args.top_k]))
    k_values = [k for k in k_values if k <= len(model_answers)]
    ensemble_results = {}
    for k in k_values:
        voted = majority_vote(model_answers, k=k, empty_answer="")
        normalized_solutions = [_normalize_numeric_text(solution) for solution in test_solutions]
        correct = sum(
            1
            for pred, gold in zip(voted, normalized_solutions, strict=True)
            if pred != "" and gold != "" and pred == gold
        )
        acc = correct / len(test_solutions)
        ensemble_results[str(k)] = {"acc": acc, "correct": int(correct), "total": len(test_solutions)}
        print(f"[ENSEMBLE K={k}] test_acc={acc:.4f} ({correct}/{len(test_solutions)})", flush=True)

    results = {
        "args": vars(args),
        "base_test_acc": base_acc,
        "top_candidates": top_candidates,
        "ensemble_results": ensemble_results,
    }
    result_path = os.path.join(run_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] results saved to {result_path}", flush=True)


if __name__ == "__main__":
    run_experiment(parse_args())
