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

import random
from collections.abc import Callable
from typing import Any

import torch
from datasets import Dataset, IterableDataset

from ...extras.profiling import profiling_context
from ...trainer.grpo_trainer import GRPOTrainer
from ...trainer.utils import get_config_model_id
from .cgrpo_config import CGRPOConfig
from .conformal import (
    aps_score,
    conformal_quantile,
    first_success_score,
    pass_rate_score,
    prediction_set,
    select_delta_auto,
)


def _as_answer(value: Any) -> str:
    return "" if value is None else str(value).strip()


class CGRPOTrainer(GRPOTrainer):
    """
    Trainer for Conformal Group Relative Policy Optimization (C-GRPO).

    GRPO draws the same `num_generations` completions for every prompt. C-GRPO instead samples each prompt in
    increments along `budget_grid` and stops as soon as the split-conformal prediction set over the completions drawn
    so far is a singleton. One threshold per budget is calibrated on a held-out `calibration_dataset` with the current
    policy, at the first step and every `recalibrate_every` steps.

    Groups stay rectangular for the rest of [`GRPOTrainer`]: a prompt that stops at `k < max(budget_grid)` is padded
    with single-EOS completions. Padded rows get a NaN reward, so the group baseline is computed over real completions
    only and their advantage is zero, and their tokens are removed from the loss mask and the loss normalizer.

    Generation works with transformers and with vLLM in server or colocate mode. With vLLM, each budget increment is
    one generation request, so a training batch costs at most `len(budget_grid)` requests.

    Args:
        model, reward_funcs, **kwargs:
            As in [`GRPOTrainer`].
        args ([`experimental.cgrpo.CGRPOConfig`], *optional*):
            Training configuration.
        calibration_dataset ([`~datasets.Dataset`]):
            Held-out examples with a `"prompt"` column, disjoint from `train_dataset`. With `score="aps"` it also needs
            the column named by `args.calibration_answer_column`.
        answer_extractor (`Callable[[str], str | None]`, *optional*):
            Maps a decoded completion to its final answer. Required when `score="aps"`. Its output is compared, after
            `str(...).strip()`, to the calibration answer column.
        verifier (`Callable[..., list[bool]]`, *optional*):
            Called as `verifier(completions=list[str], **example)`, where `example` holds the dataset columns other
            than `"prompt"`; returns one pass/fail per completion. Required when `score` is `"first_success"` or
            `"pass_rate"`.
    """

    _tag_names = ["trl", "cgrpo"]

    def __init__(
        self,
        model,
        reward_funcs,
        args: CGRPOConfig | None = None,
        calibration_dataset: Dataset | None = None,
        answer_extractor: Callable[[str], str | None] | None = None,
        verifier: Callable[..., list[bool]] | None = None,
        **kwargs,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            args = CGRPOConfig(f"{model_name.split('/')[-1]}-CGRPO")
        for name in ("tools", "environment_factory", "rollout_func"):
            if kwargs.get(name) is not None:
                raise NotImplementedError(f"CGRPOTrainer does not support `{name}` yet.")
        if calibration_dataset is None:
            raise ValueError("CGRPOTrainer requires a `calibration_dataset` to fit the conformal thresholds.")
        if isinstance(calibration_dataset, IterableDataset):
            raise ValueError("`calibration_dataset` must be a map-style `datasets.Dataset`.")
        if "prompt" not in calibration_dataset.column_names:
            raise ValueError("`calibration_dataset` must have a `prompt` column.")
        if args.score == "aps":
            if answer_extractor is None:
                raise ValueError("`score='aps'` requires an `answer_extractor`.")
            if args.calibration_answer_column not in calibration_dataset.column_names:
                raise ValueError(
                    f"`calibration_dataset` has no column {args.calibration_answer_column!r} "
                    "(set `calibration_answer_column`)."
                )
        elif verifier is None:
            raise ValueError(f"`score={args.score!r}` requires a `verifier`.")

        super().__init__(model, reward_funcs, args=args, **kwargs)

        if self.accelerator.num_processes > 1:
            raise NotImplementedError(
                "CGRPOTrainer currently supports a single process: prompt groups can be split across processes, and "
                "the stopping rule needs every completion of a group on the same process."
            )
        if self._is_vlm:
            raise NotImplementedError("CGRPOTrainer does not support vision-language models yet.")

        self.calibration_dataset = calibration_dataset
        self.answer_extractor = answer_extractor
        self.verifier = verifier
        self.budget_grid = sorted(args.budget_grid)
        self.qhats: dict[int, float] | None = None
        self.delta: float | None = None
        self._rng = random.Random(args.seed)
        self._last_calibration_step: int | None = None
        self._batch_inputs: list[dict[str, Any]] | None = None
        self._padded_rows: list[bool] | None = None

    # ------------------------------------------------------------------ calibration

    def _outcomes(self, texts: list[str], example: dict[str, Any]) -> list:
        # Extracted answers for APS, pass/fail booleans for the execution scores.
        if self.args.score == "aps":
            return [_as_answer(self.answer_extractor(text)) for text in texts]
        fields = {key: value for key, value in example.items() if key != "prompt"}
        passes = self.verifier(completions=texts, **fields)
        if len(passes) != len(texts):
            raise ValueError(f"`verifier` returned {len(passes)} results for {len(texts)} completions.")
        return [bool(p) for p in passes]

    def _score(self, outcomes: list, example: dict[str, Any], k: int) -> float:
        if self.args.score == "aps":
            gold = _as_answer(example[self.args.calibration_answer_column])
            return aps_score(gold, outcomes, k, self._rng)
        if self.args.score == "first_success":
            return first_success_score(outcomes, k)
        return pass_rate_score(outcomes, k)

    def _calibration_due(self) -> bool:
        if self.qhats is None:
            return True
        every = self.args.recalibrate_every
        return every > 0 and self.state.global_step - self._last_calibration_step >= every

    @torch.no_grad()
    def calibrate(self) -> dict[int, float]:
        """Fit one conformal threshold per budget with the current policy. Returns `{k: qhat_k}`."""
        args = self.args
        k_max = self.budget_grid[-1]
        n = len(self.calibration_dataset)
        if args.num_calibration_samples is not None:
            n = min(n, args.num_calibration_samples)
        examples = [self.calibration_dataset[i] for i in range(n)]
        batch_size = args.calibration_batch_size or args.per_device_train_batch_size

        scores = []  # scores[i][k]
        for start in range(0, n, batch_size):
            chunk = examples[start : start + batch_size]
            prompts = [example["prompt"] for example in chunk for _ in range(k_max)]
            prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
            # Every prompt fills max(budget_grid) == num_generations rows, which is the layout the base path (and
            # vLLM server mode) expects.
            completion_ids, _ = super()._generate_single_turn(prompt_ids, images, multimodal_fields)
            texts = self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            for i, example in enumerate(chunk):
                outcomes = self._outcomes(texts[i * k_max : (i + 1) * k_max], example)
                scores.append({k: self._score(outcomes, example, k) for k in self.budget_grid})

        delta_indices = quantile_indices = list(range(n))
        if args.delta is None and args.split_delta:
            half = n // 2
            delta_indices, quantile_indices = delta_indices[:half], quantile_indices[half:]

        metrics = self._metrics["train"]
        if args.delta is None:
            self.delta, solve_rate = select_delta_auto(
                [scores[i][k_max] for i in delta_indices], margin=args.delta_margin
            )
            metrics["cgrpo/calibration_solve_rate"].append(solve_rate)
        else:
            self.delta = args.delta
        self.qhats = {
            k: conformal_quantile([scores[i][k] for i in quantile_indices], self.delta) for k in self.budget_grid
        }
        self._last_calibration_step = self.state.global_step

        metrics["cgrpo/delta"].append(self.delta)
        metrics["cgrpo/calibration_rollouts"].append(n * k_max)
        for k, qhat in self.qhats.items():
            metrics[f"cgrpo/qhat_{k}"].append(qhat)
        return dict(self.qhats)

    # ------------------------------------------------------------------ adaptive generation

    def _resolved(self, outcomes: list, k: int) -> bool:
        """Whether the conformal prediction set after `k` completions is a singleton."""
        qhat = self.qhats[k]
        if self.args.score == "aps":
            return len(prediction_set(outcomes, k, qhat, self._rng)) == 1
        # For execution scores the set is {pass} once a passing completion has been seen early enough that its score
        # falls within the calibrated threshold.
        if not any(outcomes[:k]):
            return False
        score = (
            first_success_score(outcomes, k) if self.args.score == "first_success" else pass_rate_score(outcomes, k)
        )
        return score <= qhat

    def _generate_and_score_completions(self, inputs):
        training = self.model.training
        if training and self._calibration_due():
            self.calibrate()
        self._batch_inputs = inputs if training else None
        self._padded_rows = None
        try:
            output = super()._generate_and_score_completions(inputs)
            if self._padded_rows is not None and any(self._padded_rows):
                padded = torch.tensor(self._padded_rows, device=output["completion_mask"].device)
                padded_tokens = output["completion_mask"][padded].sum()
                output["completion_mask"][padded] = 0
                output["num_items_in_batch"] = output["num_items_in_batch"] - padded_tokens
        finally:
            self._batch_inputs = None
            self._padded_rows = None
        return output

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if self._padded_rows is not None:
            padded = torch.tensor(self._padded_rows, device=rewards_per_func.device)
            # NaN in every column marks the row unscorable: GRPOTrainer excludes it from the group baseline and sets
            # its advantage to zero.
            rewards_per_func[padded] = torch.nan
        return rewards_per_func

    def _sample(self, prompt_ids, num_generations, has_tool_images=False):
        """
        Generate one completion per row, where every prompt occupies `num_generations` consecutive rows.

        GRPOTrainer's vLLM path assumes `self.num_generations` rows per prompt (server mode samples
        `prompts[::num_generations]`), but budget increments are smaller, so vLLM is called with the increment size.
        """
        if not self.use_vllm:
            return super()._generate_single_turn(prompt_ids, None, {}, has_tool_images)
        if self.state.global_step != self._last_loaded_step:
            with profiling_context(self, "sync_weights"):
                self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step
        _, completion_ids, logprobs, _ = self.vllm_generation.generate(
            prompts=prompt_ids,
            images=None,
            num_generations=num_generations,
            profiler=profiling_context(self, "vLLM.generate"),
        )
        if logprobs is not None:  # per-token top-k logprobs; keep the sampled token's
            logprobs = [[lp[0] for lp in seq] for seq in logprobs]
        return [list(ids) for ids in completion_ids], logprobs

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
        if self._batch_inputs is None:  # evaluation, or any generation outside a training batch
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields, has_tool_images)

        group_size = self.num_generations
        if len(prompt_ids) % group_size != 0:
            raise RuntimeError(f"Expected a multiple of num_generations={group_size} rows, got {len(prompt_ids)}.")
        num_groups = len(prompt_ids) // group_size
        heads = []
        for g in range(num_groups):
            block = prompt_ids[g * group_size : (g + 1) * group_size]
            if any(list(ids) != list(block[0]) for ids in block[1:]):
                raise RuntimeError(
                    "CGRPOTrainer expects each prompt repeated num_generations times in consecutive rows, as produced "
                    "by GRPOTrainer's RepeatSampler."
                )
            heads.append(block[0])

        completions = [[] for _ in range(num_groups)]
        logprobs = [[] for _ in range(num_groups)]
        outcomes = [[] for _ in range(num_groups)]
        k_used = [self.budget_grid[-1]] * num_groups
        has_logprobs = False
        active, drawn = list(range(num_groups)), 0
        for k in self.budget_grid:
            need = k - drawn
            if need > 0:
                batch = [heads[g] for g in active for _ in range(need)]
                new_ids, new_logprobs = self._sample(batch, need, has_tool_images)
                has_logprobs = new_logprobs is not None
                texts = self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                for j, g in enumerate(active):
                    rows = slice(j * need, (j + 1) * need)
                    completions[g].extend(new_ids[rows])
                    if has_logprobs:
                        logprobs[g].extend(new_logprobs[rows])
                    outcomes[g].extend(self._outcomes(texts[rows], self._batch_inputs[g * group_size]))
            drawn = k
            still_active = []
            for g in active:
                if self._resolved(outcomes[g], k):
                    k_used[g] = k
                else:
                    still_active.append(g)
            active = still_active
            if not active:
                break

        pad_id = (
            self._tokenizer.eos_token_id if self._tokenizer.eos_token_id is not None else self._tokenizer.pad_token_id
        )
        completion_ids, completion_logprobs, padded = [], [] if has_logprobs else None, []
        for g in range(num_groups):
            for j in range(group_size):
                is_real = j < k_used[g]
                completion_ids.append(completions[g][j] if is_real else [pad_id])
                if has_logprobs:
                    # NaN sampling logprob: GRPOTrainer treats it as unavailable, so a padded row's vLLM
                    # importance-sampling ratio is exactly 1 rather than a spurious correction.
                    completion_logprobs.append(logprobs[g][j] if is_real else [float("nan")])
                padded.append(not is_real)
        self._padded_rows = padded

        metrics = self._metrics["train"]
        metrics["cgrpo/mean_k"].append(sum(k_used) / num_groups)
        metrics["cgrpo/rollout_savings"].append(1.0 - sum(k_used) / (num_groups * group_size))
        return completion_ids, completion_logprobs
