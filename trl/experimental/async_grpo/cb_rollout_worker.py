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
In-process rollout worker for `AsyncGRPOTrainer` backed by transformers' continuous batching.

Implements `RolloutWorkerProtocol` so it drops into `AsyncGRPOTrainer` in place of `AsyncRolloutWorker`. Unlike the
vLLM-server worker, no external server, weight transfer, or HTTP is involved — the same in-process model object is used
for both training and generation. A `threading.Lock` (exposed as `weight_lock`) guards parameter mutation: the producer
acquires it for the entire `add_requests + get_result + compute old_log_probs` window, and `AsyncGRPOTrainer` must
acquire it around `optimizer.step` (registered via `_OptimizerLockCallback`).
"""

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from datasets import Dataset
from transformers import GenerationConfig, PreTrainedTokenizerBase

from trl.trainer.utils import print_prompt_completions_sample

from .async_rollout_worker import RolloutGroup, RolloutSample


logger = get_logger(__name__)


class CBRolloutWorker(threading.Thread):
    """Producer thread that fills `rollout_buffer` with `RolloutSample`s using continuous batching."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        reward_funcs: list[Callable[..., list[float]]],
        num_generations: int = 8,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        queue_maxsize: int = 0,
        chat_template_kwargs: dict[str, Any] | None = None,
        log_completions: bool = False,
        num_completions_to_print: int = 3,
    ):
        super().__init__(daemon=True)
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self._dataset_iter = iter(dataset)
        self.reward_funcs = reward_funcs
        self.reward_func_names = [getattr(f, "__name__", "reward") for f in reward_funcs]
        self.num_generations = num_generations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.log_completions = log_completions
        self.num_completions_to_print = num_completions_to_print

        self.rollout_buffer: queue.Queue[RolloutSample] = queue.Queue(maxsize=queue_maxsize)
        self.weight_lock = threading.Lock()
        self.model_version = 0
        self._stop_event = threading.Event()
        self._paused = threading.Event()  # mostly cosmetic; the lock is what actually gates the worker
        self._total_groups_scored = 0

    # ---- RolloutWorkerProtocol -------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()

    def pause(self) -> None:
        # CB shares parameters with the trainer; pausing is a no-op. The real mutual exclusion happens
        # via `weight_lock` (acquired by the trainer's `_OptimizerLockCallback` around `optimizer.step`).
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def send_weights(self, iterator) -> None:
        # Same model object as the trainer's — no transfer needed. We still consume the iterator so
        # that FSDP `full_tensor()` collectives on non-rank-0 ranks complete.
        for _ in iterator:
            pass

    def update_model_version(self, model_version: int) -> None:
        self.model_version = model_version

    # ---- Main loop -------------------------------------------------------------------------------
    def run(self) -> None:
        gen_config = GenerationConfig(
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Persistent CB manager — warm up once.
        manager = self.model.init_continuous_batching(generation_config=gen_config)
        manager.start()
        try:
            while not self._stop_event.is_set():
                row = self._next_row()
                if row is None:
                    return
                samples = self._rollout_one_group(row, manager)
                if self.log_completions and samples:
                    print_prompt_completions_sample(
                        prompts=[s.prompt for s in samples],
                        completions=[s.completion for s in samples],
                        rewards={"reward": [s.metrics.get("reward", 0.0) for s in samples]},
                        advantages=[s.advantage for s in samples],
                        step=self._total_groups_scored,
                        num_samples=self.num_completions_to_print,
                    )
                self._total_groups_scored += 1
                for sample in samples:
                    while not self._stop_event.is_set():
                        try:
                            self.rollout_buffer.put(sample, timeout=0.5)
                            break
                        except queue.Full:
                            continue
        except Exception:
            logger.exception("CBRolloutWorker crashed")
            raise
        finally:
            manager.stop(block=True, timeout=5.0)

    def _next_row(self) -> dict[str, Any] | None:
        try:
            return next(self._dataset_iter)
        except StopIteration:
            self._dataset_iter = iter(self.dataset)
            try:
                return next(self._dataset_iter)
            except StopIteration:
                return None

    def _rollout_one_group(self, row: dict[str, Any], manager) -> list[RolloutSample]:
        prompt = row["prompt"]
        prompt_ids = self._encode_prompt(prompt)
        inputs = [prompt_ids] * self.num_generations

        # Generate + recompute old_log_probs under a single lock acquisition so the version is stable.
        with self.weight_lock:
            version_at_gen = self.model_version
            was_training = self.model.training
            self.model.eval()
            try:
                request_ids = manager.add_requests(inputs=inputs, max_new_tokens=self.max_tokens)
                pending = set(request_ids)
                results: dict[str, Any] = {}
                while pending and not self._stop_event.is_set():
                    result = manager.get_result(timeout=1.0)
                    if result is None:
                        if not manager.is_running():
                            raise RuntimeError("Continuous batching manager terminated unexpectedly.")
                        continue
                    if result.is_finished() and result.request_id in pending:
                        results[result.request_id] = result
                        pending.remove(result.request_id)
                completions_ids = [results[rid].generated_tokens for rid in request_ids]
                old_log_probs_list = self._compute_old_log_probs(prompt_ids, completions_ids)
            finally:
                if was_training:
                    self.model.train()

        completions_text = self.tokenizer.batch_decode(completions_ids, skip_special_tokens=True)
        completion_messages = [[{"role": "assistant", "content": t}] for t in completions_text]

        rewards, per_func_rewards = self._compute_rewards(prompt, row, completions_text, completions_ids)
        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())
        advantages = (rewards - reward_mean) / (reward_std + 1e-8)

        return [
            RolloutSample(
                prompt=prompt,
                completion=completion_messages[i],
                input_ids=list(prompt_ids) + list(completions_ids[i]),
                completion_mask=[0] * len(prompt_ids) + [1] * len(completions_ids[i]),
                old_log_probs=[0.0] * len(prompt_ids) + old_log_probs_list[i],
                advantage=float(advantages[i]),
                model_version=version_at_gen,
                metrics={
                    "reward": float(rewards[i]),
                    "reward_std": reward_std,
                    **{
                        f"rewards/{name}": float(per_func_rewards[fi, i])
                        for fi, name in enumerate(self.reward_func_names)
                    },
                    "buffer_qsize": float(self.rollout_buffer.qsize()),
                },
            )
            for i in range(self.num_generations)
        ]

    def _compute_old_log_probs(self, prompt_ids: list[int], completions_ids: list[list[int]]) -> list[list[float]]:
        """One extra `no_grad` forward over `[prompt + completion]` to record the policy logprobs that
        produced each sampled token. Used by `AsyncGRPOTrainer.compute_loss` as `old_log_probs` in the PPO importance
        ratio."""
        device = self.model.device
        n = len(prompt_ids)
        seqs = [torch.tensor(prompt_ids + list(c), dtype=torch.long, device=device) for c in completions_ids]
        lens = [s.size(0) for s in seqs]
        max_len = max(lens)
        input_ids = torch.full((len(seqs), max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, : lens[i]] = s
            attention_mask[i, : lens[i]] = 1

        with torch.inference_mode():
            # Note: `patch_chunked_lm_head` falls back to standard forward when labels is None.
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = out.logits / max(self.temperature, 1e-6)
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = input_ids[:, 1:]
            gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        result = []
        for i, c in enumerate(completions_ids):
            m = len(c)
            # Completion logprobs live at shifted positions [n-1, ..., n+m-2].
            slc = gathered[i, n - 1 : n - 1 + m].tolist()
            result.append(slc)
        return result

    def _compute_rewards(
        self,
        prompt,
        row: dict[str, Any],
        completions_text: list[str],
        completions_ids: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        reward_kwargs = {k: [v] * self.num_generations for k, v in row.items() if k != "prompt"}
        kwargs = dict(
            prompts=[prompt] * self.num_generations,
            completions=completions_text,
            completion_ids=completions_ids,
            **reward_kwargs,
        )
        per_func = []
        for func in self.reward_funcs:
            r = func(**kwargs)
            r = [x if x is not None else float("nan") for x in r]
            per_func.append(r)
        per_func_rewards = np.array(per_func, dtype=float)
        rewards = np.nansum(per_func_rewards, axis=0)
        return rewards, per_func_rewards

    def _encode_prompt(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            return self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.chat_template_kwargs,
            )
        return self.tokenizer.encode(prompt)
