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

"""Companion worker launched under ``accelerate launch --config_file <fsdp2>`` by the FSDP2 case in
``test_async_grpo_trainer.py``.

It runs a couple of :class:`AsyncGRPOTrainer` steps on an FSDP2-sharded model, driven by an in-process stub rollout
worker (no vLLM server, no NCCL weight transfer), and checks that training actually progresses under FSDP2: the loss is
finite and the parameters change. It then prints one machine-parseable result line (``ASYNC_GRPO_FSDP2_RESULT {json}``)
that the pytest side asserts on.

This is a *functional* FSDP2 smoke, not a performance microbenchmark. (An earlier version tried to count
``lm_head.weight`` all-gathers to answer PR #6077's per-chunk re-gather question, but under FSDP2 those gathers are
driven by autograd unshard hooks, not by ``DTensor.full_tensor``, and the trainer's own weight-sync path calls
``full_tensor`` on every parameter every step — so a ``full_tensor`` counter cannot isolate the chunked-logprob path.
The #6077 question is instead settled by static analysis: ``patch_chunked_lm_head`` uses a plain custom autograd
Function with no ``torch.utils.checkpoint`` recompute, so the per-chunk re-gather mechanism that PR #6077 fixed for
SFT's ``chunked_nll`` is structurally absent here.)

Self-contained on purpose (mirrors ``tests/experimental/_openreward_echo_env.py``): it imports only public TRL symbols
and carries its own stub, so it never imports pytest-internal classes across the subprocess boundary.
"""

from __future__ import annotations

import itertools
import json
import queue

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
RESULT_PREFIX = "ASYNC_GRPO_FSDP2_RESULT"


def dummy_reward_func(completions, **kwargs):
    # Mirrors tests/experimental/test_async_grpo_trainer.py: the stub pre-computes rewards, so this is
    # only here to satisfy the trainer's required `reward_funcs` argument.
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    """Minimal in-process rollout worker — same shape as the one in test_async_grpo_trainer.py.

    Reproduced here (rather than imported) because this module runs as ``__main__`` under ``accelerate launch``, not as
    a pytest module, so importing the test class would be fragile. Keeping it self-contained matches the openreward
    companion-script precedent.
    """

    def __init__(self, tokenizer, dataset, num_generations: int = 3, samples_per_weight_sync: int = 10):
        self.rollout_buffer = queue.Queue()
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._sample_iter = self._make_sample_iter(tokenizer, dataset, num_generations)

    def _make_sample_iter(self, tokenizer, dataset, num_generations):
        for row in itertools.cycle(dataset):
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(num_generations)
            ]
            prompt_completions = [row["prompt"] + completion for completion in completions]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                prompt_completions, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(num_generations):
                completion_ids = prompt_completion_ids[idx][len(prompt_ids) :]
                yield RolloutSample(
                    prompt=row["prompt"],
                    completion=completions[idx],
                    input_ids=prompt_ids + completion_ids,
                    completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
                    old_log_probs=[0.0] * len(prompt_ids) + [-0.5] * len(completion_ids),
                    advantage=float(advantages[idx]),
                    model_version=self._model_version,
                    metrics={"reward": float(rewards[idx]), "reward_std": float(rewards.std())},
                )

    def _fill_queue(self):
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        self._fill_queue()

    def update_model_version(self, version):
        self._model_version = version
        self._fill_queue()

    def stop(self):
        pass

    def check_health(self, stale_after_s):
        pass


def main() -> None:
    dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Same minimal, memory-frugal config as the existing single-process test_train, with 2 steps so we
    # exercise the optimizer loop more than once under FSDP2.
    args = AsyncGRPOConfig(
        output_dir="async_grpo_fsdp2_out",
        learning_rate=0.1,
        per_device_train_batch_size=3,
        num_generations=3,
        max_completion_length=8,
        max_steps=2,
        vllm_server_timeout=5.0,
        report_to="none",
    )
    trainer = AsyncGRPOTrainer(
        model=MODEL_ID,
        reward_funcs=dummy_reward_func,
        args=args,
        train_dataset=dataset,
        rollout_worker=_StubRolloutWorker(tokenizer, dataset, num_generations=3),
    )

    # Snapshot params before training so we can confirm FSDP2 training actually updated them.
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

    trainer.train()

    # Did any parameter change? Materialize DTensors (full_tensor) and move both operands to CPU before
    # comparing: the `before` snapshot is captured at construction (pre-FSDP-wrap, plain tensor) while the
    # post-train param is an FSDP2 DTensor on CUDA, so a direct torch.equal would raise a device mismatch.
    def _materialize(t):
        t = t.full_tensor() if isinstance(t, torch.distributed.tensor.DTensor) else t
        return t.detach().cpu()

    changed = False
    for n, p in trainer.model.named_parameters():
        if not torch.equal(_materialize(before[n]), _materialize(p)):
            changed = True
            break

    last = trainer.state.log_history[-1] if trainer.state.log_history else {}
    train_loss = last.get("train_loss")
    result = {
        "steps": trainer.state.global_step,
        "params_changed": changed,
        "train_loss_finite": train_loss is not None and bool(np.isfinite(train_loss)),
    }
    # Only rank 0 prints the asserted line, so the pytest side parses exactly one result.
    if trainer.accelerator.is_main_process:
        print(f"{RESULT_PREFIX} {json.dumps(result)}", flush=True)  # noqa: T201 - result channel for the launcher


if __name__ == "__main__":
    main()
