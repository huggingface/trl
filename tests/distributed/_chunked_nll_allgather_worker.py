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

"""Companion worker for the chunked-NLL FSDP2 all-gather perf-regression test.

Launched under ``accelerate launch --config_file <fsdp2_reshard>`` by ``test_distributed.py``. It runs a single SFT
``chunked_nll`` training step on an FSDP2-sharded tiny model and counts how many all-gather collectives occur during
that step, so a regression that re-gathers ``lm_head.weight`` once per vocab chunk (the PR #6077 failure mode — correct
loss, silently slow) is caught by a bounded assertion.

Why count real collectives and not ``DTensor.full_tensor()``: under FSDP2 the parameter unshard is driven by autograd
pre-hooks / c10d collectives, not by explicit ``full_tensor()`` calls, so a ``full_tensor`` counter is blind to it. We
use ``CommDebugMode`` (``torch.distributed.tensor.debug``) — torch's purpose-built, DTensor-native comm counter, which
records ``funcol.all_gather_into_tensor`` and the ``c10d`` ``_allgather_base_`` / ``allgather_`` variants that FSDP2
emits. (An earlier version also ran a hand-rolled ``TorchDispatchMode``, but re-dispatching sharded ops from a custom
mode under FSDP2 mismatches the index/weight devices on the embedding lookup, so we rely on ``CommDebugMode`` alone.)

Prints one machine-parseable line ``CHUNKED_NLL_ALLGATHER_RESULT {json}`` that the pytest side asserts on.
Self-contained (mirrors ``tests/experimental/_async_grpo_fsdp2_worker.py``): imports only public symbols.
"""

from __future__ import annotations

import json
import math

from datasets import load_dataset

from trl import SFTConfig, SFTTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
RESULT_PREFIX = "CHUNKED_NLL_ALLGATHER_RESULT"


def _count_all_gathers(comm_counts: dict) -> int:
    """Sum the all-gather collectives from a ``CommDebugMode.get_comm_counts()`` dict.

    ``CommDebugMode`` keys the dict by the comm op (funcol / c10d). FSDP2's parameter unshard shows up as
    ``funcol.all_gather_into_tensor`` (and the ``_allgather_base_`` / ``allgather_`` c10d variants), so we match on the
    op's string name containing ``all_gather`` / ``allgather`` and total those. This is the DTensor-native counter — it
    observes the autograd-hook-driven gathers that ``DTensor.full_tensor()`` is blind to, without a hand-rolled
    ``TorchDispatchMode`` (which mis-dispatches sharded ops under FSDP2).
    """
    total = 0
    for op, n in comm_counts.items():
        name = str(op).lower()
        if "all_gather" in name or "allgather" in name:
            total += int(n)
    return total


class _MeasuringSFTTrainer(SFTTrainer):
    """SFTTrainer that counts all-gather collectives during its first ``training_step``.

    The measurement must run *inside* the real ``trainer.train()`` loop, not by calling ``training_step`` directly:
    under ``fsdp_cpu_ram_efficient_loading`` the model is on CPU/meta until ``_inner_training_loop`` FSDP-wraps it and
    moves it to GPU. Calling ``training_step`` on ``trainer.model`` before ``train()`` runs the embedding lookup with a
    CPU weight against a CUDA input → device-mismatch crash. Overriding ``training_step`` lets the trainer do all
    wrapping/placement, while we wrap the (single, since ``max_steps=1``) step in ``CommDebugMode`` to tally the FSDP2
    unshard collectives.
    """

    comm_counts: dict | None = None

    def training_step(self, *args, **kwargs):
        from torch.distributed.tensor.debug import CommDebugMode

        # Only measure the first step (with max_steps=1 there is exactly one); guard anyway so the counts
        # reflect a single step even if the caller raises max_steps later.
        if self.comm_counts is not None:
            return super().training_step(*args, **kwargs)
        comm_mode = CommDebugMode()
        with comm_mode:
            loss = super().training_step(*args, **kwargs)
        self.comm_counts = comm_mode.get_comm_counts()
        return loss


def main() -> None:
    dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
    args = SFTConfig(
        output_dir="chunked_nll_fsdp2_out",
        loss_type="chunked_nll",
        per_device_train_batch_size=2,
        max_length=64,
        max_steps=1,
        report_to="none",
        bf16=True,
    )
    trainer = _MeasuringSFTTrainer(model=MODEL_ID, args=args, train_dataset=dataset)

    # vocab / chunk arithmetic: a per-chunk-regather regression would do ~ceil(vocab / chunk_size)
    # gathers of lm_head.weight per step; the fixed path does O(1). Computed, never hardcoded.
    from trl.trainer.sft_trainer import _CHUNKED_LM_HEAD_CHUNK_SIZE

    vocab_size = trainer.model.config.vocab_size
    n_chunks = -(-vocab_size // _CHUNKED_LM_HEAD_CHUNK_SIZE)  # ceil

    # Run the real training loop: it FSDP-wraps the model and moves it to GPU, then calls training_step
    # once (max_steps=1), which our subclass measures under CommDebugMode.
    trainer.train()

    comm_counts = trainer.comm_counts or {}
    all_gathers = _count_all_gathers(comm_counts)
    comm_total = sum(int(n) for n in comm_counts.values())

    last = trainer.state.log_history[-1] if trainer.state.log_history else {}
    train_loss = last.get("train_loss")

    result = {
        "vocab_size": int(vocab_size),
        "chunk_size": int(_CHUNKED_LM_HEAD_CHUNK_SIZE),
        "n_chunks_if_regressed": int(n_chunks),
        "all_gathers": int(all_gathers),
        "commdebug_total": int(comm_total),
        "loss_finite": train_loss is not None and math.isfinite(train_loss),
    }
    if trainer.accelerator.is_main_process:
        print(f"{RESULT_PREFIX} {json.dumps(result)}", flush=True)  # noqa: T201 - result channel for the launcher


if __name__ == "__main__":
    main()
