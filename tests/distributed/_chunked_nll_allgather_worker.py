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
that step, so a regression that re-gathers ``lm_head.weight`` once per token chunk (the PR #6077 failure mode — correct
loss, silently slow) is caught by a bounded assertion.

``_chunked_cross_entropy_loss`` chunks over *valid tokens* (``for start in range(0, n_valid, chunk_size)``), so the
regression scales with ``ceil(n_valid / chunk_size)`` and only shows up when more than one token chunk runs. The zen
test data is tiny, so this worker shrinks the chunk size (see ``_TEST_CHUNK_SIZE``) to force many token chunks, and
derives the regression threshold from the exact ``n_valid`` captured from inside the loss path — never from vocab size.

Why count real collectives and not ``DTensor.full_tensor()``: under FSDP2 the parameter unshard is driven by autograd
pre-hooks / c10d collectives, not by explicit ``full_tensor()`` calls, so a ``full_tensor`` counter is blind to it. We
use ``CommDebugMode`` (``torch.distributed.tensor.debug``) — torch's purpose-built, DTensor-native comm counter, which
records ``funcol.all_gather_into_tensor`` and the ``c10d`` ``_allgather_base_`` / ``allgather_`` variants that FSDP2
emits. (An earlier version also ran a hand-rolled ``TorchDispatchMode``, but re-dispatching sharded ops from a custom
mode under FSDP2 mismatches the index/weight devices on the embedding lookup, so we rely on ``CommDebugMode`` alone.)

Prints one machine-parseable line ``CHUNKED_NLL_ALLGATHER_RESULT {json}`` that the pytest side asserts on.
Self-contained on purpose: it imports only public TRL symbols and runs as ``__main__`` under ``accelerate launch``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

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


# Config for the measured step. Promoted to module constants so the baseline chunk size can be derived from
# them (see `_BASELINE_CHUNK_SIZE`) rather than hardcoded.
_PER_DEVICE_TRAIN_BATCH_SIZE = 8
_MAX_LENGTH = 64

# Probe run. The chunked-CE loop chunks over *valid tokens*, not vocab: `for start in range(0, n_valid,
# chunk_size)` in `_chunked_cross_entropy_loss`. So a per-chunk `lm_head.weight` re-gather regression scales
# with ceil(n_valid / chunk_size) — the TOKEN-chunk count — and is only observable when more than one chunk
# runs (n_valid > chunk_size). The zen test data is tiny (~120 valid tokens total), so with the default chunk
# size of 256 only a single chunk would run and a regression would be invisible. We therefore shrink the chunk
# size for this run so the tiny batch genuinely exercises many token-chunks.
_TEST_CHUNK_SIZE = 4

# Baseline run. A chunk size at least as large as the valid-token count forces exactly ONE token chunk, so the
# per-chunk regather signal is zero by construction and the measured all-gather count is the pure FSDP2
# parameter-unshard baseline B (one unshard per sharded param per fwd/bwd, independent of the chunk count).
# `per_device_train_batch_size * max_length` is the hard upper bound on valid tokens, so it guarantees a single
# chunk without over-allocating logits. Derived from the config above — never a magic collective count.
_BASELINE_CHUNK_SIZE = _PER_DEVICE_TRAIN_BATCH_SIZE * _MAX_LENGTH


def main() -> None:
    import trl.trainer.sft_trainer as sft

    # Two modes, selected by the launcher via CHUNKED_NLL_MODE:
    #   "probe"    (default): shrink the chunk size so many token chunks run; a per-chunk re-gather regression
    #              then does ~ceil(n_valid / chunk_size) lm_head all-gathers vs O(1) for the fixed path.
    #   "baseline": use a chunk size >= the valid-token count so exactly one token chunk runs, giving the
    #              regression-free FSDP2 unshard baseline B that the probe count is compared against.
    # The chunk size must be set BEFORE the trainer patches the lm_head (it reads this module constant at
    # construction).
    mode = os.environ.get("CHUNKED_NLL_MODE", "probe")
    chunk_size = _BASELINE_CHUNK_SIZE if mode == "baseline" else _TEST_CHUNK_SIZE
    sft._CHUNKED_LM_HEAD_CHUNK_SIZE = chunk_size

    # Capture the real valid-token count from inside the chunked-CE path, so the regression threshold is
    # derived from the exact n_valid the loop iterates over (never guessed from token lengths).
    captured = {}
    _orig_cce = sft._chunked_cross_entropy_loss

    def _capturing_cce(hidden_states, lm_head_weight, chunk_size, *args, **kwargs):
        out = _orig_cce(hidden_states, lm_head_weight, chunk_size, *args, **kwargs)
        # Returns (loss, correct, entropy_sum, n_valid_tensor); n_valid is the 4th element.
        captured["n_valid"] = int(out[3].item())
        captured["chunk_size"] = int(chunk_size)
        return out

    sft._chunked_cross_entropy_loss = _capturing_cce

    dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
    # Write trainer artifacts to a throwaway temp dir so the worker leaves no state in the repo checkout and
    # repeated runs can't collide. tempfile keeps this self-contained (no reliance on the launch cwd).
    tmp_out = tempfile.mkdtemp(prefix="chunked_nll_fsdp2_")
    args = SFTConfig(
        output_dir=tmp_out,
        loss_type="chunked_nll",
        # Pack as many of the tiny examples into the single measured step as possible, so n_valid is well
        # above the (shrunk) chunk size and the token-chunk count is large.
        per_device_train_batch_size=_PER_DEVICE_TRAIN_BATCH_SIZE,
        max_length=_MAX_LENGTH,
        max_steps=1,
        report_to="none",
        bf16=True,
    )
    trainer = _MeasuringSFTTrainer(model=MODEL_ID, args=args, train_dataset=dataset)

    vocab_size = trainer.model.config.vocab_size

    # Run the real training loop: it FSDP-wraps the model and moves it to GPU, then calls training_step
    # once (max_steps=1), which our subclass measures under CommDebugMode.
    trainer.train()

    comm_counts = trainer.comm_counts or {}
    all_gathers = _count_all_gathers(comm_counts)
    comm_total = sum(int(n) for n in comm_counts.values())

    n_valid = captured.get("n_valid", 0)
    chunk_size = captured.get("chunk_size", _TEST_CHUNK_SIZE)
    n_chunks = -(-n_valid // chunk_size) if n_valid else 0  # ceil(n_valid / chunk_size) — TOKEN chunks

    last = trainer.state.log_history[-1] if trainer.state.log_history else {}
    train_loss = last.get("train_loss")

    result = {
        "mode": mode,
        "vocab_size": int(vocab_size),
        "n_valid": int(n_valid),
        "chunk_size": int(chunk_size),
        "n_chunks_if_regressed": int(n_chunks),
        "all_gathers": int(all_gathers),
        "commdebug_total": int(comm_total),
        "loss_finite": train_loss is not None and math.isfinite(train_loss),
    }
    if trainer.accelerator.is_main_process:
        print(f"{RESULT_PREFIX} {json.dumps(result)}", flush=True)  # noqa: T201 - result channel for the launcher


if __name__ == "__main__":
    # Print the full traceback from this worker directly: when `accelerate launch` re-raises a child
    # failure, the parent only sees a truncated `CompletedProcess` repr, which hides the real error frame.
    # Surfacing it here puts the complete traceback in the worker's own stderr (and thus the CI log).
    import sys
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
