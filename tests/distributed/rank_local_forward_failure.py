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

"""Child process for the distributed forward-failure check. Launched by `test_distributed.py`.

`DPOTrainer` translates a forward failure into an actionable message, and the loss method that forward sits in goes on
to run sixteen metric gathers before ending in the `frac_nonfinite_loss` gather. A rank that raised on its own batch
would leave its peers inside the first of those collectives with nothing to complete it, so the run would stall until
the collective timed out rather than fail with the message. The trainer therefore agrees across ranks before raising,
and the agreement has to sit upstream of every one of those collectives, not merely upstream of the last.

A single-process test cannot check any of that: with one rank the gather is the identity and the sole rank is also the
failing one, so the raise happens whether the ranks agree or not. This file fails the forward on the last rank only and
asserts, on every rank, that the run ended in an exception rather than a stall: the failing rank raises what it saw,
and every other rank raises the peer-failure error. Two regressions are caught. Raising rank-locally leaves the peers
in the metric gathers, and placing the agreement below any of them pairs one rank's metric gather with another rank's
failure flag, so the peers complete the wrong collective and then block on the next one.

The filename is deliberately not `test_*.py`: pytest must not collect it in the parent process.
"""

import os
import sys
import tempfile
import threading

from datasets import load_dataset

from trl import DPOConfig, DPOTrainer


# A regression here stalls rather than fails, and a stalled child would sit until the CI job's own timeout. Fail the
# process instead, well inside any collective timeout. A signal handler cannot do this: the stalled rank sits inside a
# C-level collective and never returns to Python bytecode, so `SIGALRM` would be delivered but never handled. A daemon
# thread can, because the collective releases the GIL while it waits. The deadline is absolute, so a healthy run
# that outlasts it is killed too; the healthy two-rank CPU run measured under 90 seconds, so 300 leaves room.
WATCHDOG_SECONDS = 300


def _watchdog():
    sys.stderr.write(
        f"no rank raised within {WATCHDOG_SECONDS}s. A rank that fails its own forward must make the other ranks "
        f"raise too; if it does not, they wait in the loss path's gather and the run stalls here.\n"
    )
    sys.stderr.flush()
    os._exit(1)


def main() -> int:
    dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

    class FailingForwardDPOTrainer(DPOTrainer):
        # Fail on the last rank only. The real trigger is a truncated image placeholder, which needs a VLM and a
        # `max_length` short enough to cut it; any `ValueError` out of the forward reaches the same handler, so
        # raising one directly keeps this check to the rank agreement it exists to cover.
        #
        # The failure has to come out of the forward itself, not out of this override. The agreement lives inside
        # `_compute_loss`, so an override that raises before delegating would skip it and the check would pass on a
        # trainer that has no agreement at all. A pre-hook makes the model call raise instead, which is where a real
        # truncated placeholder raises too.
        def _compute_loss(self, model, inputs, return_outputs=False):
            if self.accelerator.process_index != self.accelerator.num_processes - 1:
                return super()._compute_loss(model, inputs, return_outputs)

            def fail_the_forward(module, args, kwargs):
                raise ValueError("injected forward failure on the last rank")

            handle = model.register_forward_pre_hook(fail_the_forward, with_kwargs=True)
            try:
                return super()._compute_loss(model, inputs, return_outputs)
            finally:
                handle.remove()

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = DPOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_steps=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )
        trainer = FailingForwardDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        rank = trainer.accelerator.process_index
        world_size = trainer.accelerator.num_processes
        # With one rank the failing rank is the only rank, so every assertion below passes whether or not the ranks
        # agree. Refuse to run that way rather than report a pass that checked nothing.
        assert world_size >= 2, f"this check needs at least two ranks, got {world_size}; it proves nothing on one"
        is_failing_rank = rank == world_size - 1

        watchdog = threading.Timer(WATCHDOG_SECONDS, _watchdog)
        watchdog.daemon = True
        watchdog.start()
        try:
            trainer.train()
        except ValueError as exc:
            watchdog.cancel()
            assert is_failing_rank, f"rank {rank} raised the injected failure it never saw: {exc}"
            assert "injected forward failure" in str(exc), f"rank {rank} raised an unexpected ValueError: {exc}"
        except RuntimeError as exc:
            watchdog.cancel()
            assert not is_failing_rank, f"the failing rank {rank} lost its own error and reported a peer's: {exc}"
            assert "failed on another rank" in str(exc), (
                f"rank {rank} raised a RuntimeError that is not the peer-failure error: {exc}. A collective error "
                f"here means the ranks disagreed about which collectives to run."
            )
        else:
            watchdog.cancel()
            raise AssertionError(
                f"rank {rank} finished training even though the forward failed on the last rank. The failure has to "
                f"reach every rank, otherwise only the failing rank stops and the rest train on."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
