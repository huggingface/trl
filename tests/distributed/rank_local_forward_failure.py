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

"""Child process for the distributed forward-failure checks. Launched by `test_distributed.py`.

The DPO, KTO, and SFT loss methods run metric gathers after their forwards. A rank that raised on its own batch would
leave its peers inside the first of those collectives with nothing to complete it, so the run would stall until the
collective timed out rather than fail with the message. The trainers therefore agree across ranks before raising, and
the agreement has to sit upstream of every one of those collectives, not merely upstream of the last.

A single-process test cannot check any of that: with one rank the gather is the identity and the sole rank is also the
failing one, so the raise happens whether the ranks agree or not. This file fails the forward on the last rank only and
asserts, on every rank, that the run ended in an exception rather than a stall: the failing rank raises what it saw,
and every other rank raises the peer-failure error. Two regressions are caught. Raising rank-locally leaves the peers
in the metric gathers, and placing the agreement below any of them pairs one rank's metric gather with another rank's
failure flag, so the peers complete the wrong collective and then block on the next one.

The filename is deliberately not `test_*.py`: pytest must not collect it in the parent process.
"""

import argparse
import os
import sys
import tempfile
import threading

import torch
from datasets import load_dataset

from trl import DPOConfig, DPOTrainer, KTOConfig, KTOTrainer, SFTConfig, SFTTrainer


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


FAILURE_TYPES = {
    "dpo_forward": ValueError,
    "dpo_liger_loss": RuntimeError,
    "kto_forward": torch.OutOfMemoryError,
    "kto_reference_forward": RuntimeError,
    "kto_liger_forward": ValueError,
    "kto_liger_loss": torch.OutOfMemoryError,
    "sft_forward": RuntimeError,
}


def _raise_failure(case):
    raise FAILURE_TYPES[case](f"injected {case} failure on the last rank")


def _register_failure_hook(module, case):
    def fail_the_forward(module, args, kwargs):
        _raise_failure(case)

    return module.register_forward_pre_hook(fail_the_forward, with_kwargs=True)


def _is_failing_rank(trainer):
    return trainer.accelerator.process_index == trainer.accelerator.num_processes - 1


class FailingForwardDPOTrainer(DPOTrainer):
    def _compute_loss(self, model, inputs, return_outputs=False):
        if not _is_failing_rank(self) or self.failure_case != "dpo_forward":
            return super()._compute_loss(model, inputs, return_outputs)

        # Register on the wrapped module so DDP completes its buffer broadcast before the hook raises.
        handle = _register_failure_hook(model.module, self.failure_case)
        try:
            return super()._compute_loss(model, inputs, return_outputs)
        finally:
            handle.remove()

    def _compute_loss_liger(self, model, inputs, return_outputs=False):
        if not _is_failing_rank(self) or self.failure_case != "dpo_liger_loss":
            return super()._compute_loss_liger(model, inputs, return_outputs)

        def fail_liger_loss(*args, **kwargs):
            _raise_failure(self.failure_case)

        liger_loss = self.liger_loss
        self.liger_loss = fail_liger_loss
        try:
            return super()._compute_loss_liger(model, inputs, return_outputs)
        finally:
            self.liger_loss = liger_loss


class FailingForwardKTOTrainer(KTOTrainer):
    def _compute_loss(self, model, inputs, return_outputs=False):
        if not _is_failing_rank(self):
            return super()._compute_loss(model, inputs, return_outputs)

        if self.failure_case == "kto_forward":
            module = model.module
        elif self.failure_case == "kto_reference_forward":
            module = self.ref_model
        else:
            return super()._compute_loss(model, inputs, return_outputs)

        handle = _register_failure_hook(module, self.failure_case)
        try:
            return super()._compute_loss(model, inputs, return_outputs)
        finally:
            handle.remove()

    def _compute_loss_liger(self, model, inputs, return_outputs=False):
        if not _is_failing_rank(self):
            return super()._compute_loss_liger(model, inputs, return_outputs)

        if self.failure_case == "kto_liger_forward":
            handle = _register_failure_hook(model.base_model, self.failure_case)
            try:
                return super()._compute_loss_liger(model, inputs, return_outputs)
            finally:
                handle.remove()
        if self.failure_case == "kto_liger_loss":

            def fail_liger_loss(*args, **kwargs):
                _raise_failure(self.failure_case)

            liger_loss = self.liger_loss
            self.liger_loss = fail_liger_loss
            try:
                return super()._compute_loss_liger(model, inputs, return_outputs)
            finally:
                self.liger_loss = liger_loss
        return super()._compute_loss_liger(model, inputs, return_outputs)


class FailingForwardSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _is_failing_rank(self):
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Register on the wrapped module so DDP completes its buffer broadcast before the hook raises.
        handle = _register_failure_hook(model.module, self.failure_case)
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=FAILURE_TYPES)
    args = parser.parse_args()

    if args.case.startswith("dpo_"):
        config_class = DPOConfig
        trainer_class = FailingForwardDPOTrainer
        dataset_config = "standard_preference"
    elif args.case.startswith("kto_"):
        config_class = KTOConfig
        trainer_class = FailingForwardKTOTrainer
        dataset_config = "standard_unpaired_preference"
    else:
        config_class = SFTConfig
        trainer_class = FailingForwardSFTTrainer
        dataset_config = "standard_language_modeling"

    dataset = load_dataset("trl-internal-testing/zen", dataset_config, split="train")

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = config_class(
            output_dir=tmp_dir,
            per_device_train_batch_size=2,  # reduce the batch size to reduce memory usage
            max_steps=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_liger_kernel="liger" in args.case,
        )
        trainer = trainer_class(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.failure_case = args.case

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
        except Exception as exc:
            watchdog.cancel()
            if is_failing_rank:
                assert isinstance(exc, FAILURE_TYPES[args.case]), (
                    f"the failing rank {rank} raised {type(exc).__name__}, expected "
                    f"{FAILURE_TYPES[args.case].__name__}: {exc}"
                )
                assert f"injected {args.case} failure" in str(exc), (
                    f"rank {rank} raised an unexpected {type(exc).__name__}: {exc}"
                )
            else:
                assert isinstance(exc, RuntimeError), (
                    f"rank {rank} raised {type(exc).__name__}, not RuntimeError: {exc}"
                )
                assert "failed on another rank" in str(exc), (
                    f"rank {rank} raised a RuntimeError that is not the peer-failure error: {exc}. A collective "
                    f"error here means the ranks disagreed about which collectives to run."
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
