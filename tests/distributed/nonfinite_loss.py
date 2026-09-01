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

"""Child process for the distributed `frac_nonfinite_loss` check. Launched by `test_distributed.py`.

The guard reports `frac_nonfinite_loss` from a value it gathers across ranks, so that a non-finite loss on one rank is
visible on all of them. A single-process test cannot see that: `gather` is the identity there, so removing it entirely
leaves every single-process test green. This file poisons the loss on the last rank only and asserts, on every rank,
that the fraction each one logs accounts for the whole world rather than just itself.

The filename is deliberately not `test_*.py`: pytest must not collect it in the parent process, where there is one rank
and the assertion below would be false.
"""

import sys
import tempfile

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer


def main() -> int:
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    class NonFiniteLossGRPOTrainer(GRPOTrainer):
        # Poison the last rank only, so the gathered fraction is `1 / world_size` rather than 1.0 and the test can
        # tell a world-wide rate from a rank-local flag. Adding keeps the autograd graph, so `backward` runs on the
        # non-finite value exactly as it would for a real one.
        def _compute_loss(self, model, inputs):
            loss = super()._compute_loss(model, inputs)
            if self.accelerator.process_index == self.accelerator.num_processes - 1:
                return loss + float("nan")
            return loss

    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = GRPOConfig(
            output_dir=tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            # One optimizer step, one microbatch, one logging window, so the logged value is the gathered
            # fraction itself rather than that fraction averaged over a window of several microbatches.
            max_steps=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        )
        trainer = NonFiniteLossGRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        world_size = trainer.accelerator.num_processes
        # Every rank must report the same world-wide fraction: one poisoned rank out of `world_size`. Derived
        # rather than written as a literal, so the check stays correct if the launcher's rank count changes.
        expected = 1 / world_size
        actual = trainer.state.log_history[0]["frac_nonfinite_loss"]
        # The metric is a float32 mean widened to a Python float, so it equals `1 / world_size` exactly only when
        # the world size is a power of two: at three ranks it is 0.3333333432674408, not 0.3333333333333333. Compare
        # with a tolerance instead, one far tighter than the gap this test exists to detect. Without the gather a
        # rank reports 1.0 or 0.0, so the smaller of the two gaps is `1 / world_size`: 0.5 at two ranks, narrowing
        # as the world grows but staying orders of magnitude above the tolerance, which therefore cannot hide the
        # regression.
        assert abs(actual - expected) < 1e-6, (
            f"rank {trainer.accelerator.process_index} logged frac_nonfinite_loss={actual}, expected "
            f"{expected}. Without the gather a rank sees only its own loss, so the poisoned rank reports 1.0 "
            f"and every other rank reports 0.0."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
