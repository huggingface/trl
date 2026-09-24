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

"""Reward functions that log different keys on different ranks must not desynchronize the flush (issue #7310)."""

import argparse
import os
import threading

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=["grpo", "rloo"], required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # A regression shows up as a hang in a collective. Bound it so the test fails instead of stalling the lane.
    watchdog = threading.Timer(600, os._exit, args=(124,))
    watchdog.daemon = True
    watchdog.start()

    rank = int(os.environ["RANK"])
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    def reward_func(completions, log_metric, log_extra, **kwargs):
        if rank == 0:
            # Logged on rank 0 only: the other rank must still join the collectives.
            log_metric("format_accuracy", 1.0)
            log_extra("parser_score", [1.0] * len(completions))
            # Logged with unequal counts: rank 0 logs one value per completion, rank 1 a single value below, so the
            # mean must weight the logged values (3 x 1.0 and 1 x 0.0 give 0.75), not the ranks (0.5).
            for _ in completions:
                log_metric("weighted", 1.0)
        else:
            # Logged on rank 1 only, with a different name than rank 0's metric.
            log_metric("proof_score", 0.5)
            log_metric("weighted", 0.0)
        return [float(len(c)) for c in completions]

    if args.trainer == "grpo":
        config_class, trainer_class = GRPOConfig, GRPOTrainer
    else:
        config_class, trainer_class = RLOOConfig, RLOOTrainer
    training_args = config_class(
        output_dir=args.output_dir,
        per_device_train_batch_size=3,
        num_generations=3,
        max_completion_length=8,
        max_steps=2,
        logging_steps=1,
        log_completions=True,  # the main process builds the completions table, which raises on a ragged column
        report_to="none",
        save_strategy="no",
    )
    trainer = trainer_class(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    assert trainer.accelerator.num_processes == 2
    trainer.train()

    train_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert len(train_logs) == 2
    for entry in train_logs:
        assert entry["format_accuracy"] == 1.0
        assert entry["proof_score"] == 0.5
        assert entry["weighted"] == 0.75

    extra = trainer._logs["extra"]["parser_score"]
    assert len(extra) == len(trainer._logs["prompt"])
    assert set(extra) == {None, 1.0}


if __name__ == "__main__":
    main()
