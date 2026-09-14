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
Two-rank CPU worker for the multi-teacher `DistillationTrainer` collective/normalization check.

Launched by `tests/distributed/test_distillation_trainer_multi_teacher.py`, twice through
`python -m torch.distributed.run`: once with two ranks (gloo, `--mode multi`) and once with a single rank
(`--mode reference`), with the same *global* batch in both. Completions are replaced by a deterministic function of
the prompt tokens (`FixedCompletionTrainer`), so both runs train on exactly the same (prompt, completion) pairs and
the comparison isolates the cross-rank reduction from sampling. Rank 0 writes a JSON summary and the final student
parameters to the `--out` path (and a sibling `-params.pt` file).

The two teacher checkpoints are built by every process, deterministically (a fixed seed and a fixed per-parameter
rescale, no sampling involved), so every rank ends up with bit-identical teachers without needing a shared
filesystem hand-off.
"""

import argparse
import json
import os
import tempfile

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DistillationConfig, DistillationTrainer


MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
# 4 rows per optimizer step, 2 steps: split 2 ranks x 1 row x 2 accumulation steps under the multi-process config,
# and 1 process x 2 rows x 2 accumulation steps in the single-process reference. See `main()`.
TRAIN_TEACHER_IDS = ["a", "a", "b", "b", "a", "a", "b", "b"]
PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The largest planet is",
    "Photosynthesis happens in",
    "The speed of light is",
    "Mount Everest is in",
    "The Pacific Ocean is",
    "An atom contains",
]


def _build_teacher(directory: str, scale: float) -> str:
    """Deterministically build one tiny teacher checkpoint, rescaled so its targets differ from the student's."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.mul_(scale)
    model.save_pretrained(directory)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(directory)
    return directory


def build_dataset(teacher_ids: list[str]) -> Dataset:
    return Dataset.from_dict(
        {"prompt": [PROMPTS[index % len(PROMPTS)] for index in range(len(teacher_ids))], "teacher_id": teacher_ids}
    )


class FixedCompletionTrainer(DistillationTrainer):
    """Replaces sampled completions with a deterministic function of the prompt, so both modes see equal tokens."""

    def _generate(self, prompts):
        prompt_ids, completion_ids, tool_mask, images, tool_images = super()._generate(prompts)
        fixed = []
        for ids in prompt_ids:
            offset = sum(int(token) for token in ids) % 997
            fixed.append([(offset + position * 7) % 100 + 1 for position in range(self.max_completion_length)])
        return prompt_ids, fixed, tool_mask, images, tool_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["multi", "reference"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as teacher_root:
        teacher_a = _build_teacher(os.path.join(teacher_root, "teacher-a"), scale=1.05)
        teacher_b = _build_teacher(os.path.join(teacher_root, "teacher-b"), scale=0.95)

        # Same *global* batch in both modes: 4 rows per optimizer step, split 2 ranks x 1 row x 2 accumulation
        # steps under the multi-process CPU config, and 1 process x 2 rows x 2 accumulation steps in the reference.
        per_device_train_batch_size = 1 if args.mode == "multi" else 2
        training_args = DistillationConfig(
            output_dir=args.output_dir,
            learning_rate=0.1,
            # Plain SGD, so the update is proportional to the gradient: AdamW normalizes by a near-zero second
            # moment here (the tiny student and teachers start almost identical), which would amplify the last-bit
            # gradient difference between two ranks and one process into a large parameter difference and tell us
            # nothing about the reduction itself.
            optim="sgd",
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=2,
            max_completion_length=4,
            max_steps=2,
            logging_steps=1,
            eval_strategy="no",
            shuffle_dataset=False,
            seed=11,
            report_to="none",
            use_cpu=True,
        )
        trainer = FixedCompletionTrainer(
            model=MODEL_ID,
            args=training_args,
            train_dataset=build_dataset(TRAIN_TEACHER_IDS),
            teacher_models={"a": teacher_a, "b": teacher_b},
        )
        accelerator = trainer.accelerator
        world_size = accelerator.num_processes
        if args.mode == "multi":
            # The whole point of this worker: a real two-rank run. Never let a single-process fallback be recorded
            # as distributed evidence.
            assert world_size == 2, f"expected world_size == 2 under accelerate launch, got {world_size}"
            assert torch.distributed.is_initialized(), "torch.distributed was not initialized"
            assert torch.distributed.get_backend() == "gloo", f"expected gloo, got {torch.distributed.get_backend()}"
        else:
            assert world_size == 1, f"the reference must be single-process, got {world_size}"

        trainer.train()

        step_logs = [entry for entry in trainer.state.log_history if "loss" in entry]
        teacher_metric_keys = sorted({key for entry in step_logs for key in entry if key.startswith("teacher_")})
        summary = {
            "mode": args.mode,
            "world_size": world_size,
            "optimizer_steps": trainer.state.global_step,
            "per_device_train_batch_size": per_device_train_batch_size,
            "teacher_ids": list(trainer.teacher_models),
            "train_losses": [entry["loss"] for entry in step_logs],
            "num_tokens": [entry["num_tokens"] for entry in step_logs],
            "teacher_metric_keys": teacher_metric_keys,
        }

        # Every rank writes to the shared JSON/params paths only from the main process, and every rank must reach
        # this point for the run to be a real distributed collective, not a rank that silently failed earlier.
        if accelerator.is_main_process:
            with open(args.out, "w") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            parameters = {name: param.detach().cpu() for name, param in trainer.model.named_parameters()}
            torch.save(parameters, os.path.splitext(args.out)[0] + "-params.pt")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    # `AutoTokenizer` is imported so the worker fails fast if the tiny model is not cached, rather than inside the
    # trainer where the traceback is harder to read from an `accelerate launch` log.
    AutoTokenizer.from_pretrained(MODEL_ID)
    main()
