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

from dataclasses import dataclass, field

from ...trainer.grpo_config import GRPOConfig


@dataclass
class CGRPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`experimental.cgrpo.CGRPOTrainer`].

    [`experimental.cgrpo.CGRPOConfig`] inherits every parameter from [`GRPOConfig`]. `num_generations` is the width of
    each prompt group and defaults to `max(budget_grid)`; the number of completions actually generated per prompt is
    chosen adaptively along `budget_grid`.

    Parameters:
        budget_grid (`list[int]`, *optional*, defaults to `[2, 4, 8, 16, 32]`):
            Ascending sampling budgets. A prompt is sampled up to each budget in turn and stops at the first one whose
            conformal prediction set is a singleton.
        num_generations (`int`, *optional*):
            Group width. Defaults to `max(budget_grid)`, and must equal it if set.
        delta (`float`, *optional*):
            Miscoverage level of the conformal thresholds. If `None`, it is set at every calibration from the policy's
            solve rate within the largest budget: `(1 - solve_rate) + delta_margin`, clipped to `[0.05, 0.95]`.
        delta_margin (`float`, *optional*, defaults to `0.05`):
            Margin added to the miss rate when `delta` is `None`.
        split_delta (`bool`, *optional*, defaults to `False`):
            When `delta` is `None`, choose `delta` on the first half of the calibration set and the thresholds on the
            second half. Split-conformal exactness requires `delta` to be independent of the scores that produce the
            thresholds; the default reuses the same data for both.
        score (`str`, *optional*, defaults to `"aps"`):
            Nonconformity score. `"aps"`: randomized APS over extracted answers, for tasks with a comparable final
            answer. `"first_success"`: `j*/k`, the position of the first passing completion, for execution-verified
            tasks. `"pass_rate"`: `1 - n_pass/k`; discrete, so coverage is conservative rather than exact.
        recalibrate_every (`int`, *optional*, defaults to `50`):
            Refit the thresholds with the current policy every this many optimizer steps. `0` calibrates once, before
            the first step.
        num_calibration_samples (`int`, *optional*):
            Number of calibration examples to use. Defaults to the whole calibration dataset. Each calibration
            generates `num_calibration_samples * max(budget_grid)` completions.
        calibration_batch_size (`int`, *optional*):
            Prompts per generation call during calibration. Each prompt is sampled `max(budget_grid)` times, so the
            default, `per_device_train_batch_size // max(budget_grid)` (at least 1), makes a calibration call the same
            number of rows as a training generation call.
        per_device_train_batch_size (`int`, *optional*, defaults to `32`):
            Rows per device per step. Must be a multiple of `num_generations`; the default fits the default grid.
        calibration_answer_column (`str`, *optional*, defaults to `"answer"`):
            Column of the calibration dataset holding the reference answer, used when `score="aps"`.
    """

    budget_grid: list[int] = field(
        default_factory=lambda: [2, 4, 8, 16, 32],
        metadata={"help": "Ascending sampling budgets; sampling stops at the first singleton conformal set."},
    )
    num_generations: int | None = field(
        default=None,
        metadata={"help": "Group width. Defaults to max(budget_grid) and must equal it if set."},
    )
    delta: float | None = field(
        default=None,
        metadata={"help": "Miscoverage level. If None, chosen from the policy's solve rate at each calibration."},
    )
    delta_margin: float = field(
        default=0.05,
        metadata={"help": "Margin added to the miss rate when delta is None."},
    )
    split_delta: bool = field(
        default=False,
        metadata={"help": "Choose delta and the thresholds on disjoint halves of the calibration set."},
    )
    score: str = field(
        default="aps",
        metadata={"help": "Nonconformity score.", "choices": ["aps", "first_success", "pass_rate"]},
    )
    recalibrate_every: int = field(
        default=50,
        metadata={"help": "Refit thresholds every this many optimizer steps; 0 calibrates once."},
    )
    num_calibration_samples: int | None = field(
        default=None,
        metadata={"help": "Number of calibration examples to use. Defaults to the whole calibration dataset."},
    )
    calibration_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Prompts per generation call during calibration. Defaults to "
            "per_device_train_batch_size // max(budget_grid), so a calibration call has as many rows as a training "
            "generation call."
        },
    )
    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Rows per device per step; a multiple of num_generations (max(budget_grid))."},
    )
    calibration_answer_column: str = field(
        default="answer",
        metadata={"help": "Calibration dataset column with the reference answer (score='aps')."},
    )

    def __post_init__(self):
        if (
            not self.budget_grid
            or any(k < 1 for k in self.budget_grid)
            or len(set(self.budget_grid)) != len(self.budget_grid)
        ):
            raise ValueError(f"budget_grid must be distinct positive integers, got {self.budget_grid}.")
        self.budget_grid = sorted(self.budget_grid)
        k_max = self.budget_grid[-1]
        if self.num_generations is None:
            self.num_generations = k_max
        elif self.num_generations != k_max:
            raise ValueError(
                f"num_generations ({self.num_generations}) must equal max(budget_grid) ({k_max}): it is the group "
                "width, and prompts that stop early are padded up to it."
            )
        if self.score not in ("aps", "first_success", "pass_rate"):
            raise ValueError(f"score must be 'aps', 'first_success' or 'pass_rate', got {self.score!r}.")
        if self.delta is not None and not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.delta}.")
        if self.recalibrate_every < 0:
            raise ValueError(f"recalibrate_every must be >= 0, got {self.recalibrate_every}.")
        super().__post_init__()
