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

from ..self_distillation import SelfDistillationConfig


@dataclass
class SDPOConfig(SelfDistillationConfig):
    r"""
    Configuration class for the [`SDPOTrainer`].

    This class extends [`experimental.self_distillation.SelfDistillationConfig`] with the online teacher-construction
    parameters used by Self-Distillation Policy Optimization (SDPO).

    Parameters:
        > Parameters that control the SDPO loss

        sdpo_policy_loss_mode (`str`, *optional*, defaults to `"distillation_only"`):
            How SDPO combines the online policy loss and self-distillation loss. Supported: `distillation_only`,
            `hybrid`.
        distillation_alpha (`float`, *optional*, defaults to `1.0`):
            Divergence interpolation coefficient. Token-level SDPO requires the official reverse-KL setting
            `distillation_alpha=1.0`.
        distillation_topk (`int` or `None`, *optional*):
            Top-k approximation for logit-level SDPO. Requires `full_logit_distillation=True`.

        > Parameters that control the teacher

        teacher_regularization (`str`, *optional*, defaults to `"ema"`):
            Teacher update strategy. Supported: `ema`, `none`.
        teacher_update_rate (`float` or `None`, *optional*):
            EMA update rate used when `teacher_regularization="ema"`.
        ema_update_rate (`float`, *optional*, defaults to `0.05`):
            Deprecated alias for `teacher_update_rate`.

        > Parameters that control reprompting

        use_successful_as_teacher (`bool`, *optional*, defaults to `True`):
            Whether successful rollouts are turned into teacher demonstrations.
        success_reward_threshold (`float`, *optional*, defaults to `1.0`):
            Minimum reward for a rollout to count as successful.
        include_environment_feedback (`bool`, *optional*, defaults to `False`):
            Whether `privileged_context` is injected into teacher reprompts when available.
    """

    dont_reprompt_on_self_success: bool = field(
        default=True,
        metadata={"help": "Skip reprompting when model generates correct response."},
    )
    distillation_alpha: float = field(
        default=1.0,
        metadata={
            "help": "KL divergence direction for SDPO. Token-level SDPO requires reverse KL (`distillation_alpha=1.0`)."
        },
    )
    distillation_topk: int | None = field(
        default=None,
        metadata={"help": "Top-K approximation for logit-level SDPO. Requires `full_logit_distillation=True`."},
    )
    sdpo_policy_loss_mode: str = field(
        default="distillation_only",
        metadata={"help": "SDPO policy loss mode. Supported: `distillation_only`, `hybrid`."},
    )
    teacher_regularization: str = field(
        default="ema",
        metadata={"help": "Teacher regularization mode. Supported: `ema`, `none`."},
    )
    teacher_update_rate: float | None = field(
        default=None,
        metadata={"help": "Teacher update rate used for EMA teacher synchronization."},
    )
    ema_update_rate: float = field(
        default=0.05,
        metadata={"help": "Deprecated alias for `teacher_update_rate`."},
    )
    max_reprompt_len: int = field(
        default=10240,
        metadata={"help": "Maximum length for reprompting in self-distillation."},
    )
    use_successful_as_teacher: bool = field(
        default=True,
        metadata={"help": "Use successful rollouts as implicit feedback for self-distillation."},
    )
    success_reward_threshold: float = field(
        default=1.0,
        metadata={"help": "Minimum reward for a rollout to be considered a successful demonstration."},
    )
    reprompt_template: str = field(
        default="{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n",
        metadata={"help": "Template for reprompting the teacher with a successful demonstration."},
    )
    solution_template: str = field(
        default="\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
        metadata={"help": "Template for formatting the successful demonstration text."},
    )
    feedback_template: str = field(
        default="\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
        metadata={"help": "Template for formatting environment feedback for reprompting."},
    )
    include_environment_feedback: bool = field(
        default=False,
        metadata={"help": "Whether to include environment feedback in teacher reprompts when available."},
    )
    environment_feedback_only_without_solution: bool = field(
        default=False,
        metadata={"help": "Whether to use feedback only when no successful solution is available."},
    )
    remove_thinking_from_demonstration: bool = field(
        default=False,
        metadata={"help": "Whether to remove <think>...</think> blocks from the demonstration text."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.teacher_update_rate is None:
            self.teacher_update_rate = self.ema_update_rate

        if self.teacher_regularization not in {"ema", "none"}:
            raise ValueError("teacher_regularization must be one of: 'ema', 'none'")
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError("teacher_update_rate must be in [0, 1]")
        if self.sdpo_policy_loss_mode not in {"distillation_only", "hybrid"}:
            raise ValueError("sdpo_policy_loss_mode must be one of: 'distillation_only', 'hybrid'")
        if self.sdpo_policy_loss_mode == "distillation_only" and self.distillation_weight <= 0:
            raise ValueError("distillation_only mode requires `distillation_weight > 0`.")
        if self.max_reprompt_len <= 0:
            raise ValueError("max_reprompt_len must be positive")
        if not self.full_logit_distillation and self.distillation_alpha != 1.0:
            raise ValueError(
                "SDPO token-level distillation requires `distillation_alpha=1.0`. "
                "Set `full_logit_distillation=True` to use other divergence settings."
            )
        if self.distillation_topk is not None and not self.full_logit_distillation:
            raise ValueError("SDPO `distillation_topk` requires `full_logit_distillation=True`.")
