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

from trl.trainer.grpo_config import GRPOConfig


@dataclass
class SDPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`SDPOTrainer`].

    This class extends [`GRPOConfig`] with additional parameters specific to Self-Distillation Policy Optimization
    (SDPO). SDPO augments on-policy optimization with self-distillation from the model's own high-reward trajectories.

    SDPO converts tokenized feedback into a dense learning signal without any external teacher or explicit reward
    model. SDPO treats the current model conditioned on feedback as a self-teacher and distills its feedback-informed
    next-token predictions back into the policy.

    Parameters:
        distillation_alpha (`float`, *optional*, defaults to `0.5`):
            Controls the KL divergence direction in self-distillation loss.
            - 0.0: Forward KL (teacher -> student)
            - 0.5: Jensen-Shannon divergence (recommended by SDPO paper)
            - 1.0: Reverse KL (student -> teacher)
        distillation_topk (`int` or `None`, *optional*, defaults to `100`):
            Number of top tokens to consider for top-k distillation. If `None`, all tokens are considered. When
            `full_logit_distillation` is False, this parameter is used to compute top-k log probabilities.
        full_logit_distillation (`bool`, *optional*, defaults to `False`):
            Whether to use full logit distillation instead of token-level distillation. When True, distills from the
            full logit distribution. When False, distills only from top-k tokens.
        distillation_is_clip (`float`, *optional*, defaults to `2.0`):
            Clipping coefficient for importance sampling in self-distillation loss. Values > 0 apply clipping to
            stabilize training. Recommended value is 2.0.
        distillation_add_tail (`bool`, *optional*, defaults to `False`):
            Whether to add tail log-probability to top-k distillation. When True, includes the probability mass of
            non-top-k tokens as a separate "tail" token.
        dont_reprompt_on_self_success (`bool`, *optional*, defaults to `True`):
            Whether to skip reprompting when the model generates a correct response on its own. When True, the model
            uses its own successful response as a demonstration without additional prompting. When False, the model is
            always reprompted even on successful attempts.
        ema_update_rate (`float`, *optional*, defaults to `0.05`):
            EMA update rate for the teacher model. The teacher model is updated as: teacher = ema_update_rate * student
            + (1 - ema_update_rate) * teacher. A higher value makes the teacher follow the student more closely.
        max_reprompt_len (`int`, *optional*, defaults to `10240`):
            Maximum length for reprompting when using self-distillation. This limits the length of the feedback +
            reprompt sequence to prevent excessive memory usage.
        distillation_weight (`float`, *optional*, defaults to `1.0`):
            Weight for the self-distillation loss term. The total loss is: total_loss = grpo_loss + distillation_weight
            * distillation_loss.
        use_successful_as_teacher (`bool`, *optional*, defaults to `True`):
            Whether to use successful rollouts as implicit feedback for self-distillation. When True, high-reward
            rollouts are used as teacher demonstrations. When False, only explicit feedback is used for
            self-distillation.
    """

    # Self-distillation specific parameters
    distillation_alpha: float = field(
        default=0.5,
        metadata={"help": "KL divergence direction: 0.0=forward KL, 0.5=JSD, 1.0=reverse KL."},
    )
    distillation_topk: int | None = field(
        default=100,
        metadata={"help": "Number of top tokens for top-k distillation. If None, uses all tokens."},
    )
    full_logit_distillation: bool = field(
        default=False,
        metadata={"help": "Whether to use full logit distillation instead of token-level."},
    )
    distillation_is_clip: float = field(
        default=2.0,
        metadata={"help": "Clipping coefficient for importance sampling in self-distillation."},
    )
    distillation_add_tail: bool = field(
        default=False,
        metadata={"help": "Whether to add tail log-prob to top-k distillation."},
    )
    dont_reprompt_on_self_success: bool = field(
        default=True,
        metadata={"help": "Skip reprompting when model generates correct response."},
    )
    ema_update_rate: float = field(
        default=0.05,
        metadata={"help": "EMA update rate for teacher model."},
    )
    max_reprompt_len: int = field(
        default=10240,
        metadata={"help": "Maximum length for reprompting in self-distillation."},
    )
    distillation_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for self-distillation loss term."},
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
        default="{prompt}{solution}\n\nCorrectly solve the original question.\n",
        metadata={"help": "Template for reprompting the teacher with a successful demonstration."},
    )
    solution_template: str = field(
        default="\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
        metadata={"help": "Template for formatting the successful demonstration text."},
    )
    remove_thinking_from_demonstration: bool = field(
        default=False,
        metadata={"help": "Whether to remove <think>...</think> blocks from the demonstration text."},
    )
