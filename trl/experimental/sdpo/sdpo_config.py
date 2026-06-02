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
from typing import Any, Literal

from transformers import TrainingArguments

from ...trainer.base_config import _BaseConfig


@dataclass
class SDPOConfig(_BaseConfig):
    r"""
    Configuration class for the [`SDPOTrainer`].

    Parameters:
        > Parameters that control the online policy objective

        beta (`float`, *optional*, defaults to `0.0`):
            Reference-model KL coefficient for online policy optimization.
        epsilon (`float`, *optional*, defaults to `0.2`):
            Lower clipping coefficient for GRPO-style policy loss.
        epsilon_high (`float` or `None`, *optional*):
            Upper clipping coefficient. Defaults to `epsilon` when unset.
        importance_sampling_level (`str`, *optional*, defaults to `"token"`):
            Importance-sampling granularity. Supported: `token`, `sequence`.
        reward_weights (`list[float]` or `None`, *optional*):
            Optional weights for multiple reward functions.
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Reward normalization mode. Supported: `group`, `batch`, `none`.

        > Parameters that control the SDPO loss

        sdpo_policy_loss_mode (`str`, *optional*, defaults to `"distillation_only"`):
            How SDPO combines the online policy loss and self-distillation loss. Supported: `distillation_only`,
            `hybrid`.
        distillation_alpha (`float`, *optional*, defaults to `1.0`):
            Divergence interpolation coefficient. Sampled-token SDPO requires the official reverse-KL setting
            `distillation_alpha=1.0`.
        distillation_mode (`Literal["sampled_token", "full_logits", "topk_logits"]`, *optional*, defaults to `"sampled_token"`):
            Distillation objective mode. `"sampled_token"` is the default SDPO mode and requires
            `distillation_alpha=1.0`.
        distillation_topk (`int` or `None`, *optional*):
            Top-k approximation for logit-level SDPO. Must be set when `distillation_mode="topk_logits"` and left unset
            otherwise.

        > Parameters that control the teacher

        teacher_model_kind (`str`, *optional*, defaults to `"ema"`):
            Semantic teacher choice. `base` uses the initial student, `live` uses the current student, and `ema` uses
            an exponentially averaged teacher.
        teacher_update_rate (`float`, *optional*, defaults to `0.05`):
            EMA update rate used when `teacher_model_kind="ema"`.
        teacher_sync_steps (`int`, *optional*, defaults to `1`):
            Number of optimizer steps between teacher EMA updates.

        > Parameters that control reprompting

        use_successful_as_teacher (`bool`, *optional*, defaults to `True`):
            Whether successful rollouts are turned into teacher demonstrations.
        success_reward_threshold (`float`, *optional*, defaults to `1.0`):
            Minimum reward for a rollout to count as successful.
        include_environment_feedback (`bool`, *optional*, defaults to `False`):
            Whether `privileged_context` is injected into teacher reprompts when available.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Keyword arguments for model initialization when `model` is passed as a string."},
    )
    disable_dropout: bool = field(
        default=False,
        metadata={"help": "Whether to disable dropout in the student model."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether to drop dataset columns unused by the trainer."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum prompt length. Longer prompts are truncated from the left."},
    )
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of sampled generations per prompt."},
    )
    num_generations_eval: int | None = field(
        default=None,
        metadata={"help": "Number of sampled generations per prompt during evaluation."},
    )
    max_completion_length: int | None = field(
        default=256,
        metadata={"help": "Maximum generated completion length."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={"help": "Whether to gather ZeRO-3 weights for generation."},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    generation_batch_size: int | None = field(
        default=None,
        metadata={"help": "Global batch size used for generation. Mutually exclusive with `steps_per_generation`."},
    )
    steps_per_generation: int | None = field(
        default=None,
        metadata={
            "help": "Number of optimizer steps that reuse one generated batch. Mutually exclusive with `generation_batch_size`."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling parameter."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling parameter. `0` disables top-k filtering."},
    )
    min_p: float | None = field(
        default=None,
        metadata={"help": "Minimum token probability for sampling."},
    )
    generation_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Extra generation kwargs passed to `GenerationConfig`."},
    )
    chat_template_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty used during generation."},
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Cache implementation used by transformers generation."},
    )
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for generation."},
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={"help": "vLLM mode: 'colocate' (shared GPU) or 'server' (separate vLLM server)."},
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={"help": "Model implementation for vLLM: 'vllm' or 'transformers'."},
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Whether to enable sleep mode for colocated vLLM engine."},
    )
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server. If provided, vllm_server_host and vllm_server_port are ignored."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server (server mode only)."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server (server mode only)."},
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={"help": "Port for the weight update group (server mode only)."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Timeout in seconds to wait for the vLLM server."},
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for colocated vLLM."},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={"help": "GPU memory utilization ratio for colocated vLLM."},
    )
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={"help": "Model context length for vLLM. Inferred from model config if not set."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of optimization iterations per generated batch."},
    )
    loss_type: str = field(
        default="dapo",
        metadata={"help": "Policy loss aggregation. Supported: `grpo`, `bnpo`, `dr_grpo`, `dapo`."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Whether to exclude truncated completions from the loss."},
    )
    dont_reprompt_on_self_success: bool = field(
        default=True,
        metadata={"help": "Skip reprompting when model generates correct response."},
    )
    beta: float = field(
        default=0.0,
        metadata={"help": "Reference-model KL coefficient for online policy optimization."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Lower clipping coefficient for GRPO-style policy loss."},
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={"help": "Upper clipping coefficient. Defaults to `epsilon` when unset."},
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={"help": "Importance-sampling granularity. Supported: `token`, `sequence`."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={"help": "Optional weights for multiple reward functions."},
    )
    scale_rewards: str | bool = field(
        default="group",
        metadata={"help": "Reward normalization mode. Supported: `group`, `batch`, `none`."},
    )
    distillation_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Divergence interpolation coefficient. Sampled-token SDPO requires the official reverse-KL setting "
            "`distillation_alpha=1.0`."
        },
    )
    distillation_mode: Literal["sampled_token", "full_logits", "topk_logits"] = field(
        default="sampled_token",
        metadata={
            "help": "Distillation objective mode. `sampled_token` is the default SDPO mode and requires "
            "`distillation_alpha=1.0`."
        },
    )
    distillation_topk: int | None = field(
        default=None,
        metadata={
            "help": "Top-k approximation for logit-level SDPO. Must be set when `distillation_mode=topk_logits` and left "
            "unset otherwise."
        },
    )
    distillation_is_clip: float | None = field(
        default=2.0,
        metadata={
            "help": "Clipping coefficient for importance sampling in self-distillation. `None` disables clipping."
        },
    )
    distillation_add_tail: bool = field(
        default=False,
        metadata={"help": "Whether to add a tail bucket for non-top-k probability mass."},
    )
    distillation_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the self-distillation loss term."},
    )
    sdpo_policy_loss_mode: str = field(
        default="distillation_only",
        metadata={"help": "SDPO policy loss mode. Supported: `distillation_only`, `hybrid`."},
    )
    teacher_model_kind: str = field(
        default="ema",
        metadata={
            "help": "Semantic teacher choice. `base` uses the initial student, `live` uses the current student, "
            "and `ema` uses an exponentially averaged teacher."
        },
    )
    teacher_update_rate: float = field(
        default=0.05,
        metadata={"help": "Teacher update rate used for EMA teacher synchronization."},
    )
    teacher_sync_steps: int = field(
        default=1,
        metadata={"help": "How often to synchronize the EMA teacher model."},
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
    diagnostics_warning_interval: int = field(
        default=10,
        metadata={
            "help": "Emit repeated trainer diagnostics every N consecutive degenerate steps. Set to 0 to disable."
        },
    )
    diagnostics_flat_tolerance: float = field(
        default=1e-8,
        metadata={
            "help": "Tolerance used to decide whether reward variance or reprompt activity is effectively zero."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        num_processes = self.world_size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            global_batch_size = self.per_device_train_batch_size * num_processes
            if self.generation_batch_size % global_batch_size != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size ({global_batch_size})."
                )
            self.steps_per_generation = self.generation_batch_size // global_batch_size
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError("'generation_batch_size' and 'steps_per_generation' can not both be configured")

        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations ({self.num_generations})."
            )

        if self.do_eval and self.eval_strategy != "no":
            num_generations_eval = self.num_generations_eval or self.num_generations
            if (self.per_device_eval_batch_size * num_processes) % num_generations_eval != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations_eval})."
                )

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon
