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
from typing import Any

from transformers import TrainingArguments

from ...trainer.base_config import _BaseConfig


@dataclass
class SelfDistillationConfig(_BaseConfig):
    r"""
    Shared configuration for experimental self-distillation trainers.

    This class contains only the arguments that are specific to the shared self-distillation stack. For the full set of
    generic training arguments, refer to [`~transformers.TrainingArguments`] via
    [`trl.trainer.base_config._BaseConfig`].

    Parameters:
        > Parameters that control generation and rollout reuse

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments used when the `model` argument is passed as a string.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum prompt length. Longer prompts are truncated from the left.
        num_generations (`int`, *optional*, defaults to `8`):
            Number of sampled generations per prompt.
        generation_batch_size (`int` or `None`, *optional*):
            Global batch size used for generation. Mutually exclusive with `steps_per_generation`.
        steps_per_generation (`int` or `None`, *optional*):
            Number of optimizer steps that reuse one generated batch. Mutually exclusive with `generation_batch_size`.

        > Parameters that control the online policy objective

        beta (`float`, *optional*, defaults to `0.0`):
            Reference-model KL coefficient for online policy optimization.
        loss_type (`str`, *optional*, defaults to `"dapo"`):
            Policy-loss aggregation mode. Supported: `grpo`, `bnpo`, `dr_grpo`, `dapo`.
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Reward normalization mode. Supported: `group`, `batch`, `none`.

        > Parameters that control self-distillation

        distillation_alpha (`float`, *optional*, defaults to `0.5`):
            Divergence interpolation coefficient using the official SDPO/SDFT convention: `0.0=forward KL`, `0.5=JSD`,
            `1.0=reverse KL`.
        distillation_topk (`int` or `None`, *optional*, defaults to `100`):
            Number of top tokens to keep for top-k distillation. If `None`, all logits are used.
        full_logit_distillation (`bool`, *optional*, defaults to `False`):
            Whether to use full-logit distillation instead of token-level distillation.
        distillation_is_clip (`float` or `None`, *optional*, defaults to `2.0`):
            Importance-sampling clip used by the official SDPO-style correction. `None` disables clipping.
        distillation_weight (`float`, *optional*, defaults to `1.0`):
            Weight applied to the self-distillation loss term.

        > Parameters that control diagnostics

        diagnostics_warning_interval (`int`, *optional*, defaults to `10`):
            Emit repeated trainer diagnostics every N consecutive degenerate steps. Set to `0` to disable.
        diagnostics_flat_tolerance (`float`, *optional*, defaults to `1e-8`):
            Tolerance used to decide whether reward variance or reprompt activity is effectively zero.
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
        metadata={"help": "Number of optimizer steps that reuse one generated batch."},
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
    use_transformers_paged: bool = field(
        default=False,
        metadata={"help": "Reserved for paged generation support."},
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
    beta: float = field(
        default=0.0,
        metadata={"help": "Reference-model KL coefficient for online policy optimization."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of optimization iterations per generated batch."},
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
    loss_type: str = field(
        default="dapo",
        metadata={"help": "Policy loss aggregation. Supported: `grpo`, `bnpo`, `dr_grpo`, `dapo`."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Whether to exclude truncated completions from the loss."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to synchronize the reference model with the student model."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={"help": "EMA mix coefficient used when syncing the reference model."},
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={"help": "How often to synchronize the reference model."},
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={"help": "Reserved for entropy-based token filtering."},
    )
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
        metadata={"help": "Whether to use full-logit distillation instead of token-level distillation."},
    )
    distillation_is_clip: float | None = field(
        default=2.0,
        metadata={"help": "Clipping coefficient for importance sampling in self-distillation."},
    )
    distillation_add_tail: bool = field(
        default=False,
        metadata={"help": "Whether to add a tail bucket for non-top-k probability mass."},
    )
    distillation_weight: float = field(
        default=1.0,
        metadata={"help": "Weight applied to the self-distillation loss term."},
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

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)
        if self.scale_rewards not in ["group", "batch", "none"]:
            raise ValueError("scale_rewards must be one of: 'group', 'batch', 'none'")

        if self.importance_sampling_level not in ["token", "sequence"]:
            raise ValueError("importance_sampling_level must be either 'token' or 'sequence'")
        if self.loss_type not in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            raise ValueError("loss_type must be one of: 'grpo', 'bnpo', 'dr_grpo', 'dapo'")
        if self.num_generations < 1:
            raise ValueError("num_generations must be at least 1")
        if not 0.0 <= self.distillation_alpha <= 1.0:
            raise ValueError("distillation_alpha must be in [0, 1]")
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError("distillation_topk must be positive when provided")
        if self.distillation_is_clip is not None and self.distillation_is_clip <= 0:
            raise ValueError("distillation_is_clip must be positive when provided")
        if self.distillation_weight < 0:
            raise ValueError("distillation_weight must be non-negative")
        if self.diagnostics_warning_interval < 0:
            raise ValueError("diagnostics_warning_interval must be non-negative")
        if self.diagnostics_flat_tolerance < 0:
            raise ValueError("diagnostics_flat_tolerance must be non-negative")

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

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon
