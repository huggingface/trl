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

import warnings
from dataclasses import dataclass, field
from typing import Any

from ...trainer.base_config import _BaseConfig


@dataclass
class DistillationConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`DistillationTrainer`].

    Extends [`~transformers.TrainingArguments`] with parameters specific to knowledge distillation. This config is
    independent of [`SFTConfig`] — all necessary fields are declared here.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument of the
            trainer is provided as a string.
        max_length (`int` or `None`, *optional*, defaults to `1024`):
            Maximum total sequence length (prompt + completion) for tokenization and truncation.

        > Parameters that control the distillation

        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling during generation and for computing the distillation loss. Higher values produce
            softer probability distributions.
        lmbda (`float`, *optional*, defaults to `1.0`):
            Probability of using on-policy (student-generated) data for each gradient accumulation slice. A value of
            `0.0` means fully off-policy (dataset completions only), `1.0` means fully on-policy.
        beta (`float`, *optional*, defaults to `1.0`):
            Interpolation coefficient for the Generalized Jensen-Shannon Divergence loss. When `0.0`, the loss is the
            forward KL divergence. When `1.0`, the loss is the reverse KL divergence. When `0.5`, it is the standard
            JSD.
        reverse_kl_top_1_mode (`str`, *optional*, defaults to `"sampled"`):
            Selection rule for the reverse-KL top-1 token when `beta > 0` and `loss_top_k == 1`. `"sampled"` uses the
            actual completion token in the batch. `"argmax"` uses the student's highest-probability token. This
            setting does not affect the forward-KL support, which always uses the teacher's top-1 token. Ignored when
            `beta == 0` or `loss_top_k != 1`.
        max_completion_length (`int`, *optional*, defaults to `512`):
            Maximum number of tokens to generate per completion during on-policy generation.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the student model during training.

        > Parameters that control the teacher model

        teacher_model_name_or_path (`str` or `None`, *optional*):
            Model name or path for the teacher model. Used when the teacher is loaded locally.
        teacher_model_revision (`str` or `None`, *optional*):
            Model revision of the teacher model (e.g., branch name, tag, or commit hash).
        teacher_model_init_kwargs (`dict[str, Any]` or `None`, *optional*):
            Keyword arguments passed to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        use_teacher_server (`bool`, *optional*, defaults to `False`):
            Whether to use an external vLLM teacher server instead of a local teacher model.
        teacher_model_server_url (`str` or `None`, *optional*):
            Base URL of a vLLM server hosting the teacher model (e.g., `"http://localhost:8000"`). When set, teacher
            logprobs are fetched from the server instead of running a local forward pass when `use_teacher_server=True`.
        loss_top_k (`int`, *optional*, defaults to `1`):
            Number of top tokens to use when computing the JSD/KL loss. Both student and teacher distributions are
            restricted to these K tokens and re-normalized before computing divergence. If 0, the full vocabulary
            is used. For local teachers, the general support rule is teacher top-k for forward KL, student top-k for
            reverse KL, and the union for mixed JSD. When `beta > 0` and `loss_top_k == 1`, the forward support still
            uses the teacher's top-1 token, while the reverse top-1 token is controlled by `reverse_kl_top_1_mode`.
            When `use_teacher_server=True`, the pure forward path (`beta=0`) requires this to be positive and uses the
            teacher's top-k logprobs for the forward term. When `beta > 0`, server-backed distillation requires
            `loss_top_k == 1` and only supports `"sampled"` reverse top-1 tokens.
        loss_add_tail (`bool`, *optional*, defaults to `True`):
            Whether to append a tail bucket that represents the remaining probability mass outside the selected top-k
            support when computing the loss.

        > Parameters that control on-policy generation

        num_generations (`int`, *optional*, defaults to `1`):
            Number of completions to generate per prompt during on-policy generation.
        generation_batch_size (`int` or `None`, *optional*):
            Number of unique prompts per worker per optimizer step. If `None`, computed from
            `(per_device_train_batch_size * gradient_accumulation_steps) // num_generations`.
        top_p (`float`, *optional*, defaults to `0.95`):
            Top-p (nucleus) sampling parameter for on-policy generation.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter for on-policy generation. `0` disables top-k filtering.

        > Parameters that control vLLM for student generation

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating on-policy completions from the student model.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode for student vLLM integration. Either `"server"` or `"colocate"`.
        vllm_server_base_url (`str` or `None`, *optional*):
            Base URL for the student vLLM server. If provided, `vllm_server_host` and `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the student vLLM server.
        vllm_server_port (`int`, *optional*, defaults to `8001`):
            Port of the student vLLM server.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port for the vLLM weight-update group (NCCL communicator).
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            GPU memory utilization for the colocated student vLLM engine.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine.
        vllm_max_model_length (`int` or `None`, *optional*):
            Maximum model sequence length for the colocated vLLM engine.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation backend for vLLM. Use `"vllm"` or `"transformers"`.
        vllm_structured_outputs_regex (`str` or `None`, *optional*):
            Regex pattern for vLLM structured outputs.
        vllm_sync_frequency (`int`, *optional*, defaults to `1`):
            Frequency (in training steps) to synchronize student model weights to the vLLM engine.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload student weights during the optimizer step.

        > Parameters that control logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `log_completions_steps` steps. If `rich` is
            installed, it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb`
            and/or `trackio`.
        log_completions_steps (`int`, *optional*, defaults to `100`):
            Number of steps between logging completions. Only used if `log_completions` is `True`.
        num_completions_to_print (`int` or `None`, *optional*):
            Number of completions to print. If `None`, all completions are logged.
    """

    _VALID_DICT_FIELDS = _BaseConfig._VALID_DICT_FIELDS + ["model_init_kwargs", "teacher_model_init_kwargs"]

    # Model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained`, used when the `model` argument "
            "of the trainer is provided as a string."
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={"help": "Maximum total sequence length (prompt + completion) for tokenization and truncation."},
    )

    # Overridden defaults
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )

    # Distillation core
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling and loss computation. Higher values produce softer distributions."
        },
    )
    lmbda: float = field(
        default=1.0,
        metadata={
            "help": "Probability of using on-policy (student-generated) data per gradient accumulation slice. "
            "0.0 = fully off-policy, 1.0 = fully on-policy."
        },
    )
    beta: float = field(
        default=1.0,
        metadata={
            "help": "Interpolation coefficient for the Generalized JSD loss. "
            "0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL."
        },
    )
    reverse_kl_top_1_mode: str = field(
        default="sampled",
        metadata={
            "help": "Reverse-KL top-1 token selection when beta > 0 and loss_top_k == 1. "
            "Use 'sampled' for the actual completion token or 'argmax' for the student's top-1 token. "
            "The forward-KL support always uses the teacher's top-1 token. Ignored when beta == 0 or loss_top_k != 1."
        },
    )
    max_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    max_prompt_length: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of tokens for the prompt. If None, auto-computed as "
            "max_length - max_completion_length. Prompts are truncated according to the "
            "tokenizer's truncation_side setting."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the student model during training."},
    )

    # Teacher model (local)
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Model name or path for the teacher model."},
    )
    teacher_model_revision: str | None = field(
        default=None,
        metadata={"help": "Model revision of the teacher model (e.g., branch name, tag, or commit hash)."},
    )
    teacher_model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `AutoModelForCausalLM.from_pretrained` when instantiating the teacher."
        },
    )

    # Teacher model (external vLLM server)
    use_teacher_server: bool = field(
        default=False,
        metadata={"help": "Whether to use an external vLLM teacher server instead of a local teacher model."},
    )
    teacher_model_server_url: str | None = field(
        default=None,
        metadata={
            "help": 'Base URL of a vLLM server hosting the teacher model (e.g., "http://localhost:8000"). '
            "Required when use_teacher_server=True."
        },
    )
    loss_top_k: int = field(
        default=1,
        metadata={
            "help": "Number of top tokens to use when computing the JSD/KL loss. "
            "Both student and teacher distributions are restricted to these K tokens "
            "(selected based on beta: teacher's top-k for forward KL, student's top-k for reverse KL, "
            "union of both for JSD) and re-normalized before computing divergence. "
            "If 0, the full vocabulary is used (slower but exact). "
            "When beta > 0 and loss_top_k == 1, the forward support still uses the teacher's top-1 token, "
            "while the reverse top-1 token is controlled by reverse_kl_top_1_mode. "
            "When use_teacher_server=True, beta=0 requires loss_top_k > 0 and uses the teacher's top-k "
            "logprobs for the forward term. When beta > 0, server-backed distillation requires loss_top_k == 1 "
            "and only supports 'sampled' reverse top-1 tokens."
        },
    )
    loss_add_tail: bool = field(
        default=True,
        metadata={
            "help": "Whether to append a tail bucket representing the remaining probability mass outside the selected top-k support."
        },
    )

    # On-policy generation
    num_generations: int = field(
        default=1,
        metadata={"help": "Number of completions to generate per prompt during on-policy generation."},
    )
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Number of unique prompts per worker per optimizer step. "
            "If None, computed from (per_device_train_batch_size * gradient_accumulation_steps) // num_generations."
        },
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p (nucleus) sampling parameter for on-policy generation."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling parameter for on-policy generation. 0 disables top-k filtering."},
    )

    # vLLM for student generation
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for generating on-policy completions from the student model."},
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={"help": 'Mode for student vLLM integration. Either "server" or "colocate".'},
    )
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={"help": "Base URL for the student vLLM server."},
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the student vLLM server."},
    )
    vllm_server_port: int = field(
        default=8001,
        metadata={"help": "Port of the student vLLM server."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Timeout for connecting to the student vLLM server."},
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={"help": "Port for the vLLM weight-update group (NCCL communicator)."},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={"help": "GPU memory utilization for the colocated student vLLM engine."},
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size for the colocated student vLLM engine."},
    )
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={"help": "Maximum model sequence length for the colocated vLLM engine."},
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={"help": 'Model implementation backend for vLLM. Use "vllm" or "transformers".'},
    )
    vllm_structured_outputs_regex: str | None = field(
        default=None,
        metadata={"help": "Regex pattern for vLLM structured outputs."},
    )
    vllm_sync_frequency: int = field(
        default=1,
        metadata={"help": "Frequency (in training steps) to synchronize student weights to the vLLM engine."},
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Enable vLLM sleep mode to offload student weights during the optimizer step."},
    )

    # W&B
    wandb_entity: str | None = field(
        default=None,
        metadata={"help": "The W&B entity to store runs under."},
    )
    wandb_project: str | None = field(
        default=None,
        metadata={"help": "The W&B project to store runs under."},
    )
    wandb_run_group: str | None = field(
        default=None,
        metadata={"help": "The W&B group to store runs under."},
    )

    # Logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `log_completions_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` and/or `trackio` logging is enabled, it logs it to `wandb` "
            "and/or `trackio`."
        },
    )
    log_completions_steps: int = field(
        default=100,
        metadata={"help": "Number of steps between logging completions."},
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print. If None, all completions are logged."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError(f"lmbda must be in [0.0, 1.0], got {self.lmbda}.")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError(f"beta must be in [0.0, 1.0], got {self.beta}.")
        if self.reverse_kl_top_1_mode not in {"sampled", "argmax"}:
            raise ValueError("reverse_kl_top_1_mode must be one of: 'sampled', 'argmax'")

        if self.max_length is not None and self.max_completion_length >= self.max_length:
            raise ValueError(
                f"max_completion_length ({self.max_completion_length}) must be smaller than "
                f"max_length ({self.max_length}) to leave room for the prompt."
            )

        if self.max_prompt_length is None and self.max_length is not None:
            self.max_prompt_length = self.max_length - self.max_completion_length

        if self.num_generations < 1:
            raise ValueError(f"num_generations must be at least 1, got {self.num_generations}.")

        local_sequence_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if self.generation_batch_size is None:
            self.generation_batch_size = local_sequence_batch_size // self.num_generations
        if self.generation_batch_size < 1:
            raise ValueError(f"generation_batch_size must be at least 1, got {self.generation_batch_size}.")
        if self.generation_batch_size * self.num_generations != local_sequence_batch_size:
            raise ValueError(
                "generation_batch_size * num_generations must equal per_device_train_batch_size * "
                f"gradient_accumulation_steps. Got {self.generation_batch_size} * {self.num_generations} != "
                f"{self.per_device_train_batch_size} * {self.gradient_accumulation_steps}."
            )

        if self.use_teacher_server and self.use_liger_kernel:
            raise ValueError(
                "use_liger_kernel=True is not supported with use_teacher_server=True because the Liger loss path "
                "requires a local teacher model."
            )
        if self.use_teacher_server and (
            self.teacher_model_server_url is None or not self.teacher_model_server_url.strip()
        ):
            raise ValueError("teacher_model_server_url must be set when use_teacher_server=True.")

        if self.use_teacher_server and self.beta == 0 and self.loss_top_k < 1:
            raise ValueError(
                f"loss_top_k must be positive when using use_teacher_server=True with beta=0 "
                f"(got loss_top_k={self.loss_top_k}). The pure forward server path only has access to the "
                f"teacher's top-k logprobs, so it cannot compute the exact full-vocabulary loss when loss_top_k=0."
            )
        if self.use_teacher_server and self.reverse_kl_top_1_mode == "argmax":
            raise ValueError(
                "reverse_kl_top_1_mode='argmax' is not supported with use_teacher_server=True because the server "
                "cannot provide teacher logprobs for arbitrary student-selected tokens."
            )
        if self.use_teacher_server and self.beta > 0 and self.loss_top_k != 1:
            raise ValueError(
                f"loss_top_k must be 1 when using use_teacher_server=True with beta>0 "
                f"(got loss_top_k={self.loss_top_k}). Mixed forward/reverse distillation with an external teacher "
                "is only implemented for top-1 support."
            )
        if self.reverse_kl_top_1_mode != "sampled" and (self.beta == 0 or self.loss_top_k != 1):
            warnings.warn(
                f"reverse_kl_top_1_mode='{self.reverse_kl_top_1_mode}' has no effect when beta={self.beta} "
                f"and loss_top_k={self.loss_top_k}. It only applies when beta > 0 and loss_top_k == 1.",
                UserWarning,
                stacklevel=2,
            )

        if self.num_generations > 1 and self.lmbda < 1.0:
            warnings.warn(
                f"num_generations={self.num_generations} with lmbda={self.lmbda} means off-policy batches include "
                f"{self.num_generations} copies of each sample. Consider lmbda=1.0 when num_generations > 1.",
                UserWarning,
                stacklevel=2,
            )
