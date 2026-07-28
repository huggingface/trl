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
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to allow loading models and tokenizers that ship custom Python code from the Hub. Forwarded to
            [`~transformers.AutoModelForCausalLM.from_pretrained`] and
            [`~transformers.AutoTokenizer.from_pretrained`], for both the student and teacher.

        > Parameters that control the distillation

        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling during generation and for computing the distillation loss. Higher values produce
            softer probability distributions.
        beta (`float`, *optional*, defaults to `1.0`):
            Interpolation coefficient for the Generalized Jensen-Shannon Divergence loss. When `0.0`, the loss is the
            forward KL divergence. When `1.0`, the loss is the reverse KL divergence. When `0.5`, it is the standard
            JSD.
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
        > Parameters that control on-policy generation

        num_generations (`int`, *optional*, defaults to `1`):
            Number of completions to generate per prompt during on-policy generation.
        generation_batch_size (`int` or `None`, *optional*):
            Number of unique prompts per worker per optimizer step. If `None`, computed from
            `(per_device_train_batch_size * gradient_accumulation_steps) // num_generations`.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p (nucleus) sampling parameter for on-policy generation.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter for on-policy generation. `0` disables top-k filtering.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.
        pad_to_multiple_of (`int`, *optional*):
            If set, the prompts ids and completions ids will be padded to a multiple of this value.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.

        > Parameters that control vLLM for student generation

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating on-policy completions from the student model.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode for student vLLM integration. Either `"server"` or `"colocate"`.
        vllm_server_base_url (`str` or `None`, *optional*):
            Base URL for the student vLLM server. If provided, `vllm_server_host` and `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the student vLLM server.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
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
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow loading models and tokenizers that ship custom Python code from the Hub. "
            "Forwarded to `AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`, for both the "
            "student and teacher."
        },
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
    beta: float = field(
        default=1.0,
        metadata={
            "help": "Interpolation coefficient for the Generalized JSD loss. "
            "0.0 = forward KL, 0.5 = JSD, 1.0 = reverse KL."
        },
    )
    max_completion_length: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens to generate per completion."},
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
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling parameter for on-policy generation."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling parameter for on-policy generation. 0 disables top-k filtering."},
    )
    min_p: float | None = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the prompts ids and completions ids will be padded to a multiple of this value."},
    )
    shuffle_dataset: bool | None = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
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
        default=8000,
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

        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError(f"beta must be in [0.0, 1.0], got {self.beta}.")

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
