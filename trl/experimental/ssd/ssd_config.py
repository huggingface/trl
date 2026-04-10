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
class SSDConfig(_BaseConfig):
    r"""
    Configuration class for [`SSDTrainer`].

    Implements Simple Self-Distillation (SSD) from [*Embarrassingly Simple Self-Distillation Improves Code
    Generation*](https://huggingface.co/papers/2604.01193). SSD samples completions from the model at a training-time
    temperature and truncation configuration, then fine-tunes on those raw, unverified samples with standard
    cross-entropy loss.

    The `temperature`, `top_k`, and `top_p` parameters control the training-time sampling configuration (T_train,
    rho_train in the paper). The evaluation-time configuration (T_eval, rho_eval) is set independently at inference
    time.

    Parameters:
        > Parameters that control generation and rollout reuse

        model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments used when the `model` argument is passed as a string.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum prompt length. Longer prompts are truncated from the left.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum generated completion length.
        generation_batch_size (`int` or `None`, *optional*):
            Global batch size used for generation. Mutually exclusive with `steps_per_generation`.
        steps_per_generation (`int` or `None`, *optional*):
            Number of optimizer steps that reuse one generated batch. Mutually exclusive with `generation_batch_size`.

        > Parameters that control sampling

        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature (T_train in the paper).
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter. `0` disables top-k filtering.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p (nucleus) sampling parameter.
        min_p (`float` or `None`, *optional*):
            Minimum token probability for sampling.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Repetition penalty used during generation.
        generation_kwargs (`dict[str, Any]` or `None`, *optional*):
            Extra generation kwargs passed to `GenerationConfig`.

        > Parameters that control vLLM generation

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generation instead of the training model.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            vLLM mode: `"colocate"` (shared GPU) or `"server"` (separate vLLM server).
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation for vLLM: `"vllm"`, `"transformers"`, or `"auto"`.
        vllm_server_base_url (`str` or `None`, *optional*):
            Base URL for the vLLM server. If provided, `vllm_server_host` and `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server (server mode only).
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server (server mode only).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout in seconds to wait for the vLLM server.
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port for the weight update group (server mode only).
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for colocated vLLM.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            GPU memory utilization ratio for colocated vLLM.
        vllm_max_model_length (`int` or `None`, *optional*):
            Model context length for vLLM. Inferred from model config if not set.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for colocated vLLM engine.

        > Parameters that control training behavior

        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model during training.
        filter_empty (`bool`, *optional*, defaults to `True`):
            Whether to filter out empty or single-line stub completions from the generated data.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of optimization iterations per generated batch.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            Whether to gather ZeRO-3 weights for generation.
        cache_implementation (`str` or `None`, *optional*):
            Cache implementation used by transformers generation.
        chat_template_kwargs (`dict[str, Any]` or `None`, *optional*):
            Extra kwargs forwarded to chat template application.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    model_init_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Keyword arguments for model initialization when `model` is passed as a string."},
    )
    max_prompt_length: int | None = field(
        default=512,
        metadata={"help": "Maximum prompt length. Longer prompts are truncated from the left."},
    )
    max_completion_length: int | None = field(
        default=256,
        metadata={"help": "Maximum generated completion length."},
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
        metadata={"help": "Sampling temperature (T_train in the paper)."},
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling parameter. `0` disables top-k filtering."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p (nucleus) sampling parameter."},
    )
    min_p: float | None = field(
        default=None,
        metadata={"help": "Minimum token probability for sampling."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty used during generation."},
    )
    generation_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Extra generation kwargs passed to `GenerationConfig`."},
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Cache implementation used by transformers generation."},
    )
    chat_template_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application."},
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
        metadata={"help": "Model implementation for vLLM: 'vllm', 'transformers', or 'auto'."},
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
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Timeout in seconds to wait for the vLLM server."},
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={"help": "Port for the weight update group (server mode only)."},
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
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={"help": "Whether to enable sleep mode for colocated vLLM engine."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model during training."},
    )
    filter_empty: bool = field(
        default=True,
        metadata={"help": "Whether to filter out empty or single-line stub completions."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of optimization iterations per generated batch."},
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={"help": "Whether to gather ZeRO-3 weights for generation."},
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
