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

from ...trainer.sft_config import SFTConfig


@dataclass
class GKDConfig(SFTConfig):
    """
    Configuration class for [`experimental.gkd.GKDTrainer`].

    This class includes only the parameters that are specific to GKD training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] and [`SFTConfig`] documentation.

    Args:
        temperature (`float`, *optional*, defaults to `0.9`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        lmbda (`float`, *optional*, defaults to `0.5`):
            Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy
            student-generated outputs).
        beta (`float`, *optional*, defaults to `0.5`):
            Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence loss. When
            beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL Divergence.
        max_new_tokens (`int`, *optional*, defaults to `128`):
            Maximum number of tokens to generate per completion.
        teacher_model_name_or_path (`str`, *optional*):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model being
            trained.
        teacher_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT on
            teacher-generated output).
        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions from the student model. Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"colocate"`):
            Mode for student vLLM integration. Either `"server"` (connect to a running TRL vLLM server) or `"colocate"`
            (run vLLM in the same process).
        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8001"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_port (`int`, *optional*, defaults to `8001`):
            Port of the vLLM server for the student model (if `vllm_mode="server"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server (if `vllm_mode="server"`).
        vllm_group_port (`int`, *optional*, defaults to `51216`):
            Port for the vLLM weight-update group (NCCL communicator). Unless the port is occupied, there is no need to
            change it.
        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            GPU memory utilization for the colocated student vLLM engine (if `vllm_mode="colocate"`). It is recommended
            to set this to a low value if the student and teacher models share the same GPU.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine (if `vllm_mode="colocate"`).
        vllm_max_model_length (`int`, *optional*):
            Maximum model sequence length for the colocated vLLM engine when `vllm_mode="colocate"`. Defaults to the
            model's maximum context length.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation backend to use in vLLM. Use `"vllm"` (default) or `"transformers"`.
        vllm_structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs for the student model.
        vllm_sync_frequency (`int`, *optional*, defaults to `1`):
            Frequency (in training steps) to synchronize student model weights to vLLM engine. Set to 1 to sync after
            every step.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU memory usage
            low, but waking the engine adds host–device transfer latency.
    """

    _VALID_DICT_FIELDS = SFTConfig._VALID_DICT_FIELDS + ["teacher_model_init_kwargs"]

    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    lmbda: float = field(
        default=0.5,
        metadata={
            "help": "Lambda parameter that controls the student data fraction (i.e., the proportion of on-policy "
            "student-generated outputs)."
        },
    )
    beta: float = field(
        default=0.5,
        metadata={
            "help": "Interpolation coefficient between `0.0` and `1.0` of the Generalized Jensen-Shannon Divergence "
            "loss. When beta is `0.0`, the loss is the KL divergence. When beta is `1.0`, the loss is the Inverse KL "
            "Divergence."
        },
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Model name or path of the teacher model. If `None`, the teacher model will be the same as the "
            "model being trained."
        },
    )
    teacher_model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    seq_kd: bool = field(
        default=False,
        metadata={
            "help": "Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised "
            "FT on teacher-generated output)."
        },
    )

    # vLLM parameters
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Whether to use vLLM for generating completions. Requires `vllm` to be installed."},
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": 'Mode for vLLM integration. Either "server" (connect to a running TRL vLLM server) or "colocate" (run vLLM in the same process).'
        },
    )
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": 'Base URL for the vLLM server (e.g., "http://localhost:8001"). If provided, vllm_server_host and vllm_server_port are ignored.'
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": 'Host of the vLLM server when `vllm_mode="server"`.'},
    )
    vllm_server_port: int = field(
        default=8001,
        metadata={"help": 'Port of the vLLM server when `vllm_mode="server"`.'},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": 'Timeout (in seconds) for connecting to the vLLM server when `vllm_mode="server"`.'},
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={"help": "Port for the vLLM weight-update group (NCCL communicator)."},
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": 'GPU memory utilization for the colocated vLLM engine when `vllm_mode="colocate"`. Lower values reduce contention when sharing a device with the student/teacher models.'
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": 'Tensor parallel size for the colocated vLLM engine when `vllm_mode="colocate"`.'},
    )
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={
            "help": 'Maximum model sequence length for the colocated vLLM engine when `vllm_mode="colocate"`. Defaults to the model\'s maximum context length.'
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={"help": 'Model implementation backend to use in vLLM. Use "vllm" (default) or "transformers".'},
    )
    vllm_structured_outputs_regex: str | None = field(
        default=None,
        metadata={"help": "Regex pattern used for vLLM structured outputs (optional)."},
    )
    vllm_sync_frequency: int = field(
        default=1,
        metadata={
            "help": "Frequency (in training steps) to synchronize model weights to the vLLM engine. Set to 1 to sync after every step."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable vLLM sleep mode to offload student weights/cache during the optimizer step. Keeps GPU "
            "memory usage low, but waking the engine adds host–device transfer latency."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # check lmbda and beta are in the range [0, 1]
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0].")
        if self.use_vllm and self.lmbda == 0.0 and not self.seq_kd:
            raise ValueError(
                "use_vllm=True has no effect when lmbda=0.0 and seq_kd=False, since no on-policy generation happens. "
                "Set lmbda > 0 or seq_kd=True, or disable use_vllm."
            )
        if self.vllm_mode not in {"colocate", "server"}:
            raise ValueError(f"vllm_mode must be 'colocate' or 'server', got {self.vllm_mode!r}.")
