# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from typing import Any, Optional

from .sft_config import SFTConfig


@dataclass
class GKDConfig(SFTConfig):
    """
    Configuration class for [`GKDTrainer`].

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
        teacher_model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Model name or path of the teacher model. If `None`, the teacher model will be the same as the model
            being trained.
        teacher_model_init_kwargs (`dict[str, Any]]` or `None`, *optional*, defaults to `None`):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        seq_kd (`bool`, *optional*, defaults to `False`):
            Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised FT
            on teacher-generated output).
        student_use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions from the student model. Requires `vllm` to be installed.
        student_vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode for student vLLM integration. Either `"server"` (connect to a running TRL vLLM server) or
            `"colocate"` (run vLLM in the same process).
        student_vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server for the student model (if `student_vllm_mode="server"`).
        student_vllm_server_port (`int`, *optional*, defaults to `8001`):
            Port of the vLLM server for the student model (if `student_vllm_mode="server"`).
        student_vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Timeout for connecting to the student vLLM server (if `student_vllm_mode="server"`).
        student_vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            GPU memory utilization for the colocated student vLLM engine (if `student_vllm_mode="colocate"`).
            It is recommended to set this to a low value if the student and teacher models share the same GPU.
        student_vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Tensor parallel size for the colocated student vLLM engine (if `student_vllm_mode="colocate"`).
        student_vllm_guided_decoding_regex (`str` or `None`, *optional*, defaults to `None`):
            Regex for vLLM guided decoding for the student model.
        student_vllm_sync_frequency (`int`, *optional*, defaults to `1`):
            Frequency (in training steps) to synchronize student model weights to vLLM engine. Set to 1 to sync after every step.
    """

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
    teacher_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model name or path of the teacher model. If `None`, the teacher model will be the same as the "
            "model being trained."
        },
    )
    teacher_model_init_kwargs: Optional[dict[str, Any]] = field(
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
    
    # VLLM parameters for student model
    student_use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions from the student model. Requires `vllm` to be installed."
        },
    )
    student_vllm_mode: str = field(
        default="server",
        metadata={
            "help": 'Mode for student vLLM integration. Either "server" (connect to a running TRL vLLM server) or "colocate" (run vLLM in the same process).'
        },
    )
    student_vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": 'Host of the vLLM server for the student model (if `student_vllm_mode="server"`).'},
    )
    student_vllm_server_port: int = field(
        default=8001,
        metadata={"help": 'Port of the vLLM server for the student model (if `student_vllm_mode="server"`).'},
    )
    student_vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": 'Timeout for connecting to the student vLLM server (if `student_vllm_mode="server"`).'},
    )
    student_vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": 'GPU memory utilization for the colocated student vLLM engine (if `student_vllm_mode="colocate"`). It is recommended to set this to a low value if the student and teacher models share the same GPU.'
        },
    )
    student_vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": 'Tensor parallel size for the colocated student vLLM engine (if `student_vllm_mode="colocate"`).'
        },
    )
    student_vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding for the student model."},
    )
    student_vllm_sync_frequency: int = field(
        default=1,
        metadata={
            "help": "Frequency (in training steps) to synchronize student model weights to vLLM engine. Set to 1 to sync after every step."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # check lmbda and beta are in the range [0, 1]
        if self.lmbda < 0.0 or self.lmbda > 1.0:
            raise ValueError("lmbda must be in the range [0.0, 1.0].")
        if self.beta < 0.0 or self.beta > 1.0:
            raise ValueError("beta must be in the range [0.0, 1.0].")
