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

from ..distillation.distillation_config import DistillationConfig


@dataclass
class ServerDistillationConfig(DistillationConfig):
    r"""
    Configuration class for the [`ServerDistillationTrainer`].

    Extends [`DistillationConfig`] with the address of an external vLLM server that scores the student's completions.
    The teacher is never held locally: instead of a local forward pass, per-token teacher logprobs are fetched from the
    server, so only the teacher's top-k logprobs are available and the loss is restricted to a sparse support.

    Parameters:
        teacher_model_server_url (`str` or `None`, *optional*):
            Base URL of a vLLM server hosting the teacher model (e.g., `"http://localhost:8000"`). Required.
    """

    teacher_model_server_url: str | None = field(
        default=None,
        metadata={
            "help": 'Base URL of a vLLM server hosting the teacher model (e.g., "http://localhost:8000"). Required.'
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.use_liger_kernel:
            raise ValueError(
                "use_liger_kernel=True is not supported by ServerDistillationTrainer because the Liger loss path "
                "requires a local teacher model."
            )
        if self.teacher_model_server_url is None or not self.teacher_model_server_url.strip():
            raise ValueError("teacher_model_server_url must be set for ServerDistillationTrainer.")
        if self.beta == 0 and self.loss_top_k < 1:
            raise ValueError(
                f"loss_top_k must be positive with beta=0 (got loss_top_k={self.loss_top_k}). The pure forward "
                f"server path only has access to the teacher's top-k logprobs, so it cannot compute the exact "
                f"full-vocabulary loss when loss_top_k=0."
            )
        if self.reverse_kl_top_1_mode == "argmax":
            raise ValueError(
                "reverse_kl_top_1_mode='argmax' is not supported by ServerDistillationTrainer because the server "
                "cannot provide teacher logprobs for arbitrary student-selected tokens."
            )
        if self.beta > 0 and self.loss_top_k != 1:
            raise ValueError(
                f"loss_top_k must be 1 with beta>0 (got loss_top_k={self.loss_top_k}). Mixed forward/reverse "
                "distillation with an external teacher is only implemented for top-1 support."
            )
