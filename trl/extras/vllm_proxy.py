# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from abc import ABC, abstractmethod
from typing import Optional
import torch

class BaseVLLMClient(ABC):

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:
        pass

    @abstractmethod
    def update_named_param(self, name: str, weights: torch.Tensor):
        pass

    @abstractmethod
    def reset_prefix_cache(self):
        pass

def get_vllm_client(args, accelerator, model) -> BaseVLLMClient:
    from .vllm_colocation_client import VLLMColocationClient
    from .vllm_client import VLLMClient

    if args.vllm_colocation:
        return VLLMColocationClient(accelerator, args, model)
    else:
        return VLLMClient(args.vllm_server_host, args.vllm_server_port, connection_timeout=120.0)

