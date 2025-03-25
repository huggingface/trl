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

from .vllm_base import BaseVLLMClient
from .vllm_client import VLLMClient
from .vllm_coloc_client import VLLMColocationClient

def get_vllm_client(args, accelerator, model) -> BaseVLLMClient:
    if args.vllm_colocation:
        print("\n\n\n\nColoc client !")
        return VLLMColocationClient(accelerator, args, model)
    else:
        print("\n\n\n\nOld client !")
        return VLLMClient(args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout)

