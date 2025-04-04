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


from transformers.utils.import_utils import _is_package_available


# Use same as transformers.utils.import_utils
_deepspeed_available = _is_package_available("deepspeed")
_diffusers_available = _is_package_available("diffusers")
_fastapi_available = _is_package_available("fastapi")
_llm_blender_available = _is_package_available("llm_blender")
_mergekit_available = _is_package_available("mergekit")
_pydantic_available = _is_package_available("pydantic")
_requests_available = _is_package_available("requests")
_rich_available = _is_package_available("rich")
_unsloth_available = _is_package_available("unsloth")
_uvicorn_available = _is_package_available("uvicorn")
_vllm_available = _is_package_available("vllm")
_joblib_available = _is_package_available("joblib")


def is_deepspeed_available() -> bool:
    return _deepspeed_available


def is_diffusers_available() -> bool:
    return _diffusers_available


def is_fastapi_available() -> bool:
    return _fastapi_available


def is_llm_blender_available() -> bool:
    return _llm_blender_available


def is_mergekit_available() -> bool:
    return _mergekit_available


def is_pydantic_available() -> bool:
    return _pydantic_available


def is_requests_available() -> bool:
    return _requests_available


def is_rich_available() -> bool:
    return _rich_available


def is_unsloth_available() -> bool:
    return _unsloth_available


def is_uvicorn_available() -> bool:
    return _uvicorn_available


def is_vllm_available() -> bool:
    return _vllm_available


def is_joblib_available() -> bool:
    return _joblib_available
