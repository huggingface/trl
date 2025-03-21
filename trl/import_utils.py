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

import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any

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


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))


class OptionalDependencyNotAvailable(BaseException):
    """Internally used error class for signalling an optional dependency was not found."""
