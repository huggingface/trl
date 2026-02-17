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

import importlib
import os
import warnings
from contextlib import contextmanager
from itertools import chain
from types import ModuleType
from typing import Any

from packaging.version import Version
from transformers.utils.import_utils import _is_package_available


LIGER_KERNEL_MIN_VERSION = "0.7.0"


def is_deepspeed_available() -> bool:
    return _is_package_available("deepspeed")


def is_fastapi_available() -> bool:
    return _is_package_available("fastapi")


def is_jmespath_available() -> bool:
    return _is_package_available("jmespath")


def is_joblib_available() -> bool:
    return _is_package_available("joblib")


def is_liger_kernel_available(min_version: str = LIGER_KERNEL_MIN_VERSION) -> bool:
    _liger_kernel_available, _liger_kernel_version = _is_package_available("liger_kernel", return_version=True)
    return _liger_kernel_available and Version(_liger_kernel_version) >= Version(min_version)


def is_llm_blender_available() -> bool:
    return _is_package_available("llm_blender")


def is_math_verify_available() -> bool:
    return _is_package_available("math_verify")


def is_mergekit_available() -> bool:
    return _is_package_available("mergekit")


def is_pydantic_available() -> bool:
    return _is_package_available("pydantic")


def is_requests_available() -> bool:
    return _is_package_available("requests")


def is_unsloth_available() -> bool:
    return _is_package_available("unsloth")


def is_uvicorn_available() -> bool:
    return _is_package_available("uvicorn")


def is_vllm_available() -> bool:
    _vllm_available, _vllm_version = _is_package_available("vllm", return_version=True)
    if _vllm_available:
        if not (Version("0.10.2") <= Version(_vllm_version) <= Version("0.12.0")):
            warnings.warn(
                "TRL currently supports vLLM versions: 0.10.2, 0.11.0, 0.11.1, 0.11.2, 0.12.0. You have version "
                f"{_vllm_version} installed. We recommend installing a supported version to avoid compatibility "
                "issues.",
                stacklevel=2,
            )
    return _vllm_available


def is_vllm_ascend_available() -> bool:
    return _is_package_available("vllm_ascend")


def is_weave_available() -> bool:
    return _is_package_available("weave")


class TRLExperimentalWarning(UserWarning):
    """Warning for using the 'trl.experimental' submodule."""

    pass


@contextmanager
def suppress_warning(category):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=category)
        yield


def suppress_experimental_warning():
    return suppress_warning(TRLExperimentalWarning)


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
