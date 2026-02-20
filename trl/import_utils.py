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
import warnings
from contextlib import contextmanager

from packaging.version import Version


LIGER_KERNEL_MIN_VERSION = "0.7.0"
PACKAGE_DISTRIBUTION_MAPPING = importlib.metadata.packages_distributions()


# From transformers: https://github.com/huggingface/transformers/blob/556312cd45a5e619c41b0f8adf680eab0d334324/src/transformers/utils/import_utils.py#L48-L77
def _is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Check if `pkg_name` exist, and optionally try to get its version"""
    spec = importlib.util.find_spec(pkg_name)
    package_exists = spec is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            # importlib.metadata works with the distribution package, which may be different from the import
            # name (e.g. `PIL` is the import name, but `pillow` is the distribution name)
            distributions = PACKAGE_DISTRIBUTION_MAPPING[pkg_name]
            # Per PEP 503, underscores and hyphens are equivalent in package names.
            # Prefer the distribution that matches the (normalized) package name.
            normalized_pkg_name = pkg_name.replace("_", "-")
            if normalized_pkg_name in distributions:
                distribution_name = normalized_pkg_name
            elif pkg_name in distributions:
                distribution_name = pkg_name
            else:
                distribution_name = distributions[0]
            package_version = importlib.metadata.version(distribution_name)
        except (importlib.metadata.PackageNotFoundError, KeyError):
            # If we cannot find the metadata (because of editable install for example), try to import directly.
            # Note that this branch will almost never be run, so we do not import packages for nothing here
            package = importlib.import_module(pkg_name)
            package_version = getattr(package, "__version__", "N/A")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


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
