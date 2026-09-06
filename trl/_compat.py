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

"""
Compatibility shims for third-party dependencies.

This module contains temporary patches to handle version incompatibilities between TRL's dependencies.

Each patch should be removed when minimum version requirements eliminate the need.
"""

import warnings

from packaging.version import Version

from .import_utils import _is_package_available


def _is_package_version_below(package_name: str, version_threshold: str) -> bool:
    """
    Check if installed package version is below the given threshold.

    Args:
        package_name (str): Package name.
        version_threshold (str): Maximum version threshold.

    Returns:
        - True if package is installed and version < version_threshold.
        - False if package is not installed or version >= version_threshold.
    """
    try:
        is_available, version = _is_package_available(package_name, return_version=True)
        return is_available and Version(version) < Version(version_threshold)
    except Exception as e:
        warnings.warn(
            f"Failed to check {package_name} version against {version_threshold}: {e}. "
            f"Compatibility patch may not be applied.",
            stacklevel=2,
        )
        return False


def _is_package_version_at_least(package_name: str, version_threshold: str) -> bool:
    """
    Check if installed package version is at least the given threshold.

    Args:
        package_name (str): Package name.
        version_threshold (str): Minimum version threshold.

    Returns:
        - True if package is installed and version >= version_threshold.
        - False if package is not installed or version < version_threshold.
    """
    try:
        is_available, version = _is_package_available(package_name, return_version=True)
        return is_available and Version(version) >= Version(version_threshold)
    except Exception as e:
        warnings.warn(
            f"Failed to check {package_name} version against {version_threshold}: {e}. "
            f"Compatibility patch may not be applied.",
            stacklevel=2,
        )
        return False


def _patch_vllm_logging() -> None:
    """Set vLLM logging level to ERROR by default to reduce noise."""
    if _is_package_available("vllm"):
        import os

        os.environ["VLLM_LOGGING_LEVEL"] = os.getenv("VLLM_LOGGING_LEVEL", "ERROR")


def _patch_transformers_hybrid_cache() -> None:
    """
    Fix HybridCache import for transformers v5 compatibility.

    - Issue: peft import HybridCache from transformers.cache_utils
    - HybridCache removed in https://github.com/huggingface/transformers/pull/43168 (transformers>=5.0.0)
    - Fixed in peft: https://github.com/huggingface/peft/pull/2735 (released in v0.18.0)
    - This can be removed when TRL requires peft>=0.18.0
    """
    if _is_package_version_at_least("transformers", "5.0.0") and _is_package_version_below("peft", "0.18.0"):
        try:
            import transformers.cache_utils
            from transformers.utils.import_utils import _LazyModule

            Cache = transformers.cache_utils.Cache

            # Patch for liger_kernel: Add HybridCache as an alias for Cache in the cache_utils module
            transformers.cache_utils.HybridCache = Cache

            # Patch for peft: Patch _LazyModule.__init__ to add HybridCache to transformers' lazy loading structures
            _original_lazy_module_init = _LazyModule.__init__

            def _patched_lazy_module_init(self, name, *args, **kwargs):
                _original_lazy_module_init(self, name, *args, **kwargs)
                if name == "transformers":
                    # Update _LazyModule's internal structures
                    if hasattr(self, "_import_structure") and "cache_utils" in self._import_structure:
                        if "HybridCache" not in self._import_structure["cache_utils"]:
                            self._import_structure["cache_utils"].append("HybridCache")

                    if hasattr(self, "_class_to_module"):
                        self._class_to_module["HybridCache"] = "cache_utils"

                    if hasattr(self, "__all__") and "HybridCache" not in self.__all__:
                        self.__all__.append("HybridCache")

                    self.HybridCache = Cache

            _LazyModule.__init__ = _patched_lazy_module_init

        except Exception as e:
            warnings.warn(f"Failed to patch transformers HybridCache compatibility: {e}", stacklevel=2)


def _patch_transformers_parallelism_config() -> None:
    """
    Fix ParallelismConfig for transformers compatibility.

    Ensure that ``transformers.training_args`` always defines the symbol `ParallelismConfig` so that Python's
    `typing.get_type_hints` can resolve annotations on `transformers.TrainingArguments` without raising a `NameError`.

    This is needed when running with ``accelerate<1.10.1``, where the module ``accelerate.parallelism_config`` did not
    exist and therefore the type alias is not imported by Transformers.

    See upstream fix PR in transformers#40818.

    - Issue: transformers imports ParallelismConfig only if accelerate>=1.10.1 and raises NameError if
      accelerate<1.10.1
    - Fixed in transformers: https://github.com/huggingface/transformers/pull/40818 (released in v4.57.0)
    - This can be removed when TRL requires transformers>=4.57.0 or accelerate>=1.10.1
    """
    if _is_package_version_below("transformers", "4.57.0") and _is_package_version_below("accelerate", "1.10.1"):
        try:
            from typing import Any

            import transformers.training_args

            if not hasattr(transformers.training_args, "ParallelismConfig"):
                transformers.training_args.ParallelismConfig = Any
        except Exception as e:
            warnings.warn(f"Failed to patch transformers ParallelismConfig compatibility: {e}", stacklevel=2)


# Apply vLLM patches
_patch_vllm_logging()

# Apply transformers patches
_patch_transformers_hybrid_cache()
_patch_transformers_parallelism_config()  # before creating HfArgumentParser
