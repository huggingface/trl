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
from transformers.utils.import_utils import _is_package_available


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


def _patch_vllm_disabled_tqdm() -> None:
    """
    Fix DisabledTqdm class in vLLM.

    - Bug introduced in https://github.com/vllm-project/vllm/pull/52
    - Fixed in https://github.com/vllm-project/vllm/pull/28471 (released in v0.11.1)
    - Since TRL currently supports vLLM v0.10.2-0.12.0, we patch it here
    - This can be removed when TRL requires vLLM>=0.11.1
    """
    if _is_package_version_below("vllm", "0.11.1"):
        try:
            import vllm.model_executor.model_loader.weight_utils
            from tqdm import tqdm

            class DisabledTqdm(tqdm):
                def __init__(self, *args, **kwargs):
                    kwargs["disable"] = True
                    super().__init__(*args, **kwargs)

            vllm.model_executor.model_loader.weight_utils.DisabledTqdm = DisabledTqdm
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to patch vLLM DisabledTqdm: {e}", stacklevel=2)


def _patch_vllm_cached_tokenizer() -> None:
    """
    Fix get_cached_tokenizer for transformers v5 compatibility.

    - Issue: vLLM's get_cached_tokenizer accesses all_special_tokens_extended
    - Removed in transformers: https://github.com/huggingface/transformers/pull/40936 (transformers>=5.0.0.dev0)
    - Fixed in https://github.com/vllm-project/vllm/pull/29686 (released in v0.12.0)
    - This can be removed when TRL requires vLLM>=0.12.0
    """
    if _is_package_version_at_least("transformers", "5.0.0.dev0") and _is_package_version_below("vllm", "0.12.0"):
        try:
            import contextlib
            import copy

            import vllm.transformers_utils.tokenizer

            def get_cached_tokenizer(tokenizer):
                cached_tokenizer = copy.copy(tokenizer)
                tokenizer_all_special_ids = tokenizer.all_special_ids
                tokenizer_all_special_tokens = tokenizer.all_special_tokens
                tokenizer_vocab = tokenizer.get_vocab()
                tokenizer_len = len(tokenizer)

                max_token_id = max(tokenizer_vocab.values())
                if hasattr(tokenizer, "vocab_size"):
                    with contextlib.suppress(NotImplementedError):
                        max_token_id = max(max_token_id, tokenizer.vocab_size)

                class CachedTokenizer(tokenizer.__class__):  # type: ignore
                    @property
                    def all_special_ids(self) -> list[int]:
                        return tokenizer_all_special_ids

                    @property
                    def all_special_tokens(self) -> list[str]:
                        return tokenizer_all_special_tokens

                    @property
                    def max_token_id(self) -> int:
                        return max_token_id

                    def get_vocab(self) -> dict[str, int]:
                        return tokenizer_vocab

                    def __len__(self) -> int:
                        return tokenizer_len

                    def __reduce__(self):
                        return get_cached_tokenizer, (tokenizer,)

                CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

                cached_tokenizer.__class__ = CachedTokenizer
                return cached_tokenizer

            vllm.transformers_utils.tokenizer.get_cached_tokenizer = get_cached_tokenizer
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to patch vLLM cached_tokenizer: {e}", stacklevel=2)


def _patch_transformers_hybrid_cache() -> None:
    """
    Fix HybridCache import for transformers v5 compatibility.

    - Issue: liger_kernel and peft import HybridCache from transformers.cache_utils
    - HybridCache removed in https://github.com/huggingface/transformers/pull/43168 (transformers>=5.0.0.dev0)
    - Fixed in liger_kernel: https://github.com/linkedin/Liger-Kernel/pull/1002 (released in v0.6.5)
    - Fixed in peft: https://github.com/huggingface/peft/pull/2735 (released in v0.18.0)
    - This can be removed when TRL requires liger_kernel>=0.6.5 and peft>=0.18.0
    """
    if _is_package_version_at_least("transformers", "5.0.0.dev0") and (
        _is_package_version_below("liger_kernel", "0.6.5") or _is_package_version_below("peft", "0.18.0")
    ):
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


# Apply vLLM patches
_patch_vllm_logging()
_patch_vllm_disabled_tqdm()
_patch_vllm_cached_tokenizer()

# Apply transformers patches
_patch_transformers_hybrid_cache()
