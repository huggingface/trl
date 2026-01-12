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

# Monkey-patches for vLLM.
from .import_utils import is_vllm_available  # noqa: E402


if is_vllm_available():
    import os

    os.environ["VLLM_LOGGING_LEVEL"] = os.getenv("VLLM_LOGGING_LEVEL", "ERROR")

    # Fix DisableTqdm
    # Bug introduced in https://github.com/vllm-project/vllm/pull/52
    # Fixed in https://github.com/vllm-project/vllm/pull/28471 (released in v0.11.1)
    # Since TRL currently only supports vLLM v0.10.2-0.12.0, we patch it here. This can be removed when TRL requires
    # vLLM >=0.11.1
    import vllm.model_executor.model_loader.weight_utils
    from tqdm import tqdm

    class DisabledTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    # Overwrite the class in the dependency
    vllm.model_executor.model_loader.weight_utils.DisabledTqdm = DisabledTqdm

    # Fix get_cached_tokenizer: remove all_special_tokens_extended, because it doesn't exist in transformers v5
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

    # Overwrite the function in the dependency
    vllm.transformers_utils.tokenizer.get_cached_tokenizer = get_cached_tokenizer


def _maybe_patch_transformers_hybrid_cache() -> None:
    # liger_kernel<=0.6.4 imports HybridCache from transformers, but HybridCache was removed in
    # transformers>=5.0.0.dev0 (see https://github.com/huggingface/transformers/pull/43168). This monkey patch should
    # only be needed until 0.6.5 if https://github.com/linkedin/Liger-Kernel/pull/1002 is merged and released.
    import transformers
    from packaging.version import Version
    from transformers.utils.import_utils import _is_package_available

    transformers_version = Version(transformers.__version__)
    is_liger_kernel_available, liger_kernel_version = _is_package_available("liger_kernel", return_version=True)
    liger_kernel_version = Version(liger_kernel_version) if is_liger_kernel_available else None
    if (
        is_liger_kernel_available
        and liger_kernel_version <= Version("0.6.4")
        and transformers_version >= Version("5.0.0.dev0")
    ):
        import transformers.cache_utils as cache_utils

        cache_utils.HybridCache = cache_utils.Cache


_maybe_patch_transformers_hybrid_cache()
