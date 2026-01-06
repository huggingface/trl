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

import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .import_utils import _LazyModule


try:
    __version__ = version("trl")
except PackageNotFoundError:
    __version__ = "unknown"

_import_structure = {
    "chat_template_utils": ["add_response_schema", "clone_chat_template", "get_training_chat_template"],
    "data_utils": [
        "apply_chat_template",
        "extract_prompt",
        "is_conversational",
        "is_conversational_from_value",
        "maybe_apply_chat_template",
        "maybe_convert_to_chatml",
        "maybe_extract_prompt",
        "maybe_unpair_preference_dataset",
        "pack_dataset",
        "prepare_multimodal_messages",
        "prepare_multimodal_messages_vllm",
        "truncate_dataset",
        "unpair_preference_dataset",
    ],
    "models": [
        "AutoModelForCausalLMWithValueHead",  # deprecated import
        "AutoModelForSeq2SeqLMWithValueHead",  # deprecated import
        "PreTrainedModelWrapper",  # deprecated import
        "create_reference_model",
    ],
    "scripts": ["DatasetMixtureConfig", "ScriptArguments", "TrlParser", "get_dataset", "init_zero_verbose"],
    "trainer": [
        "AllTrueJudge",  # deprecated import
        "BaseBinaryJudge",  # deprecated import
        "BaseJudge",  # deprecated import
        "BasePairwiseJudge",  # deprecated import
        "BaseRankJudge",  # deprecated import
        "BCOConfig",  # deprecated import
        "BCOTrainer",  # deprecated import
        "BEMACallback",
        "CPOConfig",  # deprecated import
        "CPOTrainer",  # deprecated import
        "DPOConfig",
        "DPOTrainer",
        "FDivergenceConstants",
        "FDivergenceType",
        "GKDConfig",  # deprecated import
        "GKDTrainer",  # deprecated import
        "GRPOConfig",
        "GRPOTrainer",
        "HfPairwiseJudge",  # deprecated import
        "KTOConfig",
        "KTOTrainer",
        "LogCompletionsCallback",
        "ModelConfig",
        "NashMDConfig",  # deprecated import
        "NashMDTrainer",  # deprecated import
        "OnlineDPOConfig",  # deprecated import
        "OnlineDPOTrainer",  # deprecated import
        "OpenAIPairwiseJudge",  # deprecated import
        "ORPOConfig",  # deprecated import
        "ORPOTrainer",  # deprecated import
        "PairRMJudge",  # deprecated import
        "PPOConfig",  # deprecated import
        "PPOTrainer",  # deprecated import
        "PRMConfig",  # deprecated import
        "PRMTrainer",  # deprecated import
        "RewardConfig",
        "RewardTrainer",
        "RichProgressCallback",
        "RLOOConfig",
        "RLOOTrainer",
        "SFTConfig",
        "SFTTrainer",
        "SyncRefModelCallback",
        "WeaveCallback",
        "WinRateCallback",  # deprecated import
        "XPOConfig",  # deprecated import
        "XPOTrainer",  # deprecated import
        "get_kbit_device_map",
        "get_peft_config",
        "get_quantization_config",
    ],
}

if TYPE_CHECKING:
    from .chat_template_utils import add_response_schema, clone_chat_template, get_training_chat_template
    from .data_utils import (
        apply_chat_template,
        extract_prompt,
        is_conversational,
        is_conversational_from_value,
        maybe_apply_chat_template,
        maybe_convert_to_chatml,
        maybe_extract_prompt,
        maybe_unpair_preference_dataset,
        pack_dataset,
        prepare_multimodal_messages,
        prepare_multimodal_messages_vllm,
        truncate_dataset,
        unpair_preference_dataset,
    )
    from .models import (
        AutoModelForCausalLMWithValueHead,  # deprecated import
        AutoModelForSeq2SeqLMWithValueHead,  # deprecated import
        PreTrainedModelWrapper,  # deprecated import
        create_reference_model,
    )
    from .scripts import DatasetMixtureConfig, ScriptArguments, TrlParser, get_dataset, init_zero_verbose
    from .trainer import (
        AllTrueJudge,  # deprecated import
        BaseBinaryJudge,  # deprecated import
        BaseJudge,  # deprecated import
        BasePairwiseJudge,  # deprecated import
        BaseRankJudge,  # deprecated import
        BCOConfig,  # deprecated import
        BCOTrainer,  # deprecated import
        BEMACallback,
        CPOConfig,  # deprecated import
        CPOTrainer,  # deprecated import
        DPOConfig,
        DPOTrainer,
        FDivergenceConstants,
        FDivergenceType,
        GKDConfig,  # deprecated import
        GKDTrainer,  # deprecated import
        GRPOConfig,
        GRPOTrainer,
        HfPairwiseJudge,  # deprecated import
        KTOConfig,
        KTOTrainer,
        LogCompletionsCallback,
        ModelConfig,
        NashMDConfig,  # deprecated import
        NashMDTrainer,  # deprecated import
        OnlineDPOConfig,  # deprecated import
        OnlineDPOTrainer,  # deprecated import
        OpenAIPairwiseJudge,  # deprecated import
        ORPOConfig,  # deprecated import
        ORPOTrainer,  # deprecated import
        PairRMJudge,  # deprecated import
        PPOConfig,  # deprecated import
        PPOTrainer,  # deprecated import
        PRMConfig,  # deprecated import
        PRMTrainer,  # deprecated import
        RewardConfig,
        RewardTrainer,
        RichProgressCallback,
        RLOOConfig,
        RLOOTrainer,
        SFTConfig,
        SFTTrainer,
        SyncRefModelCallback,
        WeaveCallback,
        WinRateCallback,  # deprecated import
        XPOConfig,  # deprecated import
        XPOTrainer,  # deprecated import
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


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
