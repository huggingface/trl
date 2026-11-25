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

import functools
import signal
import warnings
from collections.abc import Callable

import psutil
import pytest
import torch
from transformers import is_bitsandbytes_available, is_comet_available, is_wandb_available
from transformers.testing_utils import backend_device_count, torch_device
from transformers.utils import (
    is_kernels_available,
    is_peft_available,
    is_rich_available,
    is_torch_available,
    is_torch_bf16_gpu_available,
    is_torch_xla_available,
    is_triton_available,
    is_vision_available,
)

from trl.chat_template_utils import _SUPPORTS_RESPONSE_TEMPLATE
from trl.import_utils import (
    is_harbor_available,
    is_jmespath_available,
    is_math_verify_available,
    is_mergekit_available,
    is_openreward_available,
    is_requests_available,
    is_vllm_available,
)


# ============================================================================
# Tiny Model Revisions
# ============================================================================
# Tiny models are test fixtures: each one the tests load is pinned to a commit sha of its Hub repo, so that every TRL
# commit, PR and release keeps testing against the files it was written for, whatever is later pushed to the Hub.
#
# To update a tiny model:
# 1. Open a PR on its Hub repo (`--create-pr` in `scripts/generate_tiny_models`)
# 2. In a TRL PR, temporarily set the model's entry to the Hub PR ref (e.g. "refs/pr/5") to run CI against it
# 3. Once CI is green, merge the Hub PR
# 4. Replace the PR ref with the commit sha that the merge created on the Hub `main`, then merge the TRL PR
#
# A new tiny model is pushed directly to the Hub `main`: add its entry with that commit sha.
# ============================================================================

MODEL_REVISIONS = {
    "trl-internal-testing/small-Qwen2ForCausalLM-2.5": "b830994d3efdb531d8e2fda07d568895260700c9",
    "trl-internal-testing/small-Qwen3ForCausalLM": "fffedda96f7480b1feea394e5d32d6b0f999be18",
    "trl-internal-testing/tiny-BartModel": "3f66ee1917d23448af442a82b22601b65ef13957",
    "trl-internal-testing/tiny-BeamForCausalLM": "81bd596f7b53bc5d843d1218a4521119aae05d51",
    "trl-internal-testing/tiny-Cohere2ForCausalLM": "5ab7c17080fbe53a8f86417804dd1db29ee3e111",
    "trl-internal-testing/tiny-CohereForCausalLM": "83a435edaadff74a1bbb4499f1b96c9c5725d43b",
    "trl-internal-testing/tiny-DeepseekV3ForCausalLM": "2fb366d060ad5cea8917909f28ec1219ccb1ad2d",
    "trl-internal-testing/tiny-DeepseekV3ForCausalLM-0528": "569162254a4c32144b6ad2adfc0e81cc943cbec9",
    "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion": "95af0b2bf27db2637cd62d086e27db3cc85c94ae",
    "trl-internal-testing/tiny-FalconMambaForCausalLM": "32e6aa438302dfac0af01c357d268d631c3467ab",
    "trl-internal-testing/tiny-GPT2LMHeadModel": "ce0e1654a5f88d887cae3899ed6a73a2e452a8ce",
    "trl-internal-testing/tiny-GPTNeoXForCausalLM": "76f5e9019accfedbdc35b70518218ca2dac0909b",
    "trl-internal-testing/tiny-GPTNeoXForSequenceClassification": "24554e9fc0c179af69687fbb447a75598f2ee448",
    "trl-internal-testing/tiny-Gemma2ForCausalLM": "5b25f835b5ed059c0c282e79f7227f4a929ab235",
    "trl-internal-testing/tiny-Gemma3ForConditionalGeneration": "82a229b52bdddb50c4d1d79d723e48579a9b79f7",
    "trl-internal-testing/tiny-Gemma4ForConditionalGeneration": "0dc1746b7f9f623b748e735ed5a4302eb4baf346",
    "trl-internal-testing/tiny-GemmaForCausalLM": "08058fb469384b006af92d452849b0371f593a9a",
    "trl-internal-testing/tiny-Glm4MoeForCausalLM": "9c73ba05132e99afd6e5bc13468e1f72c10c7c1a",
    "trl-internal-testing/tiny-Glm5NextForConditionalGeneration": "9339c1e41517bf775383d28e286be70bcaef8334",
    "trl-internal-testing/tiny-GptOssForCausalLM": "cb95c03d6ab62791e09ef922fc28856d18b45909",
    "trl-internal-testing/tiny-GraniteMoeForCausalLM": "76b45c5376916fcc17bdeb598df3337cd14a4f5e",
    "trl-internal-testing/tiny-GraniteMoeHybridForCausalLM": "0ea8ef8b1cb6b15f9d47860269de792c96e69d23",
    "trl-internal-testing/tiny-Idefics2ForConditionalGeneration": "05488b3ae9bb1815d8bc16019d3a60f5387e3f54",
    "trl-internal-testing/tiny-Idefics3ForConditionalGeneration": "ae70d0c8b17b38347e1f7d15d064efc077048781",
    "trl-internal-testing/tiny-Inkling2LoopForConditionalGeneration": "5e761216759d5d040c09f458d5f077b9aaa5ac78",
    "trl-internal-testing/tiny-InklingForConditionalGeneration": "0e2558582d4c6c8538d178db0c2df065ac5a9ac5",
    "trl-internal-testing/tiny-InternVLForConditionalGeneration": "99c02d42d51fd4f83e12f19850a126627d473109",
    "trl-internal-testing/tiny-Lfm2ForCausalLM": "a8b729ddf80657a21272c71d36a7edc2be1bf6b4",
    "trl-internal-testing/tiny-Lfm2ForCausalLM-2.5": "1e9ac722a12cea565443504973060b960a328c37",
    "trl-internal-testing/tiny-Lfm2MoeForCausalLM-2.5": "af6b0e20385279de1530104e215a3a2f5155d182",
    "trl-internal-testing/tiny-Lfm2VlForConditionalGeneration-2.5": "754269127308ed7bdaa584630abc702e368337ad",
    "trl-internal-testing/tiny-LlamaForCausalLM-3": "fed7bca1617c3a3835860260e81aa387e7afe07d",
    "trl-internal-testing/tiny-LlamaForCausalLM-3.1": "c234f265b635252535f848a9609c50852614ac84",
    "trl-internal-testing/tiny-LlamaForCausalLM-3.2": "ca56c655e3da5ec96243a4b470faf767a3085783",
    "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2": "769052e57a412de8772065109aeb4d646985f5e6",
    "trl-internal-testing/tiny-LlavaForConditionalGeneration": "cd71a22c800134d15563aba02c2ab4d6368fbb9b",
    "trl-internal-testing/tiny-LlavaNextForConditionalGeneration": "34b0eb03416a48ccfa394261b109c845e6fee0f1",
    "trl-internal-testing/tiny-Mistral3ForConditionalGeneration": "d9f67e86d81e78ba4d6493463a55b082ed670ff5",
    "trl-internal-testing/tiny-MistralForCausalLM-0.1": "d9e019faaeff83b77b8e047562a22325082a2a18",
    "trl-internal-testing/tiny-MistralForCausalLM-0.2": "49f9e51102f6800add4544c2a6b62f3ed895dd5b",
    "trl-internal-testing/tiny-MistralLarge4ForConditionalGeneration": "9a878a791b40e08c176aeaaa7c4ee2d9cc19d77b",
    "trl-internal-testing/tiny-MuseGlimmerForConditionalGeneration": "2daa66705814668209044d2b1c84e1413b308330",
    "trl-internal-testing/tiny-MuseSparkForConditionalGeneration": "c9e83e8784d89e7e7f66f78e15adf10e3a93fdbb",
    "trl-internal-testing/tiny-NemotronHForCausalLM-3.5-lightning": "0fd0a2356bc9530b41fb96f4981b52c6a2ac95fd",
    "trl-internal-testing/tiny-NemotronHForCausalLM-nano": "fd6746d3f4f5c3950d43c6c4adc909d5fd5dd483",
    "trl-internal-testing/tiny-NemotronHForCausalLM-super": "3099eedcee3794932297381fb276d8639aa2cf26",
    "trl-internal-testing/tiny-NemotronHForCausalLM-ultra": "6056386efc9b68589b7a3a5b5174f7275250a6f5",
    "trl-internal-testing/tiny-OPTForCausalLM": "b62ab91d8bf79cfb278e9461352b33761596a307",
    "trl-internal-testing/tiny-Olmo3ForCausalLM": "cb9b96ec80fe91efa16b1603292957296526745b",
    "trl-internal-testing/tiny-PeftModel": "0b1efa692a53354c185750da2477f2ec616e1409",
    "trl-internal-testing/tiny-PeftModel-2": "04c82ed36f7936065b195bce1a26ae5aa04a81d6",
    "trl-internal-testing/tiny-Phi3ForCausalLM-3": "e82e6bc7a65dc695e68b5c80fc92df026948b290",
    "trl-internal-testing/tiny-Phi3ForCausalLM-3.5": "52f76be6ee990d030045d3eba2e84cd355e02f11",
    "trl-internal-testing/tiny-PhimoeForCausalLM": "0cef461bfb86d74930dc206ffe11f97d4bb8671a",
    "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5": "edca82771e1fad8f0b7ce110ffe770c682043fec",
    "trl-internal-testing/tiny-Qwen2ForCausalLM-R1-Distill": "8b7f1eaa6e327610d36ea8f8a677e26f3e3c2e4d",
    "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5": "58b5b1655194e908b2af57189fccd4254b894f20",
    "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration": "c94ec1814fcc56a99924bf44690f924fdcb6d934",
    "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration": "a2ae4f40e2b25c3b89a204dc3ed798b4ebc20318",
    "trl-internal-testing/tiny-Qwen3ForCausalLM": "92ee0a3c801d3ffbec16626c3b9f5fe60e778253",
    "trl-internal-testing/tiny-Qwen3ForCausalLM-Instruct-2507": "a7138d16d33439cd402a801d072bd5ee63a9d8cb",
    "trl-internal-testing/tiny-Qwen3ForSequenceClassification": "c892ea6c5849514ac3115d6e0df369279994e759",
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM": "c3314bf03b67c8e547516e9c5fda87c08a88d480",
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM-Coder": "41ed00c633513d258fb895034e1b8b32c002b101",
    "trl-internal-testing/tiny-Qwen3MoeForCausalLM-ResponseTemplate": "7437f810e325cc75beb536af291e99fa2dea5489",
    "trl-internal-testing/tiny-Qwen3NextForCausalLM-Coder-Next": "cf43fb018811873615cdc72180554037caf861dc",
    "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration": "de6ab9a29f6abaf816cf177b647ca89793c4aa98",
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-3.8": "637a826146b4e082d115b988478a3c7651d7a980",
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink": "c21e64ab01a61fe5675253768d359262f6a19ab6",
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-Think": "6860d7ec861ade2fe4e44dc69bf57064fdc82b12",
    "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6": "57faabc3b49ec3ae69b8d6158130ce19496bc2d3",
    "trl-internal-testing/tiny-RemoteForCausalLM": "e857b87273d2b5db9c3ee15a0229d715a2e307ce",
    "trl-internal-testing/tiny-SmolLM3ForCausalLM": "a93d8be3babaeb72402f6c9551689d1f9c2eb2ee",
    "trl-internal-testing/tiny-SmolVLMForConditionalGeneration": "87d4ff483bc2b4f07d3cf629305a807243829be5",
    "trl-internal-testing/tiny-T5ForConditionalGeneration": "1e80b8121ed01c2d3fc14fd85c7f70edcc4e925d",
}


require_bitsandbytes = pytest.mark.skipif(not is_bitsandbytes_available(), reason="test requires bitsandbytes")
require_comet = pytest.mark.skipif(not is_comet_available(), reason="test requires comet_ml")
require_harbor = pytest.mark.skipif(not is_harbor_available(), reason="test requires harbor")
require_kernels = pytest.mark.skipif(not is_kernels_available(), reason="test requires kernels")
require_math_latex = pytest.mark.skipif(not is_math_verify_available(), reason="test requires math_verify")
require_mergekit = pytest.mark.skipif(not is_mergekit_available(), reason="test requires mergekit")
require_openreward = pytest.mark.skipif(not is_openreward_available(), reason="test requires openreward")
require_peft = pytest.mark.skipif(not is_peft_available(), reason="test requires peft")
require_requests = pytest.mark.skipif(not is_requests_available(), reason="test requires requests")
# Response parsing needs jmespath only on transformers < 5.13, which ships the legacy `response_schema` parser; the
# new-style `response_template` parser doesn't use it. See `_SUPPORTS_RESPONSE_TEMPLATE`.
require_response_parsing = pytest.mark.skipif(
    not _SUPPORTS_RESPONSE_TEMPLATE and not is_jmespath_available(),
    reason="test requires jmespath for response parsing on transformers below 5.13.0",
)
require_rich = pytest.mark.skipif(not is_rich_available(), reason="test requires rich")
require_torch_accelerator = pytest.mark.skipif(
    torch_device is None or torch_device == "cpu", reason="test requires accelerator"
)
require_torch_multi_accelerator = pytest.mark.skipif(
    not is_torch_available() or backend_device_count(torch_device) <= 1, reason="test requires multiple accelerators"
)
require_triton = pytest.mark.skipif(not is_triton_available(), reason="test requires triton")
require_vision = pytest.mark.skipif(not is_vision_available(), reason="test requires vision")
require_vllm = pytest.mark.skipif(not is_vllm_available(), reason="test requires vllm")
require_wandb = pytest.mark.skipif(not is_wandb_available(), reason="test requires wandb")
require_no_wandb = pytest.mark.skipif(is_wandb_available(), reason="test requires no wandb")
require_3_accelerators = pytest.mark.skipif(
    not (getattr(torch, torch_device, torch.cuda).device_count() >= 3),
    reason=f"test requires at least 3 {torch_device}s",
)


def is_ampere_or_newer(device_index=0):
    if not torch.cuda.is_available():
        return False

    # "Ampere" is an NVIDIA architecture; an AMD (ROCm) GPU is never Ampere. On ROCm,
    # torch.cuda.get_device_capability returns the gfx version, which would spuriously compare >= (8, 0).
    if torch.version.hip is not None:
        return False

    major, minor = torch.cuda.get_device_capability(device_index)
    # Ampere starts at compute capability 8.0 (e.g., A100 = 8.0, RTX 30xx = 8.6)
    return (major, minor) >= (8, 0)


def is_bf16_supported() -> bool:
    """Whether the current device has hardware bf16 support for `bf16=True` training.

    `is_torch_bf16_gpu_available` already covers CUDA (Ampere or newer), XPU, HPU, NPU, etc., and XLA devices are added
    via `is_torch_xla_available`. On a CPU or a pre-Ampere GPU (e.g. T4), `bf16=True` raises "Your setup doesn't
    support bf16/gpu", so tests should pass `bf16=is_bf16_supported()` instead of a hard-coded `True`.

    `transformers`' `TrainingArguments` validation also accepts `bf16=True` when `use_cpu=True` (CPU training is an
    explicit opt-in). This helper deliberately does not treat that as supported, since it reports hardware bf16
    capability; tests setting `use_cpu=True` should not rely on it.
    """
    return is_torch_bf16_gpu_available() or is_torch_xla_available()


class TrlTestCase:
    @pytest.fixture(autouse=True)
    def set_tmp_dir(self, tmp_path):
        self.tmp_dir = str(tmp_path)


def ignore_warnings(message: str = None, category: type[Warning] = Warning) -> Callable:
    """
    Decorator to ignore warnings with a specific message and/or category.

    Args:
        message (`str`, *optional*):
            Regex pattern for the warning message to ignore. If `None`, all messages are ignored.
        category (`type[Warning]`, *optional*, defaults to `Warning`):
            Warning class to ignore. Defaults to `Warning`, which ignores all warnings.
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=message, category=category)
                return test_func(*args, **kwargs)

        return wrapper

    return decorator


def kill_process(process):
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=5)
        except psutil.TimeoutExpired:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        process.kill()
    except psutil.NoSuchProcess:
        pass
