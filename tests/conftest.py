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

import gc
import logging
import os
import sys
import traceback
from functools import wraps

import pytest
import torch
from transformers.utils import is_peft_available, is_torch_xpu_available

from trl.import_utils import is_vllm_available


# ============================================================================
# Silence transformers "LOAD REPORT" tables
# ============================================================================
# transformers >= 5 prints a "LOAD REPORT" table whenever a checkpoint has missing/mismatched/unexpected keys
# (transformers/utils/loading_report.py). TRL's tiny test models don't serialize the tied `lm_head.weight`, so
# this fires on nearly every model load and floods the test output. It is emitted through `logging` at WARNING
# level from three loggers; we drop only these records, keeping all other warnings visible.
class _DropLoadReport(logging.Filter):
    def filter(self, record):
        return "LOAD REPORT" not in record.getMessage()


for _logger_name in ("transformers.modeling_utils", "transformers.modeling_layers", "transformers.integrations.peft"):
    logging.getLogger(_logger_name).addFilter(_DropLoadReport())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Clear traceback frame locals after a failed test to release CUDA tensor references.

    When a test fails (especially with OOM), the exception traceback holds references to every local variable in every
    frame on the call stack at the time of failure — including the model, trainer, and all intermediate tensors.
    gc.collect() cannot free objects that are still reachable through a live traceback, so memory accumulates across
    reruns (~2 GiB per rerun for Gemma4, reaching 5 × 2.38 GiB = 11.89 GiB after 5 reruns). Clearing the frame locals
    breaks those reference chains so that the subsequent gc.collect() + empty_cache() in cleanup_gpu can actually
    reclaim the CUDA memory before the next attempt.
    """
    yield
    if call.when == "call" and call.excinfo is not None:
        traceback.clear_frames(call.excinfo.tb)
        # Also clear all reachable chained exception tracebacks (both __context__ and __cause__ at
        # every node): when OOM fires inside a try/except in the trainer, the OOM becomes __context__
        # of the outer exception and its traceback holds frame locals (model, tensors) that prevent gc
        # from releasing CUDA memory even after clear_frames above.
        stack, seen = [call.excinfo.value], set()
        while stack:
            exc = stack.pop()
            if exc is None or id(exc) in seen:
                continue
            seen.add(id(exc))
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            stack.append(exc.__context__)
            stack.append(exc.__cause__)


# ============================================================================
# Peak host memory per test
# ============================================================================
# Print the tests that raise their process's peak RSS by 2 GB or more, to catch memory regressions (#8429)
# Linux only: `ru_maxrss` is in bytes on macOS, and `resource` does not exist on Windows (#8441)
if sys.platform == "linux":
    import resource

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(item):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        yield
        increase = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
        item.user_properties.append(("peak_rss_increase_gb", increase / 2**20))

    def pytest_runtest_logreport(report):
        # Written as each test ends, so it reaches the log even when the job is killed before the summary (#8429)
        if report.when == "call":
            increase = dict(report.user_properties)["peak_rss_increase_gb"]
            if increase >= 2:
                print(f"\n{report.nodeid} raised its process's peak host memory by {increase:.2f} GB", flush=True)  # noqa: T201


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
    "trl-internal-testing/tiny-Idefics3ForConditionalGeneration": "9c15ab22340fc1f688542824b7a433ebef714b8e",
    "trl-internal-testing/tiny-Inkling2LoopForConditionalGeneration": "5e761216759d5d040c09f458d5f077b9aaa5ac78",
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
    "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration": "ba190cb38011bcda2f70f7781f2b8290627aef00",
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-3.8": "637a826146b4e082d115b988478a3c7651d7a980",
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink": "c21e64ab01a61fe5675253768d359262f6a19ab6",
    "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-Think": "6860d7ec861ade2fe4e44dc69bf57064fdc82b12",
    "trl-internal-testing/tiny-Qwen3_5MoeForConditionalGeneration-3.6": "57faabc3b49ec3ae69b8d6158130ce19496bc2d3",
    "trl-internal-testing/tiny-RemoteForCausalLM": "e857b87273d2b5db9c3ee15a0229d715a2e307ce",
    "trl-internal-testing/tiny-SmolLM3ForCausalLM": "a93d8be3babaeb72402f6c9551689d1f9c2eb2ee",
    "trl-internal-testing/tiny-SmolVLMForConditionalGeneration": "87d4ff483bc2b4f07d3cf629305a807243829be5",
    "trl-internal-testing/tiny-T5ForConditionalGeneration": "1e80b8121ed01c2d3fc14fd85c7f70edcc4e925d",
}


def check_pinned(model_id, revision):
    # A tiny model loaded without a revision must be pinned, otherwise the tests load whatever is on the Hub `main`
    if (
        str(model_id).startswith("trl-internal-testing/")
        and model_id not in MODEL_REVISIONS
        and revision in (None, "main")
    ):
        raise ValueError(f"{model_id} is not pinned: add it to `MODEL_REVISIONS` in tests/conftest.py")


@pytest.fixture(autouse=True)
def apply_model_revisions(monkeypatch):
    """Auto-inject revision parameter for models defined in MODEL_REVISIONS."""
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )

    def create_classmethod_wrapper(original_classmethod):
        # Extract the underlying function from the classmethod
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, pretrained_model_name_or_path, *args, **kwargs):
            check_pinned(pretrained_model_name_or_path, kwargs.get("revision"))
            # Direct lookup: only inject if model_id is in the override dict
            if pretrained_model_name_or_path in MODEL_REVISIONS:
                # The scripts pass the default revision "main" explicitly (see #8263)
                if kwargs.get("revision") in (None, "main"):
                    kwargs["revision"] = MODEL_REVISIONS[pretrained_model_name_or_path]
                    # Clear _commit_hash: Auto classes resolve it from the default branch before calling
                    # sub-loaders, so the cached hash points to main. If we don't clear it, it silently
                    # overrides the injected revision for the config load while the weight loader uses the
                    # revision, producing a config/weights shape mismatch.
                    kwargs.pop("_commit_hash", None)

            return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Re-wrap as classmethod
        return classmethod(wrapper)

    def create_method_wrapper(original_method):
        @wraps(original_method)
        def wrapper(self, model_id, *args, **kwargs):
            check_pinned(model_id, kwargs.get("revision"))
            if model_id in MODEL_REVISIONS and kwargs.get("revision") in (None, "main"):
                kwargs["revision"] = MODEL_REVISIONS[model_id]
            return original_method(self, model_id, *args, **kwargs)

        return wrapper

    # Patch the transformers Auto* classes and the base classes they dispatch to
    for cls in [
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        # `AutoTokenizer`/`AutoProcessor` resolve the chat template before dispatching, so patching only the base
        # classes loads it from the default branch.
        AutoProcessor,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    ]:
        monkeypatch.setattr(cls, "from_pretrained", create_classmethod_wrapper(cls.from_pretrained))

    def create_peft_classmethod_wrapper(original_classmethod):
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, model, model_id, *args, **kwargs):
            check_pinned(model_id, kwargs.get("revision"))
            if model_id in MODEL_REVISIONS and kwargs.get("revision") in (None, "main"):
                kwargs["revision"] = MODEL_REVISIONS[model_id]
            return original_func(cls, model, model_id, *args, **kwargs)

        return classmethod(wrapper)

    # PEFT adapters never reach the loaders above.
    if is_peft_available():
        from peft import AutoPeftModelForCausalLM, PeftModel

        monkeypatch.setattr(
            AutoPeftModelForCausalLM,
            "from_pretrained",
            create_classmethod_wrapper(AutoPeftModelForCausalLM.from_pretrained),
        )
        # `PeftModel.from_pretrained` takes the base model first and the adapter id second.
        monkeypatch.setattr(PeftModel, "from_pretrained", create_peft_classmethod_wrapper(PeftModel.from_pretrained))
        # `load_adapter` is an instance method, not a classmethod.
        monkeypatch.setattr(PeftModel, "load_adapter", create_method_wrapper(PeftModel.load_adapter))


@pytest.fixture(autouse=True)
def force_use_cpu_without_accelerator(monkeypatch):
    """Force `use_cpu=True` on all TRL configs when no accelerator is available.

    TRL configs default `bf16` to `True` (see `_BaseConfig.__post_init__`), which makes
    `transformers.TrainingArguments.__post_init__` raise `ValueError: Your setup doesn't support bf16/gpu. You need to
    assign use_cpu ...` on machines without a GPU. Setting `use_cpu=True` before that validation runs lets the whole
    suite run on CPU without editing every individual config in the tests. On a machine with an accelerator (CUDA, XPU,
    NPU, ...) this fixture is a no-op, so on-device tests and explicit `use_cpu=False` are left untouched.
    """
    from transformers.testing_utils import torch_device

    if torch_device is not None and torch_device != "cpu":
        return

    from trl.trainer.base_config import _BaseConfig

    original_post_init = _BaseConfig.__post_init__

    def patched_post_init(self):
        self.use_cpu = True
        original_post_init(self)

    monkeypatch.setattr(_BaseConfig, "__post_init__", patched_post_init)


@pytest.fixture(autouse=True)
def restore_mixed_precision_env(monkeypatch):
    """
    Restore the mixed precision that `transformers.TrainingArguments` publishes process-wide.

    On transformers < 5, `TrainingArguments.__post_init__` writes the mixed precision to `ACCELERATE_MIXED_PRECISION`
    and reads that same variable back as the default for the next instantiation, where a `bf16=False` flag is
    indistinguishable from an unset one. TRL configs default `bf16` to `True`, so the first trainer built in a worker
    sets the variable to `bf16` for the whole process, and no config built later can clear it. transformers < 5 also
    builds the `Accelerator` without passing the mixed precision, so accelerate falls back to the leaked variable and a
    test that asks for `bf16=False` silently runs in bf16: with `test_chunked_logps_match_full_logits`, the streamed
    projection enters autocast and the full-logits path does not, and the two no longer match. Setting the variable to
    its current value arms monkeypatch's restore, so each test sees the value the session started with.

    Only the minimum-versions CI job is affected: transformers >= 5 passes the mixed precision to the `Accelerator`
    explicitly and never writes the variable.
    """
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", os.environ.get("ACCELERATE_MIXED_PRECISION", "no"))


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """
    Automatically cleanup accelerator memory after each test.

    This fixture helps prevent accelerator out of memory errors when running tests in parallel with pytest-xdist by
    ensuring models and tensors are properly garbage collected and accelerator memory caches are cleared between tests.
    """
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if is_torch_xpu_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_vllm_dist_env():
    """
    Destroy the torch.distributed process group left behind by vLLM colocate mode after each test.

    vLLM colocate mode initializes `torch.distributed` internally (`distributed_executor_backend="external_launcher"`)
    but the offline `LLM` class never tears it down: neither `LLM` nor `LLMEngine` calls vLLM's own
    `cleanup_dist_env_and_memory()` on exit. Left undestroyed, PyTorch warns at interpreter shutdown (e.g.
    `TestGRPOTrainerSlow::test_vlm_processor_vllm_colocate_mode`). We call vLLM's own cleanup helper rather than a bare
    `torch.distributed.destroy_process_group()` so that vLLM's internal parallel-state globals (`_TP`, `_WORLD`, ...)
    are reset too, letting the next colocate test reinitialize cleanly.

    It also restores `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`, which TRL sets before
    building the vLLM `LLM` in colocate mode. Left set, they make the next test's `Accelerator` initialize
    `torch.distributed` on the leaked port.
    """
    dist_env = {
        name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    }
    yield
    if torch.distributed.is_initialized() and is_vllm_available():
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    for name, value in dist_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
