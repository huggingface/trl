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
import os
import traceback
import warnings
from functools import wraps

import pytest
import torch


# Per-worker baseline: captured once on the first GPU test, stays fixed for the worker's lifetime.
# This lets us track cumulative GPU memory growth across all tests in the worker, which a simple
# per-test delta cannot detect (because before_allocated grows alongside after_allocated).
_worker_gpu_baseline: int | None = None


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
# Model Revision Override
# ============================================================================
# To test a tiny model PR before merging to main:
# 1. Add the full model_id and PR revision to this dict
# 2. Commit and push to trigger CI
# 3. Once CI is green, merge the tiny model PR on HF Hub
# 4. Remove the entry from this dict and commit
#
# Example:
#   MODEL_REVISIONS = {
#       "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5": "refs/pr/3",
#       "trl-internal-testing/tiny-LlavaForConditionalGeneration": "refs/pr/5",
#   }
# ============================================================================

MODEL_REVISIONS = {
    # Add model_id: revision mappings here to test PRs
}


@pytest.fixture(autouse=True)
def apply_model_revisions(monkeypatch):
    """Auto-inject revision parameter for models defined in MODEL_REVISIONS."""
    if not MODEL_REVISIONS:
        return

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )

    def create_classmethod_wrapper(original_classmethod):
        # Extract the underlying function from the classmethod
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, pretrained_model_name_or_path, *args, **kwargs):
            # Direct lookup: only inject if model_id is in the override dict
            if pretrained_model_name_or_path in MODEL_REVISIONS:
                if "revision" not in kwargs:
                    kwargs["revision"] = MODEL_REVISIONS[pretrained_model_name_or_path]
                    # Clear _commit_hash: Auto classes resolve it from the default branch before calling
                    # sub-loaders, so the cached hash points to main. If we don't clear it, it silently
                    # overrides the injected revision for the config load while the weight loader uses the
                    # revision, producing a config/weights shape mismatch.
                    kwargs.pop("_commit_hash", None)

            return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Re-wrap as classmethod
        return classmethod(wrapper)

    # Patch all transformers Auto* classes
    for cls in [
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    ]:
        monkeypatch.setattr(cls, "from_pretrained", create_classmethod_wrapper(cls.from_pretrained))


@pytest.fixture(autouse=True)
def cleanup_gpu(request):
    """
    Automatically cleanup GPU memory after each test.

    This fixture helps prevent CUDA out of memory errors when running tests in parallel with pytest-xdist by ensuring
    models and tensors are properly garbage collected and GPU memory caches are cleared between tests.

    It also logs cumulative GPU memory growth across tests in the same worker process. A per-test delta alone is
    insufficient: if memory "sticks" after test 1, tests 2+ each measure delta≈0 because before_allocated already
    includes the stuck baseline. Cumulative tracking (from worker start) catches the slow accumulation that causes CI
    OOM.
    """
    global _worker_gpu_baseline
    if torch.cuda.is_available():
        before_allocated = torch.cuda.memory_allocated()
        if _worker_gpu_baseline is None:
            _worker_gpu_baseline = before_allocated
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after_reserved = torch.cuda.memory_reserved()
        after_allocated = torch.cuda.memory_allocated()
        delta_this_test = after_allocated - before_allocated
        cumulative = after_allocated - _worker_gpu_baseline
        # Warn if this test leaked, OR if cumulative growth from worker start is significant.
        # The cumulative check is critical: per-test delta is always ~0 after the first test
        # (before_allocated already includes the stuck memory), so it would never fire alone.
        if delta_this_test > 30 * 1024**2 or cumulative > 200 * 1024**2:
            warnings.warn(
                f"[cleanup_gpu] pid={os.getpid()} {request.node.nodeid}: "
                f"delta={delta_this_test / 1024**2:+.1f} MiB, "
                f"cumulative={cumulative / 1024**2:+.1f} MiB "
                f"(alloc: {before_allocated / 1024**2:.1f}→{after_allocated / 1024**2:.1f} MiB, "
                f"res={after_reserved / 1024**2:.1f} MiB)",
                ResourceWarning,
                stacklevel=2,
            )
