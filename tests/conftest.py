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
import sys
import traceback
from functools import wraps

import pytest
import torch


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Clear traceback frame locals after a failed test to release CUDA tensor references.

    When a test fails (especially with OOM), the exception traceback holds references to every local variable in every
    frame on the call stack at the time of failure — including the model, trainer, and all intermediate tensors.
    gc.collect() cannot free objects that are still reachable through a live traceback, so memory accumulates across
    reruns (~2 GiB per rerun for Gemma4, reaching 5 × 2.38 GiB = 11.89 GiB after 5 reruns). Clearing the frame locals
    breaks those reference chains so that the subsequent gc.collect() + empty_cache() in cleanup_gpu can actually
    reclaim the CUDA memory before the next attempt.

    We must walk the full exception chain (__context__ / __cause__) because OOM often fires inside a try/except in the
    trainer (e.g. gradient scaling, mixed-precision recovery), making the original exception a chained __context__
    whose traceback — and all its frame locals — would otherwise never be cleared.
    """
    yield
    if call.when == "call" and call.excinfo is not None:
        exc, seen = call.excinfo.value, set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            exc = exc.__cause__ if exc.__suppress_context__ else exc.__context__
        traceback.clear_frames(call.excinfo.tb)


def _host_pid() -> int:
    """Return the host-namespace PID.

    Inside a Docker container os.getpid() returns the container-namespace PID, but CUDA OOM errors report the
    host-namespace PID. /proc/self/status NSpid lists PIDs from innermost (container) to outermost (host) namespace, so
    the last entry is the one that appears in CUDA error messages.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    return int(line.split()[-1])
    except OSError:
        pass
    return os.getpid()


def pytest_runtest_logstart(nodeid, location):
    """Print host PID → xdist worker → test mapping to stderr before each test.

    When a CUDA OOM error shows process IDs, search the CI log for the PID to identify which worker and test was
    responsible for the large allocation.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    sys.stderr.write(f"[PID={_host_pid()} worker={worker}] STARTED {nodeid}\n")
    sys.stderr.flush()


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """
    Automatically cleanup GPU memory after each test.

    This fixture helps prevent CUDA out of memory errors when running tests in parallel with pytest-xdist by ensuring
    models and tensors are properly garbage collected and GPU memory caches are cleared between tests.
    """
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
