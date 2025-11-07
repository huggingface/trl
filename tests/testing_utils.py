# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
import random
import signal
import warnings
from collections.abc import Callable

import psutil
import pytest
import torch
from transformers import is_bitsandbytes_available, is_comet_available, is_sklearn_available, is_wandb_available
from transformers.testing_utils import backend_device_count, torch_device
from transformers.utils import (
    is_flash_attn_2_available,
    is_kernels_available,
    is_peft_available,
    is_rich_available,
    is_torch_available,
    is_vision_available,
)

from trl.experimental.judges import BaseBinaryJudge, BasePairwiseJudge
from trl.import_utils import (
    is_joblib_available,
    is_liger_kernel_available,
    is_llm_blender_available,
    is_math_verify_available,
    is_mergekit_available,
    is_vllm_available,
)


require_bitsandbytes = pytest.mark.skipif(not is_bitsandbytes_available(), reason="test requires bitsandbytes")
require_comet = pytest.mark.skipif(not is_comet_available(), reason="test requires comet_ml")
require_liger_kernel = pytest.mark.skipif(not is_liger_kernel_available(), reason="test requires liger-kernel")
require_llm_blender = pytest.mark.skipif(not is_llm_blender_available(), reason="test requires llm-blender")
require_math_latex = pytest.mark.skipif(not is_math_verify_available(), reason="test requires math_verify")
require_mergekit = pytest.mark.skipif(not is_mergekit_available(), reason="test requires mergekit")
require_peft = pytest.mark.skipif(not is_peft_available(), reason="test requires peft")
require_rich = pytest.mark.skipif(not is_rich_available(), reason="test requires rich")
require_sklearn = pytest.mark.skipif(
    not (is_sklearn_available() and is_joblib_available()), reason="test requires sklearn"
)
require_torch_accelerator = pytest.mark.skipif(
    torch_device is None or torch_device == "cpu", reason="test requires accelerator"
)
require_torch_multi_accelerator = pytest.mark.skipif(
    not is_torch_available() or backend_device_count(torch_device) <= 1, reason="test requires multiple accelerators"
)
require_vision = pytest.mark.skipif(not is_vision_available(), reason="test requires vision")
require_vllm = pytest.mark.skipif(not is_vllm_available(), reason="test requires vllm")
require_wandb = pytest.mark.skipif(not is_wandb_available(), reason="test requires wandb")
require_no_wandb = pytest.mark.skipif(is_wandb_available(), reason="test requires no wandb")
require_3_accelerators = pytest.mark.skipif(
    not (getattr(torch, torch_device, torch.cuda).device_count() >= 3),
    reason=f"test requires at least 3 {torch_device}s",
)


def is_bitsandbytes_multi_backend_available() -> bool:
    if is_bitsandbytes_available():
        import bitsandbytes as bnb

        return "multi_backend" in getattr(bnb, "features", set())
    return False


# Function ported from transformers.testing_utils before transformers#41283
require_torch_gpu_if_bnb_not_multi_backend_enabled = pytest.mark.skipif(
    not is_bitsandbytes_multi_backend_available() and not torch_device == "cuda",
    reason="test requires bitsandbytes multi-backend enabled or 'cuda' torch device",
)


def is_flash_attn_available():
    flash_attn_available = is_flash_attn_2_available()
    kernels_available = is_kernels_available()
    try:
        from kernels import get_kernel

        get_kernel("kernels-community/flash-attn")
    except Exception:
        kernels_available = False

    return kernels_available or flash_attn_available


# Function ported from transformers.testing_utils
require_flash_attn = pytest.mark.skipif(not is_flash_attn_available(), reason="test requires Flash Attention")


class RandomBinaryJudge(BaseBinaryJudge):
    """
    Random binary judge, for testing purposes.
    """

    def judge(self, prompts, completions, gold_completions=None, shuffle_order=True):
        return [random.choice([0, 1, -1]) for _ in range(len(prompts))]


class RandomPairwiseJudge(BasePairwiseJudge):
    """
    Random pairwise judge, for testing purposes.
    """

    def judge(self, prompts, completions, shuffle_order=True, return_scores=False):
        if not return_scores:
            return [random.randint(0, len(completion) - 1) for completion in completions]
        else:
            return [random.random() for _ in range(len(prompts))]


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
