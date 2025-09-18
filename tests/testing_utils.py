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
import shutil
import signal
import tempfile
import unittest
import warnings

import psutil
import torch
from transformers import is_bitsandbytes_available, is_comet_available, is_sklearn_available, is_wandb_available
from transformers.testing_utils import torch_device
from transformers.utils import is_rich_available

from trl import BaseBinaryJudge, BasePairwiseJudge
from trl.import_utils import is_joblib_available, is_llm_blender_available, is_mergekit_available, is_vllm_available


# transformers.testing_utils contains a require_bitsandbytes function, but relies on pytest markers which we don't use
# in our test suite. We therefore need to implement our own version of this function.
def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires bitsandbytes. Skips the test if bitsandbytes is not available.
    """
    return unittest.skipUnless(is_bitsandbytes_available(), "test requires bitsandbytes")(test_case)


def require_comet(test_case):
    """
    Decorator marking a test that requires Comet. Skips the test if Comet is not available.
    """
    return unittest.skipUnless(is_comet_available(), "test requires comet_ml")(test_case)


def require_llm_blender(test_case):
    """
    Decorator marking a test that requires llm-blender. Skips the test if llm-blender is not available.
    """
    return unittest.skipUnless(is_llm_blender_available(), "test requires llm-blender")(test_case)


def require_mergekit(test_case):
    """
    Decorator marking a test that requires mergekit. Skips the test if mergekit is not available.
    """
    return unittest.skipUnless(is_mergekit_available(), "test requires mergekit")(test_case)


def require_rich(test_case):
    """
    Decorator marking a test that requires rich. Skips the test if rich is not available.
    """
    return unittest.skipUnless(is_rich_available(), "test requires rich")(test_case)


def require_sklearn(test_case):
    """
    Decorator marking a test that requires sklearn. Skips the test if sklearn is not available.
    """
    return unittest.skipUnless(is_sklearn_available() and is_joblib_available(), "test requires sklearn")(test_case)


def require_vllm(test_case):
    """
    Decorator marking a test that requires vllm. Skips the test if vllm is not available.
    """
    return unittest.skipUnless(is_vllm_available(), "test requires vllm")(test_case)


def require_no_wandb(test_case):
    """
    Decorator marking a test that requires no wandb. Skips the test if wandb is available.
    """
    return unittest.skipUnless(not is_wandb_available(), "test requires no wandb")(test_case)


def require_3_accelerators(test_case):
    """
    Decorator marking a test that requires at least 3 accelerators. Skips the test if 3 accelerators are not available.
    """
    torch_accelerator_module = getattr(torch, torch_device, torch.cuda)
    return unittest.skipUnless(
        torch_accelerator_module.device_count() >= 3, f"test requires at least 3 {torch_device}s"
    )(test_case)


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


class TrlTestCase(unittest.TestCase):
    """
    Base test case for TRL tests. Sets up a temporary directory for testing.
    """

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        super().tearDown()


def ignore_warnings(message: str = None, category: type[Warning] = Warning) -> callable:
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
