# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import random
import unittest

from transformers import is_bitsandbytes_available, is_comet_available, is_sklearn_available, is_wandb_available

from trl import BaseBinaryJudge, BasePairwiseJudge, is_diffusers_available, is_llm_blender_available
from trl.import_utils import is_mergekit_available


# transformers.testing_utils contains a require_bitsandbytes function, but relies on pytest markers which we don't use
# in our test suite. We therefore need to implement our own version of this function.
def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires bitsandbytes. Skips the test if bitsandbytes is not available.
    """
    return unittest.skipUnless(is_bitsandbytes_available(), "test requires bitsandbytes")(test_case)


def require_diffusers(test_case):
    """
    Decorator marking a test that requires diffusers. Skips the test if diffusers is not available.
    """
    return unittest.skipUnless(is_diffusers_available(), "test requires diffusers")(test_case)


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


def require_no_wandb(test_case):
    """
    Decorator marking a test that requires no wandb. Skips the test if wandb is available.
    """
    return unittest.skipUnless(not is_wandb_available(), "test requires no wandb")(test_case)


def require_sklearn(test_case):
    """
    Decorator marking a test that requires sklearn. Skips the test if sklearn is not available.
    """
    return unittest.skipUnless(is_sklearn_available(), "test requires sklearn")(test_case)


def require_comet(test_case):
    """
    Decorator marking a test that requires Comet. Skips the test if Comet is not available.
    """
    return unittest.skipUnless(is_comet_available(), "test requires comet_ml")(test_case)


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
