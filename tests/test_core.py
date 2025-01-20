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

import unittest

import torch

from trl.core import masked_mean, masked_var, masked_whiten


class CoreTester(unittest.TestCase):
    """
    A wrapper class for testing core utils functions
    """

    def setUp(self):
        self.test_input = torch.Tensor([1, 2, 3, 4])
        self.test_mask = torch.Tensor([0, 1, 1, 0])
        self.test_input_unmasked = self.test_input[1:3]

    def test_masked_mean(self):
        self.assertEqual(torch.mean(self.test_input_unmasked), masked_mean(self.test_input, self.test_mask))

    def test_masked_var(self):
        self.assertEqual(torch.var(self.test_input_unmasked), masked_var(self.test_input, self.test_mask))

    def test_masked_whiten(self):
        def whiten(values: torch.Tensor) -> torch.Tensor:
            mean, var = torch.mean(values), torch.var(values)
            return (values - mean) * torch.rsqrt(var + 1e-8)

        whiten_unmasked = whiten(self.test_input_unmasked)
        whiten_masked = masked_whiten(self.test_input, self.test_mask)[1:3]
        diffs = (whiten_unmasked - whiten_masked).sum()
        self.assertLess(abs(diffs.item()), 0.00001)
