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

import unittest

from trl.rewards import think_format_reward


class ThinkFormatRewardTester(unittest.TestCase):
    def test_valid_format(self):
        completions = [
            "<think>This is my reasoning.</think>This is my answer.",  # Simple, one-line reasoning
            "<think>\nThis is my reasoning.\n</think>\nThis is my answer.",  # Multiline reasoning
            "<think>\nThis is\nmy reasoning.\n</think>\nThis is my answer.",  # Multiline reasoning
            "<think>\nThis is <some tag> my reasoning.</think>\nThis is my answer.",  # Reasoning including other tags
            "<think></think>\nThis is my answer.",  # Empty reasoning
        ]
        completions = [[{"content": completion}] for completion in completions]
        expected_rewards = [1.0, 1.0, 1.0, 1.0, 1.0]  # All should be valid
        rewards = think_format_reward(completions)
        self.assertEqual(rewards, expected_rewards)

    def test_invalid_format(self):
        completions = [
            "<think>\nThis is my reasoning.\nThis is my answer.",  # No closing </think>
            "<think>This is my reasoning.\nThis is my answer.",  # No closing </think>
            "This is my reasoning. This is my answer.",  # No <think> tags
            "This is my reasoning.\nThis is my answer.",  # No <think> tags
            "This is my reasoning.</think>\nThis is my answer.",  # No opening <think>
            "This is my reasoning.</think>This is my answer.",  # No opening <think>
            "This<think>is my reasoning.</think>\nThis is my answer.",  # <think> tag in the middle
            "<think>This is<think>my reasoning.</think></think>This is my answer.",  # Nested <think> tags
            "<think>This is</think>\nmy\n<think>reasoning.</think>\nThis is my answer.",  # Multiline <think>
        ]
        completions = [[{"content": completion}] for completion in completions]
        expected_rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # All should be invalid
        rewards = think_format_reward(completions)
        self.assertEqual(rewards, expected_rewards)

    def test_mixed_format(self):
        completions = [
            "<think>This is my reasoning.</think>This is my answer.",  # Valid
            "<think>\nThis is my reasoning.\n</think>\nThis is my answer.",  # Valid
            "<think>This is my reasoning.\nThis is my answer.",  # Invalid
            "This is my reasoning. This is my answer.",  # Invalid
        ]
        completions = [[{"content": completion}] for completion in completions]
        expected_rewards = [1.0, 1.0, 0.0, 0.0]
        rewards = think_format_reward(completions)
        self.assertEqual(rewards, expected_rewards)


if __name__ == "__main__":
    unittest.main()
