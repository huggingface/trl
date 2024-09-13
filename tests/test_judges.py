# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from trl import HfPairwiseJudge, PairRMJudge, RandomPairwiseJudge, RandomRankJudge


class TestJudges(unittest.TestCase):
    def _get_prompts_and_completions(self):
        prompts = ["The capital of France is", "The biggest planet in the solar system is"]
        completions = [["Paris", "Marseille"], ["Saturn", "Jupiter"]]
        return prompts, completions

    def test_random_pairwise_judge(self):
        judge = RandomPairwiseJudge()
        prompts, completions = self._get_prompts_and_completions()
        ranks = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))

    def test_random_rank_judge(self):
        judge = RandomRankJudge()
        prompts, completions = self._get_prompts_and_completions()
        ranks = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, list) for rank in ranks))
        self.assertTrue(all(all(isinstance(rank, int) for rank in ranks) for ranks in ranks))

    @unittest.skip("This test needs to be run manually since it requires a valid Hugging Face API key.")
    def test_hugging_face_judge(self):
        judge = HfPairwiseJudge()
        prompts, completions = self._get_prompts_and_completions()
        ranks = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))
        self.assertEqual(ranks, [0, 1])

    def test_pair_rm_judge(self):
        judge = PairRMJudge()
        prompts, completions = self._get_prompts_and_completions()
        ranks = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))
        self.assertEqual(ranks, [0, 1])
