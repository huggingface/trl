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

import time
import unittest

from trl import AllTrueJudge, HfPairwiseJudge, PairRMJudge

from .testing_utils import RandomBinaryJudge, require_llm_blender


class TestJudges(unittest.TestCase):
    def _get_prompts_and_pairwise_completions(self):
        prompts = ["The capital of France is", "The biggest planet in the solar system is"]
        completions = [["Paris", "Marseille"], ["Saturn", "Jupiter"]]
        return prompts, completions

    def _get_prompts_and_single_completions(self):
        prompts = ["What's the capital of France?", "What's the color of the sky?"]
        completions = ["Marseille", "blue"]
        return prompts, completions

    def test_all_true_judge(self):
        judge = AllTrueJudge(judges=[RandomBinaryJudge(), RandomBinaryJudge()])
        prompts, completions = self._get_prompts_and_single_completions()
        judgements = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(judgements), 2)
        self.assertTrue(all(judgement in {0, 1, -1} for judgement in judgements))

    @unittest.skip("This test needs to be run manually since it requires a valid Hugging Face API key.")
    def test_hugging_face_judge(self):
        judge = HfPairwiseJudge()
        prompts, completions = self._get_prompts_and_pairwise_completions()
        ranks = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))
        self.assertEqual(ranks, [0, 1])

    def load_pair_rm_judge(self):
        # When using concurrent tests, PairRM may fail to load the model while another job is still downloading.
        # This is a workaround to retry loading the model a few times.
        for _ in range(5):
            try:
                return PairRMJudge()
            except ValueError:
                time.sleep(5)
        raise ValueError("Failed to load PairRMJudge")

    @require_llm_blender
    def test_pair_rm_judge(self):
        judge = self.load_pair_rm_judge()
        prompts, completions = self._get_prompts_and_pairwise_completions()
        ranks = judge.judge(prompts=prompts, completions=completions)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))
        self.assertEqual(ranks, [0, 1])

    @require_llm_blender
    def test_pair_rm_judge_return_scores(self):
        judge = self.load_pair_rm_judge()
        prompts, completions = self._get_prompts_and_pairwise_completions()
        probs = judge.judge(prompts=prompts, completions=completions, return_scores=True)
        self.assertEqual(len(probs), 2)
        self.assertTrue(all(isinstance(prob, float) for prob in probs))
        self.assertTrue(all(0 <= prob <= 1 for prob in probs))
