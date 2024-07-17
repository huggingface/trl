import unittest

from trl import HuggingFaceJudge, MockAPIJudge, MockJudge


class TestJudges(unittest.TestCase):
    def _get_prompts_and_completion_pairs(self):
        prompts = ["What is the capital of France?", "What is the biggest planet in the solar system?"]
        completion_pairs = [["Paris", "Marseille"], ["Saturn", "Jupiter"]]
        return prompts, completion_pairs

    def test_mock_judge(self):
        judge = MockJudge()
        prompts, completion_pairs = self._get_prompts_and_completion_pairs()
        ranks = judge.judge(prompts=prompts, completion_pairs=completion_pairs)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))

    def test_mock_api_judge(self):
        judge = MockAPIJudge()
        prompts, completion_pairs = self._get_prompts_and_completion_pairs()
        ranks = judge.judge(prompts=prompts, completion_pairs=completion_pairs)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))

    def test_hugging_face_judge(self):
        judge = HuggingFaceJudge()
        prompts, completion_pairs = self._get_prompts_and_completion_pairs()
        ranks = judge.judge(prompts=prompts, completion_pairs=completion_pairs)
        self.assertEqual(len(ranks), 2)
        self.assertTrue(all(isinstance(rank, int) for rank in ranks))
        self.assertEqual(ranks, [0, 1])
