import random
from abc import ABC, abstractmethod
from typing import List


class BaseJudge(ABC):
    """Base class for local judges."""

    def shuffle_pairs(self, pairs: List[List[str]]):
        """Shuffles each pair of completions to mitigate positional bias."""
        shuffled_pairs = []
        for pair in pairs:
            shuffled_pair = pair.copy()
            random.shuffle(shuffled_pair)
            shuffled_pairs.append(shuffled_pair)
        return shuffled_pairs

    @abstractmethod
    def judge_single(self, prompt: str, completion_pair: List[str]) -> int:
        raise NotImplementedError("Judge subclasses must implement this method.")

    def judge_batch(self, prompts: List[str], completion_pairs: List[List[str]]) -> List[int]:
        results = []
        shuffled_completion_pairs = self.shuffle_pairs(completion_pairs)
        for prompt, completion_pair in zip(prompts, shuffled_completion_pairs):
            result = self.judge_single(prompt, completion_pair)
            results.append(result)
        return results


class MockJudge(BaseJudge):
    def judge_single(self, prompt: str, completion_pair: List[str]) -> int:
        return random.choice([0, 1])
