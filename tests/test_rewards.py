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

import math
import pickle
import threading

import pytest

from trl.rewards import (
    accuracy_reward,
    get_cosine_scaled_reward,
    get_length_scaled_accuracy_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
    graduated_format_reward,
    reasoning_accuracy_reward,
    think_format_reward,
)

from .testing_utils import TrlTestCase, require_math_latex


class TestThinkFormatReward(TrlTestCase):
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
        assert rewards == expected_rewards

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
        assert rewards == expected_rewards

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
        assert rewards == expected_rewards


class TestSoftOverlongPunishmentReward:
    def test_soft_overlong_punishment_short_completion(self):
        """Test soft overlong punishment reward function with a short completion."""
        # length 50, with max=100 and soft cache=20, reward should be 0.
        reward_fn = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
        completion_ids = [[1] * 50]  # 50 <= 80
        rewards = reward_fn(completion_ids=completion_ids)
        assert rewards == [0]

    def test_soft_overlong_punishment_long_completion(self):
        """Test soft overlong punishment reward function with a longer than max completion."""
        # 110 > 100, reward should be -1.
        reward_fn = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
        completion_ids = [[1] * 110]
        rewards = reward_fn(completion_ids)
        assert rewards == [-1]

    def test_soft_overlong_punishment_intermediate_completion(self):
        """Test soft overlong punishment reward function for intermediate length completion."""
        reward_fn = get_soft_overlong_punishment(max_completion_len=100, soft_punish_cache=20)
        completion_ids = [[1] * 90]  # 90 is between 80 and 100
        rewards = reward_fn(completion_ids)
        assert round(abs(rewards[0] - -0.5), 4) == 0


class TestRepetitionPenaltyReward:
    def test_no_repetition_yields_zero(self):
        """A completion with only unique n-grams gets no penalty."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completion_ids = [[1, 2, 3, 4]]
        assert reward_fn(completion_ids) == [0.0]

    def test_full_repetition_approaches_max_penalty(self):
        """A fully repetitive completion approaches max_penalty."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        # [5, 5, 5, 5, 5] -> 4 bigrams, 1 unique -> scaling = 1 - 1/4 = 0.75
        completion_ids = [[5, 5, 5, 5, 5]]
        assert reward_fn(completion_ids) == [pytest.approx(-0.75)]

    def test_partial_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        # [1, 2, 1, 2, 1, 2] -> 5 bigrams, 2 unique -> scaling = 1 - 2/5 = 0.6
        completion_ids = [[1, 2, 1, 2, 1, 2]]
        assert reward_fn(completion_ids) == [pytest.approx(-0.6)]

    def test_completion_shorter_than_ngram_size_yields_zero(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completion_ids = [[1, 2]]  # 2 tokens < ngram_size
        assert reward_fn(completion_ids) == [0.0]

    def test_completion_exactly_ngram_size_yields_zero(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completion_ids = [[1, 2, 3]]  # a single, unique n-gram
        assert reward_fn(completion_ids) == [0.0]

    def test_empty_completion_yields_zero(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completion_ids = [[]]
        assert reward_fn(completion_ids) == [0.0]

    def test_max_penalty_scales_reward(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-0.5)
        # scaling 0.75 * max_penalty -0.5 = -0.375
        completion_ids = [[5, 5, 5, 5, 5]]
        assert reward_fn(completion_ids) == [pytest.approx(-0.375)]

    def test_ngram_size_changes_reward(self):
        completion_ids = [[1, 2, 3, 1, 2, 3]]
        # bigrams: 5 total, 3 unique -> 1 - 3/5 = 0.4
        reward_bigram = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        assert reward_bigram(completion_ids) == [pytest.approx(-0.4)]
        # trigrams: 4 total, 3 unique -> 1 - 3/4 = 0.25
        reward_trigram = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        assert reward_trigram(completion_ids) == [pytest.approx(-0.25)]

    def test_batch_of_completions(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completion_ids = [
            [1, 2, 3, 4],  # no repetition
            [5, 5, 5, 5, 5],  # full repetition
            [9],  # shorter than ngram_size
        ]
        assert reward_fn(completion_ids) == [pytest.approx(0.0), pytest.approx(-0.75), pytest.approx(0.0)]

    def test_positive_max_penalty_raises(self):
        with pytest.raises(ValueError):
            get_repetition_penalty_reward(ngram_size=2, max_penalty=0.5)

    def test_extra_kwargs_are_ignored(self):
        """Trainers pass prompts/completions/etc. as kwargs; the reward must accept and ignore them."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completion_ids = [[5, 5, 5, 5, 5]]
        rewards = reward_fn(completion_ids, prompts=["x"], completions=[[{"content": "5 5 5 5 5"}]])
        assert rewards == [pytest.approx(-0.75)]

    def test_reward_is_picklable(self):
        """The reward must survive pickling for the async GRPO rollout worker."""
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        unpickled = pickle.loads(pickle.dumps(reward_fn))
        completion_ids = [[5, 5, 5, 5, 5]]
        assert unpickled(completion_ids) == [pytest.approx(-0.75)]
        assert unpickled.__name__ == "repetition_penalty_reward"


class TestAccuracyReward:
    @require_math_latex
    def test_accuracy_reward_correct_answer(self):
        """Test accuracy_reward with a correct answer."""
        completion = [[{"content": r"\boxed{\frac{63}{400}}"}], [{"content": r"\boxed{\frac{63}{400}}"}]]
        solution = [r"\frac{63}{400}", "63/400"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0

    @require_math_latex
    def test_accuracy_reward_wrong_answer(self):
        """Test accuracy_reward with an incorrect answer."""
        completion = [[{"content": r"\boxed{\frac{64}{400}}"}]]
        solution = [r"\frac{63}{400}"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    @require_math_latex
    def test_accuracy_reward_wrong_answer_no_latex(self):
        """Test accuracy_reward with an incorrect answer and gold solution with no latex."""
        completion = [[{"content": r"\boxed{3}"}]]
        solution = ["6"]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    @require_math_latex
    def test_accuracy_reward_unparsable_gold(self):
        """Test accuracy_reward with an unparsable gold solution."""
        completion = [
            [{"content": "Answer is forty two."}],
            [{"content": r"Some other content. \boxed{43}."}],
        ]
        solution = [
            "Answer is forty two.",
            "Answer is forty three.",
        ]
        rewards = accuracy_reward(completion, solution)
        assert rewards[0] is None
        assert rewards[1] is None

    @require_math_latex
    def test_accuracy_reward_in_worker_thread(self):
        """Test that accuracy_reward works when called from a non-main thread."""
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}]]
        solutions = [r"\frac{1}{3}"]
        results = []
        exceptions = []

        def target():
            try:
                results.extend(accuracy_reward(completions, solutions))
            except Exception as e:
                exceptions.append(e)

        t = threading.Thread(target=target)
        t.start()
        t.join()

        assert not exceptions, f"accuracy_reward raised in worker thread: {exceptions[0]}"
        assert results == [1.0]


class TestReasoningAccuracyReward:
    @require_math_latex
    def test_correct_answer_yields_unit_reward(self):
        completions = [
            [{"content": r"<think> Reasoning content </think> \boxed{\frac{63}{400}}"}],
            [{"content": r"Reasoning content </think> \boxed{\frac{63}{400}}"}],
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] == 1.0
        assert rewards[1] == 1.0

    @require_math_latex
    def test_correct_answer_with_custom_tags_yields_unit_reward(self):
        completions = [
            [{"content": r"<REASONING_START> Reasoning content </REASONING_END> \boxed{\frac{63}{400}}"}],
        ]
        solutions = [
            r"\frac{63}{400}",
        ]
        rewards = reasoning_accuracy_reward(completions, solutions, reasoning_delimiters=["</REASONING_END>"])
        assert rewards[0] == 1.0

    @require_math_latex
    def test_incorrect_answer_yields_zero_reward(self):
        completion = [[{"content": r"<think> Reasoning content </think> \boxed{\frac{64}{400}}"}]]
        solution = [r"\frac{63}{400}"]
        rewards = reasoning_accuracy_reward(completion, solution)
        assert rewards[0] == 0.0

    @require_math_latex
    def test_correct_answer_in_reasoning_yields_zero_reward(self):
        completions = [
            [{"content": r"<think> My answer is \boxed{42} </think> Some other text."}],
            [{"content": r"<think> The answer is \boxed{42} </think> Here's a wrong answer: \boxed{43}."}],
        ]
        solutions = [r"\boxed{42}", r"\boxed{42}"]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0

    @require_math_latex
    def test_incomplete_reasoning_yields_zero_reward(self):
        completions = [
            [{"content": r"<think> Incomplete reasoning without closing tag"}],
            [{"content": r"Correct answer \frac{63}{400} but completely missing reasoning content"}],
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0

    @require_math_latex
    def test_unparsable_gold_solution_yields_none_reward(self):
        completions = [
            [{"content": r"<think> Reasoning content </think> \boxed{42}"}],
        ]
        solutions = [
            "forty two",
        ]
        rewards = reasoning_accuracy_reward(completions, solutions)
        assert rewards[0] is None


class TestCosineScaledReward:
    @require_math_latex
    def test_correct_shorter_rewarded_more(self):
        """For correct completions, a shorter one gets a higher reward."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 25, [1] * 75]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards[0] > rewards[1]
        assert rewards == [pytest.approx(0.92678, abs=1e-4), pytest.approx(0.57322, abs=1e-4)]

    @require_math_latex
    def test_wrong_longer_penalized_less(self):
        """For wrong completions, a longer one is penalized less (closer to zero)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{2}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 25, [1] * 75]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards[1] > rewards[0]
        assert rewards == [pytest.approx(-0.92678, abs=1e-4), pytest.approx(-0.57322, abs=1e-4)]

    @require_math_latex
    def test_midpoint_values(self):
        """At half of max_len (cosine = 0), correct -> 0.75 and wrong -> -0.75 with default bounds."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 50, [1] * 50]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(0.75), pytest.approx(-0.75)]

    @require_math_latex
    def test_correct_boundary_values(self):
        """Correct: shortest -> max_value_correct (1.0), longest -> min_value_correct (0.5)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[], [1] * 100]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(1.0), pytest.approx(0.5)]

    @require_math_latex
    def test_wrong_boundary_values(self):
        """Wrong: shortest -> min_value_wrong (-1.0), longest -> max_value_wrong (-0.5)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{2}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[], [1] * 100]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(-1.0), pytest.approx(-0.5)]

    @require_math_latex
    def test_length_exceeding_max_len_is_clamped(self):
        """Completions longer than max_len stay at the long-length bound (no climb back up past max_len)."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}], [{"content": r"\boxed{\frac{1}{2}}"}]]
        solution = [r"\frac{1}{3}", r"\frac{1}{3}"]
        completion_ids = [[1] * 200, [1] * 200]  # both 2x max_len
        rewards = reward_fn(completions, solution, completion_ids)
        # correct -> min_value_correct (0.5), wrong -> max_value_wrong (-0.5); same as at exactly max_len
        assert rewards == [pytest.approx(0.5), pytest.approx(-0.5)]

    @require_math_latex
    def test_unparsable_gold_yields_none(self):
        """An unparseable gold solution is skipped, as in accuracy_reward."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": r"\boxed{42}"}]]
        solution = ["forty two"]
        completion_ids = [[1] * 50]
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [None]

    @require_math_latex
    def test_custom_value_bounds(self):
        reward_fn = get_cosine_scaled_reward(max_len=100, min_value_correct=0.0, max_value_correct=2.0)
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}"]
        completion_ids = [[1] * 50]  # progress 0.5, cosine 0 -> 0.0 + 0.5 * (2.0 - 0.0) * 1 = 1.0
        rewards = reward_fn(completions, solution, completion_ids)
        assert rewards == [pytest.approx(1.0)]

    @require_math_latex
    def test_reward_is_picklable(self):
        """The reward must survive pickling for the async GRPO rollout worker."""
        reward_fn = get_cosine_scaled_reward(max_len=100)
        unpickled = pickle.loads(pickle.dumps(reward_fn))
        completions = [[{"content": r"\boxed{\frac{1}{3}}"}]]
        solution = [r"\frac{1}{3}"]
        completion_ids = [[1] * 50]
        assert unpickled(completions, solution, completion_ids) == [pytest.approx(0.75)]
        assert unpickled.__name__ == "cosine_scaled_reward"


class TestGraduatedFormatReward:
    def test_strict_match_yields_full_reward(self):
        completions = [
            [{"content": "<think>\nReasoning.\n</think>\nAnswer."}],
            [{"content": "<think>short</think> answer"}],
            [{"content": "<think></think>\nAnswer."}],
        ]
        rewards = graduated_format_reward(completions)
        assert rewards == [1.0, 1.0, 1.0]

    def test_both_tags_but_malformed_yields_half_reward(self):
        completions = [
            # Closer before opener.
            [{"content": "</think>\nReasoning.\n<think>"}],
            # Nested <think> tags.
            [{"content": "<think>outer <think>inner</think></think> answer"}],
            # Content before the opening <think>.
            [{"content": "prefix <think>reasoning</think> answer"}],
        ]
        rewards = graduated_format_reward(completions)
        assert rewards == [0.5, 0.5, 0.5]

    def test_single_tag_yields_quarter_reward(self):
        completions = [
            [{"content": "<think>reasoning without closing tag"}],
            [{"content": "reasoning without opening tag</think> answer"}],
        ]
        rewards = graduated_format_reward(completions)
        assert rewards == [0.25, 0.25]

    def test_no_tags_yields_zero_reward(self):
        completions = [
            [{"content": "Plain answer with no reasoning tags."}],
            [{"content": ""}],
        ]
        rewards = graduated_format_reward(completions)
        assert rewards == [0.0, 0.0]


class TestLengthScaledAccuracyReward:
    @require_math_latex
    def test_shorter_correct_completion_gets_higher_reward_than_longer(self):
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.5)
        completions = [
            [{"content": r"<think> short </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> much longer reasoning content here </think> \boxed{\frac{1}{3}}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
        prompts = ["same prompt", "same prompt"]
        rewards = reward_fn(completions=completions, solution=solutions, prompts=prompts)
        # Both are correct, so rewards should be within a group and geometric-mean 1.0 around the unit point.
        assert rewards[0] > rewards[1]
        assert math.isclose(rewards[0] * rewards[1], 1.0, rel_tol=1e-6)

    @require_math_latex
    def test_incorrect_completion_gets_incorrect_reward(self):
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.5, incorrect_reward=-1.0)
        completions = [
            [{"content": r"<think> short </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> reasoning </think> \boxed{\frac{1}{2}}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
        prompts = ["same prompt", "same prompt"]
        rewards = reward_fn(completions=completions, solution=solutions, prompts=prompts)
        # Only one correct completion in the group → no length scaling, reward=1.0.
        assert rewards[0] == 1.0
        assert rewards[1] == -1.0

    @require_math_latex
    def test_unparsable_gold_passes_through_as_none(self):
        reward_fn = get_length_scaled_accuracy_reward()
        completions = [
            [{"content": r"<think> Reasoning </think> \boxed{42}"}],
        ]
        solutions = ["forty two"]
        prompts = ["p"]
        rewards = reward_fn(completions=completions, solution=solutions, prompts=prompts)
        assert rewards[0] is None

    @require_math_latex
    def test_groups_are_scaled_independently(self):
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.5)
        # Two prompts, two completions each. Per-group, the shorter correct beats the longer correct.
        completions = [
            [{"content": r"<think> a </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> a much longer chain </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> b </think> \boxed{2}"}],
            [{"content": r"<think> b much longer chain </think> \boxed{2}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}", "2", "2"]
        prompts = ["p1", "p1", "p2", "p2"]
        rewards = reward_fn(completions=completions, solution=solutions, prompts=prompts)
        assert rewards[0] > rewards[1]
        assert rewards[2] > rewards[3]
        # Cross-group rewards are comparable: the "short correct" of each group should match.
        assert math.isclose(rewards[0], rewards[2], rel_tol=1e-6)

    @require_math_latex
    def test_token_length_from_completion_ids_is_used_when_available(self):
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.5)
        # Identical string content but different token counts via completion_ids → shorter ids gets higher reward.
        completions = [
            [{"content": r"<think> reasoning </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> reasoning </think> \boxed{\frac{1}{3}}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
        prompts = ["p", "p"]
        completion_ids = [[1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        rewards = reward_fn(
            completions=completions, solution=solutions, prompts=prompts, completion_ids=completion_ids
        )
        assert rewards[0] > rewards[1]

    @require_math_latex
    def test_alpha_zero_recovers_binary_accuracy(self):
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.0, incorrect_reward=-1.0)
        completions = [
            [{"content": r"<think> short </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> much longer reasoning </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> x </think> \boxed{\frac{1}{2}}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}", r"\frac{1}{3}"]
        prompts = ["p", "p", "p"]
        rewards = reward_fn(completions=completions, solution=solutions, prompts=prompts)
        assert rewards == [1.0, 1.0, -1.0]

    @require_math_latex
    def test_log_extra_emits_upstream_and_length_z_columns(self):
        # Capture the columns the reward function pushes through `log_extra`. The wrapper must preserve the
        # diagnostic columns emitted by `reasoning_accuracy_reward` and add a `length_z` column on top.
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.5)
        completions = [
            [{"content": r"<think> a </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> a much longer chain </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> wrong </think> \boxed{\frac{1}{2}}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}", r"\frac{1}{3}"]
        prompts = ["p", "p", "p"]

        logged: dict[str, list] = {}

        def log_extra(column: str, values: list) -> None:
            logged[column] = values

        reward_fn(completions=completions, solution=solutions, prompts=prompts, log_extra=log_extra)

        assert set(logged) == {"solution", "gold_parsed", "answer_parsed", "length_z"}
        assert logged["solution"] == solutions
        assert len(logged["gold_parsed"]) == len(completions)
        assert len(logged["answer_parsed"]) == len(completions)
        # Two correct completions in the group → both get non-zero z-scores; the incorrect one gets None.
        assert logged["length_z"][0] is not None and logged["length_z"][1] is not None
        assert logged["length_z"][0] < 0 < logged["length_z"][1]
        assert logged["length_z"][2] is None

    @require_math_latex
    def test_reward_is_picklable(self):
        """The reward must survive pickling for the async GRPO rollout worker."""
        reward_fn = get_length_scaled_accuracy_reward(alpha=0.5, incorrect_reward=-1.0)
        unpickled = pickle.loads(pickle.dumps(reward_fn))
        completions = [
            [{"content": r"<think> short </think> \boxed{\frac{1}{3}}"}],
            [{"content": r"<think> reasoning </think> \boxed{\frac{1}{2}}"}],
        ]
        solutions = [r"\frac{1}{3}", r"\frac{1}{3}"]
        prompts = ["p", "p"]
        rewards = unpickled(completions=completions, solution=solutions, prompts=prompts)
        assert rewards == [1.0, -1.0]
        assert unpickled.__name__ == "length_scaled_accuracy_reward"
