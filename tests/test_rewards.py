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

import pickle
import threading

import pytest

from trl.rewards import (
    accuracy_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
    ifeval_reward,
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


class TestIFEvalReward:
    # (instruction_id, kwargs, satisfying_response, violating_response)
    CASES = [
        ("keywords:existence", {"keywords": ["AI"]}, "AI is great", "nothing here"),
        ("keywords:frequency", {"keyword": "AI", "frequency": 2, "relation": "at least"}, "AI and AI", "only AI once"),
        ("keywords:forbidden_words", {"forbidden_words": ["foo"]}, "bar baz", "a foo here"),
        ("keywords:letter_frequency", {"letter": "a", "let_frequency": 3, "let_relation": "at least"}, "banana", "bb"),
        ("length_constraints:number_words", {"num_words": 3, "relation": "at least"}, "one two three", "one two"),
        ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "para one\n***\npara two", "single para"),
        ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] and [date]", "[name] only"),
        ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Body.\nP.S. extra", "Body only"),
        ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n* b", "* a only"),
        ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* and *b*", "*a* only"),
        (
            "detectable_format:multiple_sections",
            {"section_spliter": "Section", "num_sections": 2},
            "Section 1\nx\nSection 2\ny",
            "Section 1\nx",
        ),
        ("detectable_format:title", {}, "<<My Title>>\nbody", "no title"),
        ("detectable_format:json_format", {}, '{"a": 1}', "not json"),
        ("detectable_format:constrained_response", {}, "My answer is yes.", "maybe not"),
        ("startend:end_checker", {"end_phrase": "the end"}, "this is the end", "this is not"),
        ("startend:quotation", {}, '"all quoted"', "not quoted"),
        ("punctuation:no_comma", {}, "no commas here", "a, b, c"),
    ]

    @pytest.mark.parametrize("instruction_id, args, ok, bad", CASES)
    def test_checker_satisfied(self, instruction_id, args, ok, bad):
        rewards = ifeval_reward([[{"content": ok}]], [[instruction_id]], [[args]])
        assert rewards == [1.0]

    @pytest.mark.parametrize("instruction_id, args, ok, bad", CASES)
    def test_checker_violated(self, instruction_id, args, ok, bad):
        rewards = ifeval_reward([[{"content": bad}]], [[instruction_id]], [[args]])
        assert rewards == [0.0]

    def test_fraction_of_satisfied_constraints(self):
        """The reward is the fraction of satisfied constraints."""
        completions = [[{"content": "AI, but with a comma"}]]
        instruction_id_list = [["keywords:existence", "punctuation:no_comma"]]
        instruction_kwargs = [[{"keywords": ["AI"]}, {}]]
        # existence satisfied, no_comma violated -> 1/2
        assert ifeval_reward(completions, instruction_id_list, instruction_kwargs) == [0.5]

    def test_all_constraints_satisfied(self):
        completions = [[{"content": "AI is here"}]]
        instruction_id_list = [["keywords:existence", "punctuation:no_comma"]]
        instruction_kwargs = [[{"keywords": ["AI"]}, {}]]
        assert ifeval_reward(completions, instruction_id_list, instruction_kwargs) == [1.0]

    def test_no_constraints_yields_zero(self):
        assert ifeval_reward([[{"content": "anything"}]], [[]], [[]]) == [0.0]

    def test_missing_optional_arg_uses_default_relation(self):
        """A missing/None `relation` defaults to 'at least'."""
        completions = [[{"content": "one two three four"}]]
        instruction_id_list = [["length_constraints:number_words"]]
        instruction_kwargs = [[{"num_words": 3, "relation": None}]]  # None should be ignored -> "at least"
        assert ifeval_reward(completions, instruction_id_list, instruction_kwargs) == [1.0]

    def test_unsupported_instruction_id_raises(self):
        with pytest.raises(ValueError):
            ifeval_reward([[{"content": "x"}]], [["language:response_language"]], [[{"language": "en"}]])

    def test_batch_of_completions(self):
        completions = [[{"content": "AI is great"}], [{"content": "no keyword"}]]
        instruction_id_list = [["keywords:existence"], ["keywords:existence"]]
        instruction_kwargs = [[{"keywords": ["AI"]}], [{"keywords": ["AI"]}]]
        assert ifeval_reward(completions, instruction_id_list, instruction_kwargs) == [1.0, 0.0]


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
