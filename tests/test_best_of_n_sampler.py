import unittest

import torch
from transformers import AutoTokenizer, GenerationConfig

from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler


def queries_to_scores(list_of_strings):
    return [torch.rand(1).item() for _ in list_of_strings]


class BestOfNSamplerTester(unittest.TestCase):
    """
    Tests the BestOfNSampler class
    """

    ref_model_name = "trl-internal-testing/dummy-GPT2-correct-vocab"
    output_length_sampler = LengthSampler(2, 6)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ref_model_name)
    tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    output_length_sampler = LengthSampler(2, 6)

    def test_different_input_types(self):
        r"""
        Tests if the different input types normalizer works
        """

        generation_config = GenerationConfig(
            min_length=-1,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_length_sampler = LengthSampler(2, 6)

        best_of_n = BestOfNSampler(
            self.model,
            self.tokenizer,
            queries_to_scores,
            length_sampler=output_length_sampler,
            generation_config=generation_config,
        )

        queries = ["hello world", "goodbye world"]
        tokenized_queries = [self.tokenizer.encode(query) for query in queries]

        various_queries_formats = [
            (tokenized_queries[0], 1),
            (tokenized_queries, 2),
            (torch.tensor(tokenized_queries[1]), 1),
            ([torch.tensor(query) for query in tokenized_queries], 2),
        ]

        for q, expected_length in various_queries_formats:
            results = best_of_n.generate(q)
            self.assertIsInstance(results, list)
            assert len(results) == expected_length

    def test_different_sample_sizes_and_n_candidates_values(self):
        r"""
        Tests different sample sizes and n_candidates values
        """
        generation_config = GenerationConfig(
            min_length=-1,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        output_length_sampler = LengthSampler(6, 10)

        for sample_value, n_candidates_values, expected in [
            (4, 2, 2),
            (10, 3, 3),
            (6, 4, 4),
        ]:
            best_of_n = BestOfNSampler(
                self.model,
                self.tokenizer,
                queries_to_scores,
                length_sampler=output_length_sampler,
                generation_config=generation_config,
                sample_size=sample_value,
                n_candidates=n_candidates_values,
            )

            queries = ["hello world", "troll the world"]
            tokenized_queries = [self.tokenizer.encode(query) for query in queries]
            results = best_of_n.generate(tokenized_queries)
            for result in results:
                assert len(result) == expected
