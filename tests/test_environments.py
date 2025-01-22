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
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, DynamicCache

from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory


class DummyTool:
    def __call__(self, text):
        return text


def dummy_generate(
    histories, past_key_values=None, past_attention_masks=None, past_input_ids=None, last_active_histories=None
):
    for i in range(len(histories)):
        histories[i].append_segment("<request><DummyTool>test<call>", torch.tensor([1, 2, 3]), system=False)
    return histories, None, None, None, None


def reshape_cache(cache):
    new_cache = []
    for layer in cache:
        keys, values = layer
        keys = keys.reshape((-1, 1, 1, 1))
        values = values.reshape((-1, 1, 1, 1))
        new_cache.append((keys, values))
    return tuple(new_cache)


class TextHistoryTest(unittest.TestCase):
    def test_text_history_init(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])

        history = TextHistory(text, tokens)
        self.assertEqual(history.text, text)
        self.assertTrue(torch.equal(history.tokens, tokens))
        self.assertTrue(torch.equal(history.token_masks, torch.zeros_like(tokens)))

        history = TextHistory(text, tokens, system=False)
        self.assertTrue(torch.equal(history.token_masks, torch.ones_like(tokens)))

    def test_text_history_append_segment(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])

        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]), system=False)
        self.assertEqual(history.text, (text + "General Kenobi!"))
        self.assertTrue(torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6])))
        self.assertTrue(torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1])))

        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]))
        self.assertEqual(history.text, ((text + "General Kenobi!") + "You are a bold one!"))
        self.assertTrue(torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])))
        self.assertTrue(torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0])))

    def test_text_history_complete(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.complete()
        self.assertTrue(history.completed)
        self.assertFalse(history.truncated)

        history.complete(truncated=True)
        self.assertTrue(history.completed)
        self.assertTrue(history.truncated)

    def test_text_history_last_segment(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]))
        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]))
        self.assertEqual(history.last_text_segment, "You are a bold one!")
        self.assertTrue(torch.all(history.last_token_segment == torch.tensor([7, 8, 9])).item())

    def test_text_history_split_query_response(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]), system=False)
        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]), system=True)
        query, response, mask = history.split_query_response_tokens()

        self.assertTrue(torch.equal(query, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(response, torch.tensor([4, 5, 6, 7, 8, 9])))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 1, 0, 0, 0])))


class TextEnvironmentTester(unittest.TestCase):
    def setUp(self):
        # model_id
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

        # get models and tokenizer
        self.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id)
        self.gpt2_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

    def test_text_environment_setup(self):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools=[DummyTool()],
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
        )
        self.assertEqual(env.prompt, "I am a prompt!\n")
        self.assertListEqual(list(env.tools.keys()), ["DummyTool"])
        self.assertIsInstance(env.tools["DummyTool"], DummyTool)
        self.assertEqual(env.reward_fn("Hello there!"), 1)

    def test_text_environment_generate(self):
        generation_kwargs = {"do_sample": False, "max_new_tokens": 4, "pad_token_id": self.gpt2_tokenizer.eos_token_id}
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools=[DummyTool()],
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
            generation_kwargs=generation_kwargs,
        )

        input_texts = ["this is a test", "this is another, longer test"]

        model_inputs = [self.gpt2_tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts]

        generations_batched, _, _, _, _ = env._generate_batched(model_inputs, batch_size=2)
        generations_batched = self.gpt2_tokenizer.batch_decode(generations_batched)

        generations_single = [env._generate_batched([inputs], batch_size=1)[0][0] for inputs in model_inputs]
        generations_single = self.gpt2_tokenizer.batch_decode(generations_single)

        self.assertEqual(generations_single, generations_batched)

    def test_text_environment_tool_call_parsing(self):
        string_valid = "Something something <request><Tool1>Hello there!<call>"
        string_invalid_request = "Something something <Tool1>Hello there!<call>"
        string_invalid_call = "Something something <request><Tool1>Hello there!"
        string_invalid_tool = "Something something <request>|Tool2|Hello there!<call>"
        string_invalid_random = "<>abcdefghijklm<>nopqrstuvwxyz<>"

        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools=[DummyTool()],
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
        )
        tool, response = env.parse_tool_call(string_valid)
        self.assertEqual(tool, "Tool1")
        self.assertEqual(response, "Hello there!")

        tool, response = env.parse_tool_call(string_invalid_request)
        self.assertIsNone(tool)
        self.assertIsNone(response)

        tool, response = env.parse_tool_call(string_invalid_call)
        self.assertIsNone(tool)
        self.assertIsNone(response)

        tool, response = env.parse_tool_call(string_invalid_tool)
        self.assertIsNone(tool)
        self.assertIsNone(response)

        tool, response = env.parse_tool_call(string_invalid_random)
        self.assertIsNone(tool)
        self.assertIsNone(response)

    def test_text_environment_tool_truncation(self):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools={"dummy": lambda x: "a" * 1000},
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
        )

        env.max_tool_response = 100
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        self.assertEqual((len(history.last_text_segment) - len(env.response_token)), 100)

        env.max_tool_response = 500
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        self.assertEqual((len(history.last_text_segment) - len(env.response_token)), 500)

        env.max_tool_response = 1001
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        self.assertEqual((len(history.last_text_segment) - len(env.response_token)), 1000)

        env.max_tool_response = 2000
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        self.assertEqual((len(history.last_text_segment) - len(env.response_token)), 1000)

    @patch.object(TextEnvironment, "generate", side_effect=dummy_generate)
    def test_text_environment_max_calls(self, mock_generate):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools={"DummyTool": DummyTool()},
            reward_fn=lambda x: [torch.tensor(1) for _ in x],
            prompt="I am a prompt!\n",
        )

        env.max_turns = 1
        _, _, _, _, histories = env.run(["test"])
        self.assertEqual(
            histories[0].text,
            ("I am a prompt!\n" + "test") + (1 * "<request><DummyTool>test<call>test<response>"),
        )

        env.max_turns = 2
        _, _, _, _, histories = env.run(["test"])
        self.assertEqual(
            histories[0].text,
            ("I am a prompt!\n" + "test") + (2 * "<request><DummyTool>test<call>test<response>"),
        )

        env.max_turns = 4
        _, _, _, _, histories = env.run(["test"])
        self.assertEqual(
            histories[0].text,
            ("I am a prompt!\n" + "test") + (4 * "<request><DummyTool>test<call>test<response>"),
        )

    def test_text_environment_compute_rewards(self):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools={"DummyTool": DummyTool()},
            reward_fn=lambda x: [torch.tensor(i) for i, _ in enumerate(x)],
            prompt="I am a prompt!\n",
        )

        histories = [TextHistory("<request><DummyTool>test<call>", torch.tensor([1, 2, 3])) for _ in range(8)]
        histories = env.compute_reward(histories)

        for i in range(8):
            self.assertEqual(histories[i].reward, i)

    @patch.object(TextEnvironment, "generate", side_effect=dummy_generate)
    def test_text_environment_run(self, mock_generate):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools={"DummyTool": DummyTool()},
            reward_fn=lambda x: [torch.tensor(i) for i, _ in enumerate(x)],
            prompt="I am a prompt!\n",
            max_turns=2,
        )
        task_1 = "Hello there!"
        task_2 = "Hello there! General Kenobi!"

        query, response, response_mask, reward, histories = env.run([task_1, task_2])
        self.assertEqual(len(query[0]), 8)
        self.assertEqual(len(query[1]), 12)
        self.assertEqual(len(response[0]), 14)
        self.assertEqual(len(response[1]), 14)
        self.assertEqual(response_mask[0].sum(), (2 * 3))
        # mocked generate always adds 3 toknes
        self.assertEqual(response_mask[1].sum(), (2 * 3))
        # mocked generate always adds 3 toknes
        self.assertEqual(reward[1], 1)
        self.assertEqual(
            histories[0].text,
            ("I am a prompt!\n" + "Hello there!") + (2 * "<request><DummyTool>test<call>test<response>"),
        )
        self.assertEqual(
            histories[1].text,
            ("I am a prompt!\n" + "Hello there! General Kenobi!")
            + (2 * "<request><DummyTool>test<call>test<response>"),
        )

    def test_combine_cache(self):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools={"DummyTool": DummyTool()},
            reward_fn=lambda x: [torch.tensor(i) for i, _ in enumerate(x)],
            prompt="I am a prompt!\n",
            max_turns=2,
        )

        caches = [
            (
                (torch.tensor([[[[1], [13]]], [[[2], [14]]]]), torch.tensor([[[[3], [15]]], [[[4], [16]]]])),
                (torch.tensor([[[[7], [17]]], [[[8], [18]]]]), torch.tensor([[[[9], [19]]], [[[10], [20]]]])),
            ),
            (
                (torch.tensor([[[[5]]]]), torch.tensor([[[[6]]]])),
                (torch.tensor([[[[11]]]]), torch.tensor([[[[12]]]])),
            ),
        ]
        caches = [DynamicCache().from_legacy_cache(cache) for cache in caches]
        attention_masks = [torch.tensor([[-1, 1, 7], [1, 0, 8]]), torch.tensor([[2, 4]])]
        input_ids = [torch.tensor([[1, 4, 7], [2, 5, 8]]), torch.tensor([[3, 6]])]
        example_mask = [True, False, True]

        expected_cache = (
            (torch.tensor([[[[1], [13]]], [[[0], [5]]]]), torch.tensor([[[[3], [15]]], [[[0], [6]]]])),
            (torch.tensor([[[[7], [17]]], [[[0], [11]]]]), torch.tensor([[[[9], [19]]], [[[0], [12]]]])),
        )
        expected_attention_mask = torch.tensor([[-1, 1, 7], [0, 2, 4]])
        expected_input_ids = torch.tensor([[1, 4, 7], [self.gpt2_tokenizer.pad_token_id, 3, 6]])

        combined_cache, combined_attention_masks, combined_input_ids = env._combine_cache(
            example_mask, caches, attention_masks, input_ids
        )

        self.assertEqual(len(combined_cache), len(expected_cache))
        self.assertEqual(len(combined_cache[0]), len(expected_cache[0]))
        self.assertTrue(torch.all(combined_cache[0][0] == expected_cache[0][0]))
        self.assertTrue(torch.all(combined_cache[0][1] == expected_cache[0][1]))
        self.assertEqual(len(combined_cache[1]), len(expected_cache[1]))
        self.assertTrue(torch.all(combined_cache[1][0] == expected_cache[1][0]))
        self.assertTrue(torch.all(combined_cache[1][1] == expected_cache[1][1]))
        self.assertTrue(torch.all(combined_attention_masks == expected_attention_mask))
        self.assertTrue(torch.all(combined_input_ids == expected_input_ids))

    def test_get_batched_cache(self):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools={"DummyTool": DummyTool()},
            reward_fn=lambda x: [torch.tensor(i) for i, _ in enumerate(x)],
            prompt="I am a prompt!\n",
            max_turns=2,
        )

        cache = reshape_cache(
            (
                (torch.tensor([[1], [2], [3]]), torch.tensor([[4], [5], [6]])),
                (torch.tensor([[7], [8], [9]]), torch.tensor([[10], [11], [12]])),
            )
        )
        attention_masks = torch.tensor([[1], [2], [3]])
        input_ids = torch.tensor([[4], [5], [6]])
        batched_cache, batched_attention_masks, batched_input_ids = env._get_batched_cache(
            1, 3, cache, attention_masks, input_ids
        )
        batched_cache = batched_cache.to_legacy_cache()
        expected_cache = reshape_cache(
            (
                (torch.tensor([[2], [3]]), torch.tensor([[5], [6]])),
                (torch.tensor([[8], [9]]), torch.tensor([[11], [12]])),
            )
        )

        self.assertEqual(len(batched_cache), len(expected_cache))
        self.assertEqual(len(batched_cache[0]), len(expected_cache[0]))
        self.assertTrue(torch.all(batched_cache[0][0] == expected_cache[0][0]))
        self.assertTrue(torch.all(batched_cache[0][1] == expected_cache[0][1]))
        self.assertEqual(len(batched_cache[1]), len(expected_cache[1]))
        self.assertTrue(torch.all(batched_cache[1][0] == expected_cache[1][0]))
        self.assertTrue(torch.all(batched_cache[1][1] == expected_cache[1][1]))

        expected_attention_mask = torch.tensor([[2], [3]])
        self.assertTrue(torch.all(batched_attention_masks == expected_attention_mask))

        expected_input_ids = torch.tensor([[5], [6]])
        self.assertTrue(torch.all(batched_input_ids == expected_input_ids))

    def test_cached_generate_batched(self):
        generation_kwargs = {"do_sample": False, "max_new_tokens": 4, "pad_token_id": self.gpt2_tokenizer.eos_token_id}
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools=[DummyTool()],
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
            generation_kwargs=generation_kwargs,
        )

        input_texts = ["this is a test", "this is another, longer test", "some other batch", "something unnecessary"]
        model_inputs = [self.gpt2_tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts]
        outputs, past_key_values, past_attention_masks, past_input_ids, _ = env._generate_batched(
            model_inputs, batch_size=2
        )

        past_key_values, past_attention_masks, past_input_ids = env._combine_cache(
            [True, True, True, False], past_key_values, past_attention_masks, past_input_ids
        )

        input_texts2 = [" short interim", " a somewhat longer section in between"]
        model_inputs2 = [self.gpt2_tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts2]
        # for single token query
        model_inputs2.append(
            torch.tensor([self.gpt2_tokenizer(" a", return_tensors="pt").input_ids], dtype=model_inputs2[0].dtype)
        )
        outputs_cached, _, _, _, _, all_logits_cached = env._generate_batched(
            model_inputs2,
            batch_size=2,
            combined_past_key_values=past_key_values,
            combined_past_attention_masks=past_attention_masks,
            combined_past_input_ids=past_input_ids,
            output_logits=True,
        )

        model_inputs2_full = [
            torch.concat([in1, out1, in2], dim=0) for in1, out1, in2 in zip(model_inputs[:-1], outputs, model_inputs2)
        ]
        outputs_uncached, _, _, _, _, all_logits_uncached = env._generate_batched(
            model_inputs2_full, batch_size=2, output_logits=True
        )
        for cached, uncached, logits_cached, logits_uncached in zip(
            outputs_cached, outputs_uncached, all_logits_cached, all_logits_uncached
        ):
            self.assertTrue(torch.all(cached == uncached))
            self.assertEqual(logits_cached.shape[0], 4)
            self.assertEqual(logits_uncached.shape[0], 4)
            self.assertTrue(torch.all(torch.abs(logits_cached - logits_uncached) < 1e-6))

    def test_different_sequence_lengths(self):
        generation_kwargs = {"do_sample": False, "max_new_tokens": 4, "pad_token_id": self.gpt2_tokenizer.eos_token_id}
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools=[DummyTool()],
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
            generation_kwargs=generation_kwargs,
        )

        input_texts = ["this is a test", "this is another, longer test", "some other batch"]
        model_inputs = [self.gpt2_tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts]
        outputs, past_key_values, past_attention_masks, past_input_ids, _ = env._generate_batched(
            model_inputs, batch_size=2
        )
        # remove the last two tokens from the second batch to pretend they were never generated
        second_cache = past_key_values[1].to_legacy_cache()
        edited_cache = []
        for layer in second_cache:
            keys, values = layer
            new_keys = keys[:, :, :-2, :]
            new_values = values[:, :, :-2, :]
            edited_cache.append((new_keys, new_values))

        past_key_values[1] = DynamicCache().from_legacy_cache(tuple(edited_cache))
        past_attention_masks[1] = past_attention_masks[1][:, :-2]
        past_input_ids[1] = past_input_ids[1][:, :-2]

        # ensure this actually removes generated tokens and not skipped tokens / padding
        self.assertEqual(len(outputs[2]), 4)

        past_key_values, past_attention_masks, past_input_ids = env._combine_cache(
            [True, True, True], past_key_values, past_attention_masks, past_input_ids
        )

        self.assertEqual(past_attention_masks.shape, past_input_ids.shape)
        self.assertEqual(past_key_values[0][0].shape[2], past_attention_masks.shape[1] - 1)
        self.assertEqual(past_key_values[0][0].shape[0], past_attention_masks.shape[0])
        input_texts2 = [" short interim", " a somewhat longer section in between"]
        model_inputs2 = [self.gpt2_tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts2]
        # for single token query
        model_inputs2.append(
            torch.tensor([self.gpt2_tokenizer(" a", return_tensors="pt").input_ids], dtype=model_inputs2[0].dtype)
        )
        outputs_cached, _, _, _, _, all_logits_cached = env._generate_batched(
            model_inputs2,
            batch_size=2,
            combined_past_key_values=past_key_values,
            combined_past_attention_masks=past_attention_masks,
            combined_past_input_ids=past_input_ids,
            output_logits=True,
        )
        outputs[2] = outputs[2][:-2]  # remove last two generated tokens from input
        model_inputs2_full = [
            torch.concat([in1, out1, in2], dim=0) for in1, out1, in2 in zip(model_inputs, outputs, model_inputs2)
        ]
        outputs_uncached, _, _, _, _, all_logits_uncached = env._generate_batched(
            model_inputs2_full, batch_size=2, output_logits=True
        )
        for cached, uncached, logits_cached, logits_uncached in zip(
            outputs_cached, outputs_uncached, all_logits_cached, all_logits_uncached
        ):
            self.assertTrue(torch.all(cached == uncached))
            self.assertEqual(logits_cached.shape[0], 4)
            self.assertEqual(logits_uncached.shape[0], 4)
            self.assertTrue(torch.all(torch.abs(logits_cached - logits_uncached) < 1e-6))


if __name__ == "__main__":
    pass
