# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory


class DummyTool:
    def __call__(self, text):
        return text


def dummy_generate(histories):
    for i in range(len(histories)):
        histories[i].append_segment("<request><DummyTool>test<call>", torch.tensor([1, 2, 3]), system=False)
    return histories


class TextHistoryTest(unittest.TestCase):
    def test_text_history_init(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])

        history = TextHistory(text, tokens)
        assert history.text == text
        assert torch.equal(history.tokens, tokens)
        assert torch.equal(history.token_masks, torch.zeros_like(tokens))

        history = TextHistory(text, tokens, system=False)
        assert torch.equal(history.token_masks, torch.ones_like(tokens))

    def test_text_history_append_segment(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])

        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]), system=False)
        assert history.text == (text + "General Kenobi!")
        assert torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6]))
        assert torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1]))

        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]))
        assert history.text == ((text + "General Kenobi!") + "You are a bold one!")
        assert torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        assert torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0]))

    def test_text_history_complete(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.complete()
        assert history.completed
        assert not history.truncated

        history.complete(truncated=True)
        assert history.completed
        assert history.truncated

    def test_text_history_last_segment(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]))
        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]))
        assert history.last_text_segment == "You are a bold one!"

    def test_text_history_split_query_response(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]), system=False)
        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]), system=True)
        query, response, mask = history.split_query_response_tokens()

        assert torch.equal(query, torch.tensor([1, 2, 3]))
        assert torch.equal(response, torch.tensor([4, 5, 6, 7, 8, 9]))
        assert torch.equal(mask, torch.tensor([1, 1, 1, 0, 0, 0]))


class TextEnvironmentTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # model_id
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"

        # get models and tokenizer
        cls.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(cls.model_id)
        cls.gpt2_tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.gpt2_tokenizer.pad_token = cls.gpt2_tokenizer.eos_token

    def test_text_environment_setup(self):
        env = TextEnvironment(
            self.gpt2_model,
            self.gpt2_tokenizer,
            tools=[DummyTool()],
            reward_fn=lambda x: torch.tensor(1),
            prompt="I am a prompt!\n",
        )
        assert env.prompt == "I am a prompt!\n"
        assert list(env.tools.keys()) == ["DummyTool"]
        assert isinstance(env.tools["DummyTool"], DummyTool)
        assert env.reward_fn("Hello there!") == 1

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

        generations_batched = env._generate_batched(model_inputs, batch_size=2)
        generations_batched = self.gpt2_tokenizer.batch_decode(generations_batched)

        generations_single = [env._generate_batched([inputs], batch_size=1)[0] for inputs in model_inputs]
        generations_single = self.gpt2_tokenizer.batch_decode(generations_single)

        assert generations_single == generations_batched

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
        assert tool == "Tool1"
        assert response == "Hello there!"

        tool, response = env.parse_tool_call(string_invalid_request)
        assert tool is None
        assert response is None

        tool, response = env.parse_tool_call(string_invalid_call)
        assert tool is None
        assert response is None

        tool, response = env.parse_tool_call(string_invalid_tool)
        assert tool is None
        assert response is None

        tool, response = env.parse_tool_call(string_invalid_random)
        assert tool is None
        assert response is None

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
        assert (len(history.last_text_segment) - len(env.response_token)) == 100

        env.max_tool_response = 500
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        assert (len(history.last_text_segment) - len(env.response_token)) == 500

        env.max_tool_response = 1001
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        assert (len(history.last_text_segment) - len(env.response_token)) == 1000

        env.max_tool_response = 2000
        history = env.step(TextHistory("<request><dummy>Hello there!<call>", torch.tensor([1, 2, 3])))
        assert (len(history.last_text_segment) - len(env.response_token)) == 1000

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
        assert histories[0].text == (
            ("I am a prompt!\n" + "test") + (1 * "<request><DummyTool>test<call>test<response>")
        )

        env.max_turns = 2
        _, _, _, _, histories = env.run(["test"])
        assert histories[0].text == (
            ("I am a prompt!\n" + "test") + (2 * "<request><DummyTool>test<call>test<response>")
        )

        env.max_turns = 4
        _, _, _, _, histories = env.run(["test"])
        assert histories[0].text == (
            ("I am a prompt!\n" + "test") + (4 * "<request><DummyTool>test<call>test<response>")
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
            assert histories[i].reward == i

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
        assert len(query[0]) == 9
        assert len(query[1]) == 12
        assert len(response[0]) == 14
        assert len(response[1]) == 14
        assert response_mask[0].sum() == (2 * 3)
        # mocked generate always adds 3 toknes
        assert response_mask[1].sum() == (2 * 3)
        # mocked generate always adds 3 toknes
        assert reward[0] == 0
        assert reward[1] == 1
        assert histories[0].text == (
            ("I am a prompt!\n" + "Hello there!") + (2 * "<request><DummyTool>test<call>test<response>")
        )
        assert histories[1].text == (
            ("I am a prompt!\n" + "Hello there! General Kenobi!")
            + (2 * "<request><DummyTool>test<call>test<response>")
        )
