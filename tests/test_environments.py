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

from trl import TextEnvironment, TextHistory, AutoModelForCausalLMWithValueHead
import torch
import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer


class DummyTool:
    def __call__(text):
        return text 


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
        self.assertEqual(history.text, text + "General Kenobi!")
        self.assertTrue(torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6])))
        self.assertTrue(torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1])))

        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]))
        self.assertEqual(history.text, text + "General Kenobi!" + "You are a bold one!")
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

    def test_text_history_split_query_response(self):
        text = "Hello there!"
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment("General Kenobi!", torch.tensor([4, 5, 6]), system=False)
        history.append_segment("You are a bold one!", torch.tensor([7, 8, 9]), system=True)
        query, response, mask = history.split_query_response_tokens()
        
        self.assertTrue(torch.equal(query, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(response,torch.tensor([4, 5, 6, 7, 8, 9])))
        self.assertTrue(torch.equal(mask, torch.tensor([1, 1, 1, 0, 0, 0])))


class TextEnvironmentTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # model_id
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"

        # get models and tokenizer
        cls.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(cls.model_id)
        cls.gpt2_tokenizer = AutoTokenizer.from_pretrained(cls.model_id)

    def test_text_environment_setup(self):
        env = TextEnvironment(self.gpt2_model, self.gpt2_tokenizer, tools=[DummyTool()], reward_fn=lambda x: torch.tensor(1), prompt="I am a prompt!\n")
        self.assertEqual(env.prompt, "I am a prompt!\n")
        self.assertEqual(list(env.tools.keys()), ["DummyTool"])
        self.assertTrue(isinstance(env.tools["DummyTool"], DummyTool))
        self.assertEqual(env.reward_fn("Hello there!"), 1)

    def test_text_environment_generate(self):
        pass

    def test_text_environment_run(self):
        pass

    def test_text_environment_tool_call_parsing(self):
        pass

    def test_text_environment_tool_truncation(self):
        pass

    def test_text_environment_max_calls(self):
        pass

    def test_text_environment_max_length(self):
        pass

    def test_text_environment_commpute_rewards(self):
        pass
   