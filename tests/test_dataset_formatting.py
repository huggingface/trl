import unittest
from typing import Callable

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.extras.dataset_formatting import get_formatting_func_from_dataset
from trl.models.utils import ChatMlSpecialTokens, setup_chat_format


class DatasetFormattingTestCase(unittest.TestCase):
    def setUp(self):
        self.llama_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.chatml_tokenizer = AutoTokenizer.from_pretrained("philschmid/gpt2-chatml-tokenizer")

    def test_get_formatting_func_from_dataset_with_chatml_messages(self):
        dataset = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi, how can I help you?"},
                    ]
                ]
            }
        )

        # Llama tokenizer
        formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
        assert isinstance(formatting_func, Callable)
        formatted_text = formatting_func(dataset[0])
        expected = "<s>[INST] <<SYS>>\nYou are helpful\n<</SYS>>\n\nHello [/INST] Hi, how can I help you? </s>"
        assert formatted_text == expected
        formatted_text = formatting_func(dataset[0:1])
        assert formatted_text == [expected]

        # ChatML tokenizer
        formatting_func = get_formatting_func_from_dataset(dataset, self.chatml_tokenizer)
        formatted_text = formatting_func(dataset[0])
        expected = "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi, how can I help you?<|im_end|>\n"
        assert formatted_text == expected
        formatted_text = formatting_func(dataset[0:1])
        assert formatted_text == [expected]

    def test_get_formatting_func_from_dataset_with_chatml_conversations(self):
        dataset = Dataset.from_dict(
            {
                "conversations": [
                    [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi, how can I help you?"},
                    ]
                ]
            }
        )
        # Llama tokenizer
        formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
        assert isinstance(formatting_func, Callable)
        formatted_text = formatting_func(dataset[0])
        expected = "<s>[INST] <<SYS>>\nYou are helpful\n<</SYS>>\n\nHello [/INST] Hi, how can I help you? </s>"
        assert formatted_text == expected
        formatted_text = formatting_func(dataset[0:1])
        assert formatted_text == [expected]

        # ChatML tokenizer
        formatting_func = get_formatting_func_from_dataset(dataset, self.chatml_tokenizer)
        formatted_text = formatting_func(dataset[0])
        expected = "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi, how can I help you?<|im_end|>\n"
        assert formatted_text == expected
        formatted_text = formatting_func(dataset[0:1])
        assert formatted_text == [expected]

    def test_get_formatting_func_from_dataset_with_instruction(self):
        dataset = Dataset.from_list(
            [{"prompt": "What is 2+2?", "completion": "4"}, {"prompt": "What is 3+3?", "completion": "6"}]
        )
        formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
        assert formatting_func is not None
        assert isinstance(formatting_func, Callable)
        formatted_text = formatting_func(dataset[0])
        assert formatted_text == "<s>[INST] What is 2+2? [/INST] 4 </s>"
        formatted_text = formatting_func(dataset[0:1])
        assert formatted_text == ["<s>[INST] What is 2+2? [/INST] 4 </s>"]

    def test_get_formatting_func_from_dataset_from_hub(self):
        ds_1 = load_dataset("philschmid/trl-test-instruction", split="train")
        ds_2 = load_dataset("philschmid/dolly-15k-oai-style", split="train")
        for ds in [ds_1, ds_2]:
            formatting_func = get_formatting_func_from_dataset(ds, self.llama_tokenizer)
            assert formatting_func is not None
            assert isinstance(formatting_func, Callable)
        ds_3 = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
        formatting_func = get_formatting_func_from_dataset(ds_3, self.llama_tokenizer)
        assert formatting_func is None

    def test_get_formatting_func_from_dataset_with_unknown_format(self):
        dataset = Dataset.from_dict({"text": "test"})
        formatting_func = get_formatting_func_from_dataset(dataset, self.llama_tokenizer)
        assert formatting_func is None


class SetupChatFormatTestCase(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-MistralForCausalLM")

    def test_setup_chat_format(self):
        original_tokenizer_len = len(self.tokenizer)
        modified_model, modified_tokenizer = setup_chat_format(
            self.model, self.tokenizer, format="chatml", resize_to_multiple_of=64
        )

        _chatml = ChatMlSpecialTokens()
        # Check if special tokens are correctly set
        assert modified_tokenizer.eos_token == "<|im_end|>"
        assert modified_tokenizer.pad_token == "<|im_end|>"
        assert modified_tokenizer.bos_token == "<|im_start|>"
        assert modified_tokenizer.eos_token == _chatml.eos_token
        assert modified_tokenizer.pad_token == _chatml.pad_token
        assert modified_tokenizer.bos_token == _chatml.bos_token
        assert len(modified_tokenizer) == (original_tokenizer_len + 2)
        assert (self.model.get_input_embeddings().weight.shape[0] % 64) == 0
        assert self.model.get_input_embeddings().weight.shape[0] == (original_tokenizer_len + 64)

    def test_example_with_setup_model(self):
        modified_model, modified_tokenizer = setup_chat_format(
            self.model,
            self.tokenizer,
        )
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi, how can I help you?"},
        ]
        prompt = modified_tokenizer.apply_chat_template(messages, tokenize=False)

        assert (
            prompt
            == "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi, how can I help you?<|im_end|>\n"
        )
