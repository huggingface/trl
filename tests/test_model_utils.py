import unittest

from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.models.utils import ChatMlSpecialTokens, setup_chat_format


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
        assert len(modified_tokenizer) == original_tokenizer_len + 2
        assert self.model.get_input_embeddings().weight.shape[0] % 64 == 0
        assert self.model.get_input_embeddings().weight.shape[0] == original_tokenizer_len + 64
