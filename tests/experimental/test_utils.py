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


import pytest
import torch
from datasets import Dataset, load_dataset
from torch import nn
from transformers import AutoTokenizer

from trl.data_utils import apply_chat_template
from trl.experimental.utils import (
    DataCollatorForChatML,
    get_reward,
    get_reward_from_policy_tokens,
    prepare_peft_model,
    truncate_dataset,
)

from ..testing_utils import TrlTestCase, require_bitsandbytes, require_peft, require_torch_accelerator


class TestDataCollatorForChatML(TrlTestCase):
    def setup_method(self):
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define token IDs
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        # Token ID for "true", the last assistant's response in the example:
        self.ignore_index = -100
        self.max_length = 1024
        self.messages_key = "messages"

        # Example input
        dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
        self.examples = dataset.to_list()

        # Initialize the data collator
        self.collator = DataCollatorForChatML(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            ignore_index=self.ignore_index,
        )

    def test_data_collator_for_chatml(self):
        # Process the data
        data = self.collator(self.examples)

        # Verify basic shapes and types
        assert "input_ids" in data
        assert "attention_mask" in data
        assert "labels" in data
        assert "prompts" in data
        assert "prompt_attention_mask" in data

        # Decode input_ids and labels for verification
        input_ids = data["input_ids"][0].tolist()
        labels = data["labels"][0].tolist()
        prompt_only = data["prompts"][0].tolist()

        # Get the last assistant's response for comparison
        last_message = self.examples[0][self.messages_key][-1]
        assert last_message["role"] == "assistant", "Last message should be from assistant"
        last_assistant_response = last_message["content"]

        # Verify that input_ids contain both prompt and response
        decoded_input = self.tokenizer.decode(input_ids)
        assert last_assistant_response in decoded_input, "Input should contain assistant's response"

        # Verify that prompts only contain the conversation up to the last response
        decoded_prompt = self.tokenizer.decode(prompt_only)
        assert last_assistant_response not in decoded_prompt, "Prompt should not contain assistant's response"

        # Verify labels are -100 for non-assistant parts
        prompt_length = len(prompt_only)
        assert all(label == self.ignore_index for label in labels[:prompt_length]), (
            "Labels should be ignore_index for prompt tokens"
        )

        # Verify labels match assistant response after prompt
        # Add a filter to remove any trailing tokens after the first <|im_end|>
        last_assistant_response_with_end = last_assistant_response + self.tokenizer.eos_token
        last_assistant_response_tokens = self.tokenizer.encode(
            last_assistant_response_with_end, add_special_tokens=False
        )

        response_labels = []
        for label in labels[prompt_length:]:
            if label == self.ignore_index:
                continue
            response_labels.append(label)
            if label == self.tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break
        assert response_labels == last_assistant_response_tokens, "Labels should match assistant response tokens"

        # Verify there isn't a generation prompt at the end
        generation_prompt = "<|im_start|>assistant"
        assert not decoded_input.strip().endswith(generation_prompt), (
            f"Input should not end with generation prompt '{generation_prompt}'"
        )

        assert response_labels == last_assistant_response_tokens, "Labels should match assistant response tokens"


class TestTruncateExamples(TrlTestCase):
    def test_with_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples)
        dataset = dataset.with_format("numpy", dtype="float32")
        format = dataset.format
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        assert dataset.to_dict() == expected_output
        assert format == dataset.format

    def test_with_iterable_dataset(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
        }
        dataset = Dataset.from_dict(examples).to_iterable_dataset()
        dataset = dataset.with_format("numpy")
        formatting = dataset._formatting
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
        }
        dataset = truncate_dataset(dataset, max_length)
        num_examples = len(examples[next(iter(examples))])
        assert next(iter(dataset.with_format(None).batch(batch_size=num_examples))) == expected_output
        assert formatting == dataset._formatting

    def test_with_extra_column(self):
        examples = {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8]],
            "attention_mask": [[0, 1, 1], [0, 0, 1, 1], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = Dataset.from_dict(examples)
        max_length = 2
        expected_output = {
            "input_ids": [[1, 2], [4, 5], [8]],
            "attention_mask": [[0, 1], [0, 0], [1]],
            "my_column": ["a", "b", "c"],
        }
        dataset = truncate_dataset(dataset, max_length)
        assert dataset.to_dict() == expected_output


class TestPreparePeftModel(TrlTestCase):
    @require_peft
    @require_bitsandbytes
    @require_torch_accelerator
    def test_qlora_bf16_yields_uniform_dtype(self):
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            ),
            dtype=torch.bfloat16,
        )
        args = TrainingArguments(output_dir=self.tmp_dir, bf16=True, report_to=[])
        peft_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = prepare_peft_model(model, peft_config, args)

        fp32 = [name for name, param in model.named_parameters() if param.dtype == torch.float32]
        assert fp32 == [], f"expected no float32 params after prepare_peft_model, got e.g. {fp32[:5]}"


class _TinyBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,
    ):
        hidden = self.embed(input_ids)

        class Output:
            pass

        output = Output()
        output.hidden_states = (hidden, hidden)
        return output


class _TinyRewardModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=8):
        super().__init__()
        self.base_model_prefix = "model"
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": hidden_size})()
        self.model = _TinyBackbone(vocab_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)


class _IndexReportingBackbone(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,
    ):
        hidden = self.embed(input_ids).clone()
        token_index = torch.arange(input_ids.size(1), device=input_ids.device, dtype=hidden.dtype)
        hidden[..., 0] = token_index
        output = type("Output", (), {})()
        output.hidden_states = (hidden, hidden)
        return output


class _IndexReportingRewardModel(nn.Module):
    """`score` returns the token index so tests can see which position was selected."""

    def __init__(self, vocab_size, hidden_size=8):
        super().__init__()
        self.base_model_prefix = "model"
        self.config = type("Config", (), {"vocab_size": vocab_size, "hidden_size": hidden_size})()
        self.model = _IndexReportingBackbone(vocab_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        with torch.no_grad():
            self.score.weight.zero_()
            self.score.weight[0, 0] = 1.0


class TestGetRewardFromPolicyTokens(TrlTestCase):
    def setup_method(self):
        self.policy_tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.reward_tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-LlamaForCausalLM-3.2")
        if self.policy_tokenizer.pad_token is None:
            self.policy_tokenizer.pad_token = self.policy_tokenizer.eos_token
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        reward_vocab = max(self.reward_tokenizer.vocab_size, len(self.reward_tokenizer))
        self.reward_model = _TinyRewardModel(reward_vocab)

    def test_retokenizes_policy_ids_outside_reward_vocab(self):
        prompt = "hello"
        completion = " world"
        prompt_ids = self.policy_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        completion_ids = self.policy_tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        query_responses = torch.cat([prompt_ids, completion_ids], dim=1)
        context_length = prompt_ids.shape[1]

        oob_id = self.reward_model.config.vocab_size
        if oob_id < self.policy_tokenizer.vocab_size:
            query_responses = query_responses.clone()
            query_responses[0, -1] = oob_id
            with pytest.raises((IndexError, RuntimeError)):
                get_reward(self.reward_model, query_responses, self.reward_tokenizer.pad_token_id, context_length)

        scores = get_reward_from_policy_tokens(
            self.reward_model,
            query_responses,
            context_length,
            [prompt],
            self.policy_tokenizer,
            self.reward_tokenizer,
        )
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()

    def test_conversational_prompts(self):
        prompt = [{"role": "user", "content": "Hi"}]
        completion = "Hello"
        completion_ids = self.policy_tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        context_length = 1
        query_responses = torch.cat([torch.zeros((1, context_length), dtype=torch.long), completion_ids], dim=1)

        scores = get_reward_from_policy_tokens(
            self.reward_model,
            query_responses,
            context_length,
            [prompt],
            self.policy_tokenizer,
            self.reward_tokenizer,
        )
        assert scores.shape == (1,)
        assert torch.isfinite(scores).all()

    def test_conversational_scores_last_token_not_prompt_eos(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id == tokenizer.eos_token_id

        prompt = [{"role": "user", "content": "Hi"}]
        completion = "Hello there"
        example = {"messages": prompt + [{"role": "assistant", "content": completion}]}
        text = apply_chat_template(example, tokenizer)["text"]
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        token_ids = encoded["input_ids"][0]
        pad_positions = (token_ids == tokenizer.pad_token_id).nonzero(as_tuple=False)
        last_index = token_ids.size(0) - 1
        assert pad_positions.numel() > 0, "chat template must emit eos/pad at turn boundaries for this test"
        first_pad_index = int(pad_positions[0])
        assert first_pad_index < last_index, "first pad/eos must not be the completion end"

        vocab = max(tokenizer.vocab_size, len(tokenizer), int(token_ids.max()) + 1)
        reward_model = _IndexReportingRewardModel(vocab)
        completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        context_length = 2
        query_responses = torch.cat([torch.zeros((1, context_length), dtype=torch.long), completion_ids], dim=1)

        scores = get_reward_from_policy_tokens(
            reward_model,
            query_responses,
            context_length,
            [prompt],
            tokenizer,
            tokenizer,
        )
        scored_index = int(scores.item())
        assert scored_index == last_index
        assert scored_index != first_pad_index - 1

    def test_batched_chat_padding_scores_per_row_last_token(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        tokenizer.pad_token = tokenizer.eos_token
        prompts = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "A much longer user question for padding"}],
        ]
        completions = ["Yes", "This is a longer assistant reply"]
        expected = []
        for prompt, completion in zip(prompts, completions, strict=True):
            text = apply_chat_template(
                {"messages": prompt + [{"role": "assistant", "content": completion}]}, tokenizer
            )["text"]
            token_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            expected.append(token_ids.size(0) - 1)

        vocab = max(tokenizer.vocab_size, len(tokenizer))
        reward_model = _IndexReportingRewardModel(vocab)
        completion_ids = tokenizer(completions, add_special_tokens=False, padding=True, return_tensors="pt")[
            "input_ids"
        ]
        context_length = 1
        query_responses = torch.cat(
            [torch.zeros((len(prompts), context_length), dtype=torch.long), completion_ids], dim=1
        )

        scores = get_reward_from_policy_tokens(
            reward_model,
            query_responses,
            context_length,
            prompts,
            tokenizer,
            tokenizer,
        )
        assert [int(x) for x in scores.tolist()] == expected

    def test_same_tokenizer_matches_add_special_tokens_false(self):
        tokenizer = self.reward_tokenizer
        if tokenizer.bos_token_id is None:
            pytest.skip("tokenizer needs a BOS token to pin the Online DPO encoding")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        prompt = "The answer is 2 + 2?"
        completion = " that is four."
        with_bos = tokenizer(prompt + completion, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        without_bos = tokenizer(prompt + completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        assert with_bos[0] == tokenizer.bos_token_id
        assert without_bos[0] != tokenizer.bos_token_id
        assert with_bos.tolist() != without_bos.tolist()

        vocab = max(tokenizer.vocab_size, len(tokenizer), int(with_bos.max()) + 1)
        reward_model = _IndexReportingRewardModel(vocab)

        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prompt_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]], dtype=prompt_ids.dtype), prompt_ids], dim=1)
        completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        query_responses = torch.cat([prompt_ids, completion_ids], dim=1)

        scores = get_reward_from_policy_tokens(
            reward_model,
            query_responses,
            prompt_ids.shape[1],
            [prompt],
            tokenizer,
            tokenizer,
        )
        assert int(scores.item()) == without_bos.size(0) - 1
        assert int(scores.item()) != query_responses.size(1) - 1
