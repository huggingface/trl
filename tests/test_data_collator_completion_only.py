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

import torch
from transformers import AutoTokenizer

from trl import DataCollatorForCompletionOnlyLM


# Define samples globally for reuse
CHATML_SAMPLE_BASIC_MULTI_TURN = """<|im_start|>system
system prompt system ptompt system prompt

<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
T T T T T<|im_end|>
<|im_start|>user
T T T T T<|im_end|>
<|im_start|>user
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>"""  # 4 assistant turns

CHATML_SAMPLE_MULTI_ROLE_MULTI_TURN = """<|im_start|>system
system prompt system ptompt system prompt

<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>"""  # 8 assistant turns

CHATML_SAMPLE_CONSECUTIVE_ASSISTANT_MULTI_ROLE = """<|im_start|>system
system prompt system ptompt system prompt

<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>user
U U U U U<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>tool
T T T T T<|im_end|>
<|im_start|>assistant
A A A A A<|im_end|>"""  # 8 assistant turns

CHATML_SAMPLE_CONSECUTIVE_ASSISTANT_SIMPLE = """<|im_start|>system
Prompt.
<|im_end|>
<|im_start|>user
User query.<|im_end|>
<|im_start|>assistant
Assistant response 1.<|im_end|>
<|im_start|>assistant
Assistant response 2.<|im_end|>
<|im_start|>user
Another user query.<|im_end|>
<|im_start|>assistant
Assistant response 3.<|im_end|>"""  # 3 assistant turns total, 2 consecutive

# Expected decoded output for a single assistant turn based on the samples above
EXPECTED_DECODED_ASSISTANT_CHUNK = "A A A A A<|im_end|>\n"


class DataCollatorForCompletionOnlyLMTester(unittest.TestCase):
    def test_data_collator_finds_response_template_llama2_tokenizer(self):
        # this should ideally be tested with meta-llama/Llama-2-7b-hf
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.instruction = """### System: You are a helpful assistant.

### User: How much is 2+2?

### Assistant: 2+2 equals 4"""
        self.instruction_template = "\n### User:"
        self.response_template = "\n### Assistant:"

        # GPT2Tokenizer: [198, 21017, 11787, 25] -> [21017, 11787, 25]
        # Llama2Tokenizer: [29871, 13, 2277, 29937, 4911, 29901] -> [2277, 29937, 4911, 29901]
        # Note: If this test is ever switched to Llama2Tokenizer, this should be double checked,
        # and possibly switched back to [2:] instead of [1:].
        # With GPT2Tokenizer, [1:] is correct - we want the 21017 token included, which is ###.
        self.tokenized_instruction_w_context = self.tokenizer.encode(
            self.instruction_template, add_special_tokens=False
        )[1:]

        # GPT2Tokenizer: [198, 21017, 15286, 25] -> [15286, 25]
        # Llama2Tokenizer: [29871, 13, 2277, 29937, 4007, 22137, 29901] -> [2277, 29937, 4007, 22137, 29901]
        self.tokenized_response_w_context = self.tokenizer.encode(self.response_template, add_special_tokens=False)[2:]

        # Plain check on string
        self.assertIn(self.response_template, self.instruction)
        self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)

        # Test the fix for #598
        # Pass already tokenized (w context) and truncated response_template so token_ids are like in the instruction + response
        self.collator = DataCollatorForCompletionOnlyLM(self.tokenized_response_w_context, tokenizer=self.tokenizer)
        self.collator.torch_call([self.tokenized_instruction])

        # Test for PR #749
        # Pass already tokenized (w context) instruction and response both so token_ids are like in the instruction + response
        self.collator = DataCollatorForCompletionOnlyLM(
            self.tokenized_response_w_context, self.tokenized_instruction_w_context, tokenizer=self.tokenizer
        )
        self.collator.torch_call([self.tokenized_instruction])

        # Test for PR #1185
        # We pass in a string where the first user template is different than the rest.
        # Usually this would happen due to context-sensitive tokenization, but here we
        # explicitly change the template to test the fix.
        self.instruction = """## User: First instruction

### Assistant: First response

### User: Second instruction

### Assistant: Second response"""
        self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)
        self.collator = DataCollatorForCompletionOnlyLM(
            self.tokenized_response_w_context, self.tokenized_instruction_w_context, tokenizer=self.tokenizer
        )
        collator_output = self.collator.torch_call([self.tokenized_instruction])
        collator_text = self.tokenizer.decode(
            collator_output["labels"][torch.where(collator_output["labels"] != -100)]
        )
        expected_text = " First response\n\n Second response"
        self.assertEqual(collator_text, expected_text)

    def test_data_collator_handling_of_long_sequences(self):
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.instruction = """### System: You are a helpful assistant.

### User: How much is 2+2? I'm asking because I'm not sure. And I'm not sure because I'm not good at math.
"""
        self.response_template = "\n### Assistant:"
        # check DataCollatorForCompletionOnlyLM using response template only
        self.tokenized_instruction = self.tokenizer.encode(self.instruction, add_special_tokens=False)
        self.collator = DataCollatorForCompletionOnlyLM(self.response_template, tokenizer=self.tokenizer)

        with self.assertWarns(UserWarning):  # it should raise a warning since the response_template isn't found
            encoded_instance = self.collator.torch_call([self.tokenized_instruction])

        result = torch.all(encoded_instance["labels"] == -100)
        self.assertTrue(result, "Not all values in the tensor are -100.")

        # check DataCollatorForCompletionOnlyLM using response template and instruction template
        self.instruction_template = "\n### User:"
        self.collator = DataCollatorForCompletionOnlyLM(
            self.response_template, self.instruction_template, tokenizer=self.tokenizer
        )
        with self.assertWarns(UserWarning):  # it should raise a warning since the response_template isn't found
            encoded_instance = self.collator.torch_call([self.tokenized_instruction])
        result = torch.all(encoded_instance["labels"] == -100)
        self.assertTrue(result, "Not all values in the tensor are -100.")

    def test_padding_free(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inst1 = "### System: You are a helpful assistant.\n\n### User: How much is 2+2?\n\n### Assistant: 2+2 equals 4"
        inst2 = "### System: You are a honest and helpful assistant.\n\n### User: What is the answer of 22x22?\n\n### Assistant: 22x22 equals 484"

        response_template = "### Assistant:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        collator_paddingfree = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer, padding_free=True
        )

        tokenized_instruction = [tokenizer(x, add_special_tokens=False) for x in [inst1, inst2]]
        batch = collator(tokenized_instruction)
        batch_paddingfree = collator_paddingfree(tokenized_instruction)

        self.assertNotIn("attention_mask", batch_paddingfree)
        self.assertIn("input_ids", batch_paddingfree)
        self.assertIn("labels", batch_paddingfree)
        self.assertIn("position_ids", batch_paddingfree)
        self.assertEqual(batch_paddingfree["input_ids"].size(), batch_paddingfree["labels"].size())
        self.assertEqual(batch_paddingfree["labels"].size(), batch_paddingfree["position_ids"].size())

        attn_mask = batch["attention_mask"]
        input_ids_remove_pad = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
        expected_position_ids = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
        expected_labels = []
        for idx in range(batch["input_ids"].size(0)):
            expected_labels.append(batch["labels"][idx][attn_mask[idx].bool()])
            expected_labels[-1][0] = collator.ignore_index
        expected_labels = torch.cat(expected_labels).unsqueeze(0)

        self.assertTrue((input_ids_remove_pad == batch_paddingfree["input_ids"]).all())
        self.assertTrue((expected_position_ids == batch_paddingfree["position_ids"]).all())
        self.assertTrue((expected_labels == batch_paddingfree["labels"]).all())

    def test_data_collator_for_completion_only_lm(self):
        # The tokenizer isn't use but the collator needs it to be provided.
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")

        collator = DataCollatorForCompletionOnlyLM(tokenizer.decode(9999), tokenizer=tokenizer, padding_free=True)

        tokenized_instruction = [
            {"input_ids": [1, 2, 3, 9999, 4, 5], "attention_mask": [1, 1, 1, 1, 1, 1]},
            {"input_ids": [6, 7, 8, 9, 9999, 10, 11], "attention_mask": [1, 1, 1, 1, 1, 1, 1]},
        ]
        batch = collator(tokenized_instruction)

        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6]])  # flat pos ids
        self.assertEqual(
            batch["cu_seq_lens_q"].tolist(), [[0, 6, 13]]
        )  # start idx of each seq + total number of tokens
        self.assertEqual(batch["cu_seq_lens_k"].tolist(), [[0, 6, 13]])  # idem
        self.assertEqual(batch["max_length_k"], torch.tensor([7]))  # max length in batch, here 7 (second sequence)
        self.assertEqual(batch["max_length_q"], torch.tensor([7]))  # idem

    def test_masking_basic_multi_turn(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_start|>assistant\n"

        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )

        conversations = [
            CHATML_SAMPLE_BASIC_MULTI_TURN,
            CHATML_SAMPLE_BASIC_MULTI_TURN,
        ]  # Batch of 2 identical samples
        tokenized = tokenizer(conversations, add_special_tokens=False)

        # Prepare input for collator in the typical dictionary format
        batch_input = [
            {"input_ids": tokenized.input_ids[i], "attention_mask": tokenized.attention_mask[i]}
            for i in range(len(tokenized.input_ids))
        ]
        collated_batch = data_collator(batch_input)

        # Expected output: 4 assistant turns per sample
        expected_decoded_output = EXPECTED_DECODED_ASSISTANT_CHUNK * 4

        # Check labels for each sample in the batch
        for i in range(len(collated_batch["labels"])):
            valid_indices = collated_batch["labels"][i] != -100
            valid_labels = collated_batch["labels"][i][valid_indices]
            decoded_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
            # strip potential leading/trailing whitespace artefacts from decode
            self.assertEqual(
                decoded_text.strip(), expected_decoded_output.strip(), f"Mismatch in decoded labels for sample {i}"
            )

    def test_masking_multi_role_multi_template(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use a list for multiple instruction templates
        instruction_template = ["<|im_start|>tool\n", "<|im_start|>user\n"]
        response_template = "<|im_start|>assistant\n"

        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )

        conversations = [CHATML_SAMPLE_MULTI_ROLE_MULTI_TURN, CHATML_SAMPLE_CONSECUTIVE_ASSISTANT_MULTI_ROLE]
        tokenized = tokenizer(conversations, add_special_tokens=False)

        batch_input = [
            {"input_ids": tokenized.input_ids[i], "attention_mask": tokenized.attention_mask[i]}
            for i in range(len(tokenized.input_ids))
        ]
        collated_batch = data_collator(batch_input)

        # Expected outputs based on the number of assistant turns
        expected_outputs = [
            EXPECTED_DECODED_ASSISTANT_CHUNK * 8,  # CHATML_SAMPLE_MULTI_ROLE_MULTI_TURN has 8 assistant turns
            EXPECTED_DECODED_ASSISTANT_CHUNK
            * 8,  # CHATML_SAMPLE_CONSECUTIVE_ASSISTANT_MULTI_ROLE has 8 assistant turns
        ]

        # Check labels for each sample in the batch
        self.assertEqual(len(collated_batch["labels"]), len(expected_outputs), "Batch size mismatch")

        for i in range(len(collated_batch["labels"])):
            valid_indices = collated_batch["labels"][i] != -100
            valid_labels = collated_batch["labels"][i][valid_indices]
            decoded_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
            self.assertEqual(
                decoded_text.strip(), expected_outputs[i].strip(), f"Mismatch in decoded labels for sample {i}"
            )

    def test_masking_consecutive_assistant(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        instruction_template = ["<|im_start|>tool\n", "<|im_start|>user\n"]
        response_template = "<|im_start|>assistant\n"

        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )

        tokenized = tokenizer([CHATML_SAMPLE_CONSECUTIVE_ASSISTANT_SIMPLE], add_special_tokens=False)
        batch_input = [
            {"input_ids": tokenized.input_ids[i], "attention_mask": tokenized.attention_mask[i]}
            for i in range(len(tokenized.input_ids))
        ]
        collated_batch = data_collator(batch_input)

        # Expected: Only the content *after* the response_template should be unmasked for all assistant turns.
        # The logic correctly handles consecutive turns by masking up to the *next* instruction or the end.
        expected_decoded_output = (
            "Assistant response 1.<|im_end|>\nAssistant response 2.<|im_end|>\nAssistant response 3.<|im_end|>\n"
        )

        valid_indices = collated_batch["labels"][0] != -100
        valid_labels = collated_batch["labels"][0][valid_indices]
        decoded_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
        self.assertEqual(
            decoded_text.strip(),
            expected_decoded_output.strip(),
            "Mismatch in decoded labels for consecutive assistant test",
        )

    def test_masking_left_padding(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # Explicitly set left padding
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        instruction_template = ["<|im_start|>tool\n", "<|im_start|>user\n"]
        response_template = "<|im_start|>assistant\n"

        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )

        conversations = [CHATML_SAMPLE_MULTI_ROLE_MULTI_TURN, CHATML_SAMPLE_BASIC_MULTI_TURN]
        tokenized = tokenizer(conversations, add_special_tokens=False, padding=True, truncation=True, max_length=512)

        batch_input = [
            {"input_ids": tokenized.input_ids[i], "attention_mask": tokenized.attention_mask[i]}
            for i in range(len(tokenized.input_ids))
        ]
        collated_batch = data_collator(batch_input)

        # Expected outputs based on the number of assistant turns in the specific samples used
        expected_outputs = [
            EXPECTED_DECODED_ASSISTANT_CHUNK * 8,  # CHATML_SAMPLE_MULTI_ROLE_MULTI_TURN
            EXPECTED_DECODED_ASSISTANT_CHUNK * 4,  # CHATML_SAMPLE_BASIC_MULTI_TURN
        ]

        self.assertEqual(len(collated_batch["labels"]), len(expected_outputs), "Batch size mismatch")

        for i in range(len(collated_batch["labels"])):
            valid_indices = collated_batch["labels"][i] != -100
            valid_labels = collated_batch["labels"][i][valid_indices]
            decoded_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
            self.assertEqual(
                decoded_text.strip(),
                expected_outputs[i].strip(),
                f"Mismatch in decoded labels for left padding, sample {i}",
            )

    def test_masking_tokenized_templates(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Pre-tokenize the templates
        instruction_templates_str = ["<|im_start|>tool\n", "<|im_start|>user\n"]
        response_template_str = "<|im_start|>assistant\n"

        instruction_token_ids = [
            tokenizer.encode(tmpl, add_special_tokens=False) for tmpl in instruction_templates_str
        ]
        response_token_ids = tokenizer.encode(response_template_str, add_special_tokens=False)

        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_token_ids,  # Pass List[List[int]]
            response_template=response_token_ids,  # Pass List[int]
            tokenizer=tokenizer,
            mlm=False,
        )

        conversations = [CHATML_SAMPLE_MULTI_ROLE_MULTI_TURN, CHATML_SAMPLE_CONSECUTIVE_ASSISTANT_MULTI_ROLE]
        tokenized = tokenizer(conversations, add_special_tokens=False, padding=True, truncation=True, max_length=512)

        batch_input = [
            {"input_ids": tokenized.input_ids[i], "attention_mask": tokenized.attention_mask[i]}
            for i in range(len(tokenized.input_ids))
        ]
        collated_batch = data_collator(batch_input)

        # Expected outputs based on the number of assistant turns
        expected_outputs = [
            EXPECTED_DECODED_ASSISTANT_CHUNK * 8,
            EXPECTED_DECODED_ASSISTANT_CHUNK * 8,
        ]

        self.assertEqual(len(collated_batch["labels"]), len(expected_outputs), "Batch size mismatch")

        for i in range(len(collated_batch["labels"])):
            valid_indices = collated_batch["labels"][i] != -100
            valid_labels = collated_batch["labels"][i][valid_indices]
            decoded_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
            self.assertEqual(
                decoded_text.strip(),
                expected_outputs[i].strip(),
                f"Mismatch in decoded labels for tokenized templates, sample {i}",
            )
