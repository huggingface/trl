# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from transformers.testing_utils import require_peft
from transformers.utils import is_peft_available

from trl.trainer.model_config import ModelConfig
from trl.trainer.utils import decode_and_strip_padding, generate_model_card, get_peft_config, pad


if is_peft_available():
    from peft import LoraConfig


class TestPad(unittest.TestCase):
    def test_pad_1_dim_left(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_1_dim_right(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_left(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[0, 0], [5, 6]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5, 6]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))

    def test_pad_2_dim_right_multidim(self):
        x = torch.tensor([[1, 2], [3, 4]])
        y = torch.tensor([[5]])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 0], [0, 0]],
            ]
        )
        self.assertTrue(torch.equal(output, expected))


@require_peft
class TestGetPEFTConfig(unittest.TestCase):
    def test_create_peft_config_use_peft_false(self):
        """Test that when use_peft is False, the function returns None."""
        model_config = ModelConfig(use_peft=False)
        peft_config = get_peft_config(model_config)
        self.assertIsNone(peft_config)

    def test_create_peft_config_use_peft_true(self):
        """Test that when use_peft is True, the function returns a LoraConfig object."""
        # Provide non-default values to the model config for testing
        peft_kwargs = {
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_task_type": "SEQ_CLS",
            "use_rslora": True,
            "lora_target_modules": ["up_proj", "down_proj"],
            "lora_modules_to_save": ["up_proj"],
        }
        model_config = ModelConfig(use_peft=True, **peft_kwargs)
        peft_config = get_peft_config(model_config)
        self.assertTrue(isinstance(peft_config, LoraConfig))
        for arg, value in peft_kwargs.items():
            # Test that lists of modules are converted to sets
            if arg == "lora_target_modules":
                value = set(value)
            # Rename the argument to match the LoraConfig attribute name
            if arg in ["lora_r", "lora_task_type", "lora_target_modules", "lora_modules_to_save"]:
                arg = arg[len("lora_") :] if arg.startswith("lora_") else arg

            self.assertEqual(getattr(peft_config, arg), value)


class TestDecodeAndStripPadding(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    def test_example_with_padding(self):
        inputs = self.tokenizer(["Hello world", "Hello"], padding=True, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello world", "Hello"])

    def test_example_without_padding(self):
        inputs = self.tokenizer(["Hello", "Hello"], padding=False, return_tensors="pt")
        decoded = decode_and_strip_padding(inputs["input_ids"], self.tokenizer)
        self.assertEqual(decoded, ["Hello", "Hello"])


class TestGenerateModelCard(unittest.TestCase):
    def test_full(self):
        model_card = generate_model_card(
            base_model="username/my_base_model",
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name="username/my_dataset",
            tags=["trl", "trainer-tag"],
            wandb_url="https://wandb.ai/username/project_id/runs/abcd1234",
            trainer_name="My Trainer",
            trainer_citation="@article{my_trainer, ...}",
            paper_title="My Paper",
            paper_id="1234.56789",
        )
        card_text = str(model_card)
        assert "[username/my_base_model](https://huggingface.co/username/my_base_model)" in card_text
        assert "my_model" in card_text
        assert 'pipeline("text-generation", model="username/my_hub_model", device="cuda")' in card_text
        assert "datasets: username/my_dataset" in card_text
        assert "](https://wandb.ai/username/project_id/runs/abcd1234)" in card_text
        assert "My Trainer" in card_text
        assert "```bibtex\n@article{my_trainer, ...}\n```" in card_text
        assert "[My Paper](https://huggingface.co/papers/1234.56789)" in card_text

    def test_val_none(self):
        model_card = generate_model_card(
            base_model=None,
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name=None,
            tags=None,
            wandb_url=None,
            trainer_name="My Trainer",
            trainer_citation=None,
            paper_title=None,
            paper_id=None,
        )
        card_text = str(model_card)
        assert "my_model" in card_text
        assert 'pipeline("text-generation", model="username/my_hub_model", device="cuda")' in card_text
        assert "My Trainer" in card_text
