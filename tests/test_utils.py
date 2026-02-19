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

import textwrap
from io import StringIO
from unittest.mock import patch

import pytest
import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.utils import is_peft_available

from trl import ModelConfig
from trl.trainer.utils import (
    RepeatSampler,
    entropy_from_logits,
    flush_left,
    flush_right,
    forward_masked_logits,
    generate_model_card,
    get_peft_config,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
    use_adapter,
)

from .testing_utils import TrlTestCase, require_peft, require_rich


if is_peft_available():
    from peft import AutoPeftModelForCausalLM, LoraConfig


@require_peft
class TestUseAdapter(TrlTestCase):
    def test_disables_on_none(self):
        model = AutoPeftModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-PeftModel", adapter_name="my_adapter"
        )
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with model.disable_adapter():
            expected = model(input_ids).logits

        with use_adapter(model, None):
            output = model(input_ids).logits

        assert torch.equal(output, expected)

    def test_restores_previous_adapter(self):
        model = AutoPeftModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-PeftModel", adapter_name="my_adapter"
        )
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        expected = model(input_ids).logits
        with use_adapter(model, "my_adapter"):
            pass
        output = model(input_ids).logits
        assert torch.equal(output, expected)

        with use_adapter(model, None):
            pass
        output = model(input_ids).logits
        assert torch.equal(output, expected)

    def test_with_multiple_adapters(self):
        model = AutoPeftModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-PeftModel", adapter_name="my_adapter_1"
        )
        model.load_adapter("trl-internal-testing/tiny-PeftModel-2", "my_adapter_2")
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        model.set_adapter("my_adapter_1")  # should be a no-op, but let's keep it for clarity
        expected_1 = model(input_ids).logits
        model.set_adapter("my_adapter_2")
        expected_2 = model(input_ids).logits

        with use_adapter(model, "my_adapter_1"):
            output_1 = model(input_ids).logits

        with use_adapter(model, "my_adapter_2"):
            output_2 = model(input_ids).logits

        assert torch.equal(output_1, expected_1)
        assert torch.equal(output_2, expected_2)


class TestPad(TrlTestCase):
    def test_pad_1_dim_left(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="left")
        expected = torch.tensor([[1, 2, 3], [0, 4, 5]])
        assert torch.equal(output, expected)

    def test_pad_1_dim_right(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        output = pad((x, y), padding_value=0, padding_side="right")
        expected = torch.tensor([[1, 2, 3], [4, 5, 0]])
        assert torch.equal(output, expected)

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
        assert torch.equal(output, expected)

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
        assert torch.equal(output, expected)

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
        assert torch.equal(output, expected)

    def test_pad_to_multiple_of_1(self):
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="right", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        assert torch.equal(output, expected)

    def test_pad_to_multiple_of_2(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([6, 7, 8])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="right", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]])
        assert torch.equal(output, expected)

    def test_pad_to_multiple_of_side_left(self):
        x = torch.tensor([1, 2, 3, 4, 5])
        y = torch.tensor([6, 7, 8])
        # Max length is 3, pad to multiple of 4
        output = pad((x, y), padding_value=0, padding_side="left", pad_to_multiple_of=4)
        expected = torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 6, 7, 8]])
        assert torch.equal(output, expected)

    def test_pad_to_multiple_of_no_extra_padding(self):
        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([5, 6, 7, 8])
        # Already multiple of 4
        output = pad((x, y), padding_value=0, padding_side="left", pad_to_multiple_of=4)
        expected = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert torch.equal(output, expected)


@require_peft
class TestGetPEFTConfig(TrlTestCase):
    def test_create_peft_config_use_peft_false(self):
        """Test that when use_peft is False, the function returns None."""
        model_args = ModelConfig(use_peft=False)
        peft_config = get_peft_config(model_args)
        assert peft_config is None

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
        model_args = ModelConfig(use_peft=True, **peft_kwargs)
        peft_config = get_peft_config(model_args)
        assert isinstance(peft_config, LoraConfig)
        for arg, value in peft_kwargs.items():
            # Test that lists of modules are converted to sets
            if arg == "lora_target_modules":
                value = set(value)
            # Rename the argument to match the LoraConfig attribute name
            if arg in ["lora_r", "lora_task_type", "lora_target_modules", "lora_modules_to_save"]:
                arg = arg[len("lora_") :] if arg.startswith("lora_") else arg

            assert getattr(peft_config, arg) == value


class TestNanStd(TrlTestCase):
    def test_nanstd_ignores_nans(self):
        x = torch.tensor([1.0, 2.0, 3.0, float("nan")])
        result = nanstd(x)
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_nanstd_dim_and_keepdim(self):
        x = torch.tensor([[1.0, float("nan")], [3.0, 5.0]])
        result = nanstd(x, dim=1, keepdim=True)
        assert torch.isnan(result[0, 0])
        torch.testing.assert_close(result[1, 0], torch.tensor(1.4142135), rtol=1e-5, atol=1e-6)

    def test_nanstd_all_nan(self):
        x = torch.tensor([float("nan"), float("nan")])
        result = nanstd(x)
        assert torch.isnan(result)


class TestGenerateModelCard(TrlTestCase):
    def test_full(self):
        model_card = generate_model_card(
            base_model="username/my_base_model",
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name="username/my_dataset",
            tags=["trl", "trainer-tag"],
            wandb_url="https://wandb.ai/username/project_id/runs/abcd1234",
            trackio_url="https://huggingface.co/spaces/username/space_id",
            comet_url="https://www.comet.com/username/project_id/experiment_id",
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
        assert "](https://huggingface.co/spaces/username/space_id)" in card_text
        assert "](https://www.comet.com/username/project_id/experiment_id" in card_text
        assert "My Trainer" in card_text
        assert "```bibtex\n@article{my_trainer, ...}\n```" in card_text
        assert "[My Paper](https://huggingface.co/papers/1234.56789)" in card_text

    def test_val_none(self):
        model_card = generate_model_card(
            base_model=None,
            model_name="my_model",
            hub_model_id="username/my_hub_model",
            dataset_name=None,
            tags=[],
            wandb_url=None,
            trackio_url=None,
            comet_url=None,
            trainer_name="My Trainer",
            trainer_citation=None,
            paper_title=None,
            paper_id=None,
        )
        card_text = str(model_card)
        assert "my_model" in card_text
        assert 'pipeline("text-generation", model="username/my_hub_model", device="cuda")' in card_text
        assert "My Trainer" in card_text


class TestFlushLeft(TrlTestCase):
    def test_basic_case(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        tensor1 = torch.tensor([[0, 0, 2, 3, 4], [0, 5, 6, 0, 0]])
        tensor2 = torch.tensor([[0, 0, 7, 8, 9], [0, 10, 11, 0, 0]])
        new_mask, new_tensor1, new_tensor2 = flush_left(mask, tensor1, tensor2)

        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        expected_tensor1 = torch.tensor([[2, 3, 4], [5, 6, 0]])
        expected_tensor2 = torch.tensor([[7, 8, 9], [10, 11, 0]])

        assert torch.equal(new_mask, expected_mask)
        assert torch.equal(new_tensor1, expected_tensor1)
        assert torch.equal(new_tensor2, expected_tensor2)

    def test_single_row(self):
        mask = torch.tensor([[0, 0, 1, 1]])
        tensor1 = torch.tensor([[0, 0, 2, 3]])
        new_mask, new_tensor1 = flush_left(mask, tensor1)

        expected_mask = torch.tensor([[1, 1]])
        expected_tensor1 = torch.tensor([[2, 3]])

        assert torch.equal(new_mask, expected_mask)
        assert torch.equal(new_tensor1, expected_tensor1)

    def test_no_shift_needed(self):
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
        tensor1 = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]])
        new_mask, new_tensor1 = flush_left(mask, tensor1)

        expected_mask = torch.tensor([[1, 1], [1, 0]])
        expected_tensor1 = torch.tensor([[5, 6], [7, 0]])

        assert torch.equal(new_mask, expected_mask)
        assert torch.equal(new_tensor1, expected_tensor1)

    def test_no_tensors(self):
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
        new_mask = flush_left(mask)
        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        assert torch.equal(new_mask, expected_mask)


class TestFlushRight(TrlTestCase):
    def test_basic_case(self):
        mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        tensor1 = torch.tensor([[2, 3, 4, 0, 0], [0, 0, 5, 6, 0]])
        tensor2 = torch.tensor([[7, 8, 9, 0, 0], [0, 0, 10, 11, 0]])
        new_mask, new_tensor1, new_tensor2 = flush_right(mask, tensor1, tensor2)

        expected_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        expected_tensor1 = torch.tensor([[2, 3, 4], [0, 5, 6]])
        expected_tensor2 = torch.tensor([[7, 8, 9], [0, 10, 11]])

        assert torch.equal(new_mask, expected_mask)
        assert torch.equal(new_tensor1, expected_tensor1)
        assert torch.equal(new_tensor2, expected_tensor2)

    def test_single_row(self):
        mask = torch.tensor([[1, 1, 0, 0]])
        tensor1 = torch.tensor([[2, 3, 0, 0]])
        new_mask, new_tensor1 = flush_right(mask, tensor1)

        expected_mask = torch.tensor([[1, 1]])
        expected_tensor1 = torch.tensor([[2, 3]])

        assert torch.equal(new_mask, expected_mask)
        assert torch.equal(new_tensor1, expected_tensor1)

    def test_no_shift_needed(self):
        mask = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]])
        tensor1 = torch.tensor([[0, 0, 5, 6], [0, 0, 0, 7]])
        new_mask, new_tensor1 = flush_right(mask, tensor1)

        expected_mask = torch.tensor([[1, 1], [0, 1]])
        expected_tensor1 = torch.tensor([[5, 6], [0, 7]])

        assert torch.equal(new_mask, expected_mask)
        assert torch.equal(new_tensor1, expected_tensor1)

    def test_no_tensors(self):
        mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        new_mask = flush_right(mask)
        expected_mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        assert torch.equal(new_mask, expected_mask)


class TestRepeatRandomSampler(TrlTestCase):
    def test_sampler(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 1, 1, 2, 2, 6, 6, 5, 5]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated twice
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))

    def test_sampler_no_shuffle(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, shuffle=False)
        sampled = list(sampler)
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        assert sampled == expected

    def test_sampler_no_repeat(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1)
        # Should output something like [4, 3, 0, 1, 2, 6, 5]
        sampled = list(sampler)
        # Check that the length is the same
        assert len(sampled) == len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))

    def test_sampler_with_batch_size(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g", "h"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6, 5, 7, 5, 7]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * len(dataset)
        # Check that all indexes are present
        assert set(sampled) == set(range(len(dataset)))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_batch_size_and_drop(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=1, batch_size=2, repeat_count=2)
        # Should output something like [4, 3, 4, 3, 0, 1, 0, 1, 2, 6, 2, 6]
        sampled = list(sampler)
        # Check that the length is doubled
        assert len(sampled) == 2 * (
            len(dataset) - 1
        )  # one element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i : i + 1] == sampled[i + 2 : i + 3] for i in range(0, len(sampled), 4))

    def test_sampler_with_mini_repeat_count_and_batch_size_1(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=3, repeat_count=2)
        # Should output something like [4, 4, 3, 3, 0, 0, 4, 4, 3, 3, 0, 0,
        #                               1, 1, 2, 2, 6, 6, 1, 1, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is quadrupled
        assert len(sampled) == 4 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]

    def test_sampler_with_mini_repeat_count_and_batch_size_2(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=3, batch_size=2, repeat_count=2)
        # Should output something like [4, 4, 4, 3, 3, 3, 4, 4, 4, 3, 3, 3,
        #                               0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
        #                               2, 2, 2, 6, 6, 6, 2, 2, 2, 6, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        assert len(sampler) == len(sampled)  # the length should be the same as the sampled length
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] == sampled[i + 2] for i in range(0, len(sampled), 3))
        # Check that the batch is repeated as expected
        assert sampled[0:6] == sampled[6:12]
        assert sampled[12:18] == sampled[18:24]
        assert sampled[24:30] == sampled[30:36]

    def test_sampler_with_mini_repeat_count_and_batch_size_3(self):
        dataset = ["a", "b", "c", "d", "e", "f", "g"]
        sampler = RepeatSampler(dataset, mini_repeat_count=2, batch_size=2, repeat_count=3)
        # Should output something like [4, 4, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3,
        #                               0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
        #                               2, 2, 6, 6, 2, 2, 6, 6, 2, 2, 6, 6]
        sampled = list(sampler)
        # Check that the length is sextupled
        assert len(sampled) == 6 * (len(dataset) - 1)  # 1 element is dropped, because it's not enough to form a batch
        # Check that the sampled indexes are a subset of the dataset indexes
        assert set(sampled).issubset(set(range(len(dataset))))
        # Check that each element is repeated as expected
        assert all(sampled[i] == sampled[i + 1] for i in range(0, len(sampled), 2))
        # Check that the batch is repeated as expected
        assert sampled[0:4] == sampled[4:8] == sampled[8:12]
        assert sampled[12:16] == sampled[16:20] == sampled[20:24]
        assert sampled[24:28] == sampled[28:32] == sampled[32:36]


class TestEntropyFromLogits(TrlTestCase):
    @pytest.mark.parametrize("shape", [(768,), (32, 768), (8, 16, 768), (2, 4, 8, 768)])
    @pytest.mark.parametrize("chunk_size", [1, 16])
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_entropy_from_logits_2_dims(self, dtype, chunk_size, shape):
        logits = torch.randn(*shape, dtype=dtype)
        if dtype in (torch.float64, torch.float32):
            p = logits.softmax(-1)
            entropy = -torch.sum(p * p.log(), dim=-1)
        else:
            logps = logits.log_softmax(dim=-1)
            entropy = -(torch.exp(logps) * logps).sum(-1)
        predicted_entropy = entropy_from_logits(logits, chunk_size=chunk_size)
        torch.testing.assert_close(predicted_entropy, entropy, rtol=1e-5, atol=1e-5)


@require_rich
class TestPrintPromptCompletionsSample(TrlTestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_print_output(self, mock_stdout):
        prompts = ["The sky is", "The sun is"]
        completions = [" blue.", " in the sky."]
        rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
        advantages = [0.987, 0.654]
        step = 42

        print_prompt_completions_sample(prompts, completions, rewards, advantages, step)

        output = mock_stdout.getvalue()

        # docstyle-ignore
        expected_output = textwrap.dedent("""\
        ╭──────────────────────────── Step 42 ─────────────────────────────╮
        │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
        │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ │
        │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
        │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ │
        │ ├────────────┼──────────────┼─────────────┼────────┼───────────┤ │
        │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ │
        │ └────────────┴──────────────┴─────────────┴────────┴───────────┘ │
        ╰──────────────────────────────────────────────────────────────────╯
        """)

        assert output == expected_output

    @patch("sys.stdout", new_callable=StringIO)
    def test_num_samples(self, mock_stdout):
        prompts = ["A", "B"]
        completions = ["1", "2"]
        rewards = {"Score": [0.1, 0.2]}
        advantages = [0.3, 0.4]
        step = 10

        print_prompt_completions_sample(prompts, completions, rewards, advantages, step, num_samples=1)
        output = mock_stdout.getvalue()

        # docstyle-ignore
        possible_outputs = [
            textwrap.dedent("""\
            ╭────────────────── Step 10 ──────────────────╮
            │ ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓ │
            │ ┃ Prompt ┃ Completion ┃ Score ┃ Advantage ┃ │
            │ ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩ │
            │ │ A      │ 1          │  0.10 │      0.30 │ │
            │ └────────┴────────────┴───────┴───────────┘ │
            ╰─────────────────────────────────────────────╯
                """),
            # docstyle-ignore
            textwrap.dedent("""\
            ╭────────────────── Step 10 ──────────────────╮
            │ ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓ │
            │ ┃ Prompt ┃ Completion ┃ Score ┃ Advantage ┃ │
            │ ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩ │
            │ │ B      │ 2          │  0.20 │      0.40 │ │
            │ └────────┴────────────┴───────┴───────────┘ │
            ╰─────────────────────────────────────────────╯
                """),
        ]
        assert output in possible_outputs

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_messages(self, mock_stdout):
        prompts = [
            [
                {"role": "system", "content": "You are an helpful assistant."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            [
                {"role": "system", "content": "You are an helpful assistant."},
                {"role": "user", "content": "Where is the sun?"},
            ],
        ]
        completions = [
            [{"role": "assistant", "content": "It is blue."}],
            [{"role": "assistant", "content": "In the sky."}],
        ]
        rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
        advantages = [0.987, 0.654]
        step = 42

        print_prompt_completions_sample(prompts, completions, rewards, advantages, step)

        output = mock_stdout.getvalue()

        # docstyle-ignore
        expected_output = textwrap.dedent("""\
        ╭────────────────────────────────── Step 42 ───────────────────────────────────╮
        │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
        │ ┃ Prompt                  ┃ Completion  ┃ Correctness ┃ Format ┃ Advantage ┃ │
        │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
        │ │ SYSTEM                  │ ASSISTANT   │        0.12 │   0.79 │      0.99 │ │
        │ │ You are an helpful      │ It is blue. │             │        │           │ │
        │ │ assistant.              │             │             │        │           │ │
        │ │                         │             │             │        │           │ │
        │ │ USER                    │             │             │        │           │ │
        │ │ What color is the sky?  │             │             │        │           │ │
        │ ├─────────────────────────┼─────────────┼─────────────┼────────┼───────────┤ │
        │ │ SYSTEM                  │ ASSISTANT   │        0.46 │   0.10 │      0.65 │ │
        │ │ You are an helpful      │ In the sky. │             │        │           │ │
        │ │ assistant.              │             │             │        │           │ │
        │ │                         │             │             │        │           │ │
        │ │ USER                    │             │             │        │           │ │
        │ │ Where is the sun?       │             │             │        │           │ │
        │ └─────────────────────────┴─────────────┴─────────────┴────────┴───────────┘ │
        ╰──────────────────────────────────────────────────────────────────────────────╯
        """)

        assert output == expected_output

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_messages_with_tools(self, mock_stdout):
        prompts = [
            [{"role": "user", "content": "What is the temperature in Paris?"}],
            [{"role": "user", "content": "What is the weather in London?"}],
        ]
        completions = [
            [{"role": "tool", "name": "get_temperature", "args": {"location": "Paris"}}],
            [{"role": "tool", "name": "get_weather", "args": {"location": "London"}}],
        ]
        rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
        advantages = [0.987, 0.654]
        step = 42

        print_prompt_completions_sample(prompts, completions, rewards, advantages, step)

        output = mock_stdout.getvalue()

        # docstyle-ignore
        expected_output = textwrap.dedent("""\
        ╭────────────────────────────────── Step 42 ───────────────────────────────────╮
        │ ┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
        │ ┃ Prompt            ┃ Completion        ┃ Correctness ┃ Format ┃ Advantage ┃ │
        │ ┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
        │ │ USER              │ TOOL              │        0.12 │   0.79 │      0.99 │ │
        │ │ What is the       │ get_temperature(… │             │        │           │ │
        │ │ temperature in    │ 'Paris'})         │             │        │           │ │
        │ │ Paris?            │                   │             │        │           │ │
        │ ├───────────────────┼───────────────────┼─────────────┼────────┼───────────┤ │
        │ │ USER              │ TOOL              │        0.46 │   0.10 │      0.65 │ │
        │ │ What is the       │ get_weather({'lo… │             │        │           │ │
        │ │ weather in        │ 'London'})        │             │        │           │ │
        │ │ London?           │                   │             │        │           │ │
        │ └───────────────────┴───────────────────┴─────────────┴────────┴───────────┘ │
        ╰──────────────────────────────────────────────────────────────────────────────╯
        """)

        assert output == expected_output


class TestSelectiveLogSoftmax(TrlTestCase):
    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    def test_selective_log_softmax(self, dtype):
        """Test selective_log_softmax with logits of different dtypes"""
        vocab_size = 1024
        batch_size = 4
        seq_len = 32

        input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=dtype)

        expected_output = torch.gather(logits.log_softmax(-1), dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        actual_output = selective_log_softmax(logits, input_ids)

        if dtype in [torch.float16, torch.bfloat16]:
            # half-precision dtypes fall back to an exact method
            assert torch.equal(actual_output, expected_output)
        else:
            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("k", [1, 8])
    def test_selective_log_softmax_multi_index(self, dtype, k):
        """Test selective_log_softmax with logits of different dtypes and index widths"""
        vocab_size = 1024
        batch_size = 4
        seq_len = 32

        index = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len, k))
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=dtype)

        expected_output = torch.gather(logits.log_softmax(-1), dim=-1, index=index)
        actual_output = selective_log_softmax(logits, index)

        assert actual_output.shape == (batch_size, seq_len, k)
        if dtype in [torch.float16, torch.bfloat16]:
            # half-precision dtypes fall back to an exact method
            assert torch.equal(actual_output, expected_output)
        else:
            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)


class TestShuffleSequenceDict(TrlTestCase):
    def test_shuffle_preserves_shape(self):
        x = torch.arange(6).reshape(3, 2)
        y = torch.arange(3).reshape(3, 1)
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_sequence_dict(tensor_dict)

        assert shuffled["x"].shape == x.shape
        assert shuffled["y"].shape == y.shape

    def test_shuffle_consistent_across_tensors(self):
        # Use known patterns to check alignment
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = torch.tensor([[1], [2], [3]])
        tensor_dict = {"x": x.clone(), "y": y.clone()}

        shuffled = shuffle_sequence_dict(tensor_dict)

        # Build a reverse map from shuffled x rows to y values
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i].item()

            if torch.equal(x_row, torch.tensor([10, 11])):
                assert y_val == 1
            elif torch.equal(x_row, torch.tensor([20, 21])):
                assert y_val == 2
            elif torch.equal(x_row, torch.tensor([30, 31])):
                assert y_val == 3
            else:
                pytest.fail("Unexpected x row in shuffled output.")

    def test_none_tensor_remains_none(self):
        x = torch.arange(6).reshape(3, 2)
        tensor_dict = {"x": x.clone(), "y": None}

        shuffled = shuffle_sequence_dict(tensor_dict)

        assert shuffled["y"] is None
        assert shuffled["x"].shape == x.shape

    def test_shuffle_with_list(self):
        x = torch.tensor([[10, 11], [20, 21], [30, 31]])
        y = ["a", "b", "c"]

        sequence_dict = {"x": x.clone(), "y": y}

        shuffled = shuffle_sequence_dict(sequence_dict)

        # Check that the list y is shuffled in the same order as x
        for i in range(3):
            x_row = shuffled["x"][i]
            y_val = shuffled["y"][i]

            if torch.equal(x_row, torch.tensor([10, 11])):
                assert y_val == "a"
            elif torch.equal(x_row, torch.tensor([20, 21])):
                assert y_val == "b"
            elif torch.equal(x_row, torch.tensor([30, 31])):
                assert y_val == "c"
            else:
                pytest.fail("Unexpected x row in shuffled output.")


class TestSplitTensorDict(TrlTestCase):
    def test_split_equal_chunks(self):
        x = torch.arange(12).reshape(6, 2)
        y = torch.arange(6).reshape(6, 1)
        tensor_dict = {"x": x, "y": y}

        result = split_tensor_dict(tensor_dict, 3)

        expected_x_chunks = torch.chunk(x, 3, dim=0)
        expected_y_chunks = torch.chunk(y, 3, dim=0)
        assert len(result) == 3
        for i in range(3):
            assert torch.equal(result[i]["x"], expected_x_chunks[i])
            assert torch.equal(result[i]["y"], expected_y_chunks[i])

    def test_with_none_tensor(self):
        x = torch.arange(12).reshape(6, 2)
        tensor_dict = {"x": x, "y": None}

        result = split_tensor_dict(tensor_dict, 2)

        expected_x_chunks = torch.chunk(x, 2, dim=0)
        assert len(result) == 2
        for i in range(2):
            assert torch.equal(result[i]["x"], expected_x_chunks[i])
            assert result[i]["y"] is None

    def test_with_scalar(self):
        x = torch.arange(12).reshape(6, 2)
        tensor_dict = {"x": x, "y": torch.tensor(1)}

        result = split_tensor_dict(tensor_dict, 2)

        expected_x_chunks = torch.chunk(x, 2, dim=0)
        assert len(result) == 2
        for i in range(2):
            assert torch.equal(result[i]["x"], expected_x_chunks[i])
            assert torch.equal(result[i]["y"], torch.tensor(1))


class TestSplitPixelValuesByGrid(TrlTestCase):
    def test_split_correctly_0(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            "num_images": [1, 1],
            "pixel_values": torch.arange(8 * 3).reshape(8, 3),  # Shape: [8, 3]
        }
        result = split_pixel_values_by_grid(batch)
        assert isinstance(result["pixel_values"], list)
        assert len(result["pixel_values"]) == 2
        assert torch.equal(result["pixel_values"][0], batch["pixel_values"][:4])
        assert torch.equal(result["pixel_values"][1], batch["pixel_values"][4:])
        assert isinstance(result["image_grid_thw"], list)
        assert len(result["image_grid_thw"]) == 2
        assert torch.equal(result["image_grid_thw"][0], torch.tensor([[1, 2, 2]]))
        assert torch.equal(result["image_grid_thw"][1], torch.tensor([[1, 2, 2]]))

    def test_split_correctly_1(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4]]),
            "num_images": [1, 1],
            "pixel_values": torch.arange(12 * 3).reshape(12, 3),  # Shape: [12, 3]
        }
        result = split_pixel_values_by_grid(batch)
        assert isinstance(result["pixel_values"], list)
        assert len(result["pixel_values"]) == 2
        assert torch.equal(result["pixel_values"][0], batch["pixel_values"][:4])
        assert torch.equal(result["pixel_values"][1], batch["pixel_values"][4:12])
        assert isinstance(result["image_grid_thw"], list)
        assert len(result["image_grid_thw"]) == 2
        assert torch.equal(result["image_grid_thw"][0], torch.tensor([[1, 2, 2]]))
        assert torch.equal(result["image_grid_thw"][1], torch.tensor([[1, 2, 4]]))

    def test_missing_keys(self):
        batch = {"pixel_values": torch.tensor([1.0])}
        result = split_pixel_values_by_grid(batch)
        assert result == batch

    def test_mismatched_length(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 2, 1]]),  # Total = 8
            "num_images": [1, 1],
            "pixel_values": torch.randn(3, 5),  # Only 3 rows
        }
        with pytest.raises(ValueError):
            split_pixel_values_by_grid(batch)

    def test_multi_images(self):
        batch = {
            "image_grid_thw": torch.tensor([[1, 1, 2], [1, 2, 2], [1, 2, 1]]),  # Total = 8
            "num_images": [1, 2],
            "pixel_values": torch.arange(8 * 3).reshape(8, 3),  # Shape: [8, 3]
        }
        result = split_pixel_values_by_grid(batch)
        assert isinstance(result["pixel_values"], list)
        assert len(result["pixel_values"]) == 2
        assert torch.equal(result["pixel_values"][0], batch["pixel_values"][:2])
        assert torch.equal(result["pixel_values"][1], batch["pixel_values"][2:])
        assert isinstance(result["image_grid_thw"], list)
        assert len(result["image_grid_thw"]) == 2
        assert torch.equal(result["image_grid_thw"][0], torch.tensor([[1, 1, 2]]))
        assert torch.equal(result["image_grid_thw"][1], torch.tensor([[1, 2, 2], [1, 2, 1]]))


class TestUnsplitPixelValuesByGrid(TrlTestCase):
    def test_unsplit_correctly(self):
        pixel_values = [torch.randn(4, 5), torch.randn(2, 5)]
        pixel_values_merged = torch.cat(pixel_values, dim=0)
        image_grid_thw = [torch.tensor([[1, 2, 2]]), torch.tensor([[1, 2, 1]])]
        image_grid_thw_merged = torch.cat(image_grid_thw, dim=0)
        batch = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "other_key": torch.tensor([1])}
        result = unsplit_pixel_values_by_grid(batch)
        assert isinstance(result["pixel_values"], torch.Tensor)
        torch.testing.assert_close(result["pixel_values"], pixel_values_merged)
        assert isinstance(result["image_grid_thw"], torch.Tensor)
        assert torch.equal(result["image_grid_thw"], image_grid_thw_merged)
        assert "other_key" in result

    def test_no_op_if_not_list(self):
        original = torch.randn(5, 3)
        batch = {"pixel_values": original}
        result = unsplit_pixel_values_by_grid(batch)
        assert torch.equal(result["pixel_values"], original)


class TestForwardMaskedLogits:
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-CohereForCausalLM",
            "trl-internal-testing/tiny-Cohere2ForCausalLM",
            "trl-internal-testing/tiny-DeepseekV3ForCausalLM",
            "trl-internal-testing/tiny-DeepseekV3ForCausalLM-0528",
            "trl-internal-testing/tiny-Gemma2ForCausalLM",
            "trl-internal-testing/tiny-GemmaForCausalLM",
            "trl-internal-testing/tiny-Glm4MoeForCausalLM",
            "trl-internal-testing/tiny-GptOssForCausalLM",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
            "trl-internal-testing/tiny-LlamaForCausalLM-3",
            "trl-internal-testing/tiny-MistralForCausalLM-0.1",
            "trl-internal-testing/tiny-MistralForCausalLM-0.2",
            "trl-internal-testing/tiny-Phi3ForCausalLM",
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
        ],
    )
    def test_llm(self, model_id):
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map=device)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 8), device=device)
        logits_mask = torch.tensor(
            [[1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1]],
            device=device,
        )

        full_outputs = model(input_ids=input_ids)
        masked_outputs = forward_masked_logits(model, logits_mask, input_ids=input_ids)

        torch.testing.assert_close(
            masked_outputs.flat_logits,
            full_outputs.logits[logits_mask.bool()],
        )

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Gemma3ForConditionalGeneration",
            "trl-internal-testing/tiny-Idefics2ForConditionalGeneration",
            "trl-internal-testing/tiny-Idefics3ForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaForConditionalGeneration",
            "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            # "trl-internal-testing/tiny-SmolVLMForConditionalGeneration", seems not to support bf16 properly
            pytest.param(
                "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration",
                marks=[
                    pytest.mark.skipif(
                        Version(transformers.__version__) < Version("4.57.0"),
                        reason="Qwen3-VL series were introduced in transformers-4.57.0",
                    ),
                    pytest.mark.xfail(
                        Version("5.0.0") <= Version(transformers.__version__) < Version("5.1.0"),
                        reason="Upstream transformers bug (transformers#43334) in 5.0.x; fixed in 5.1.0",
                    ),
                ],
            ),
        ],
    )
    def test_vlm(self, model_id):
        device = torch.device("cuda")
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto", device_map=device)
        input_ids = torch.randint(0, model.config.text_config.vocab_size, (2, 8), device=device)
        logits_mask = torch.tensor(
            [[1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0, 1]],
            device=device,
        )

        full_outputs = model(input_ids=input_ids)
        masked_outputs = forward_masked_logits(model, logits_mask, input_ids=input_ids)

        torch.testing.assert_close(
            masked_outputs.flat_logits,
            full_outputs.logits[logits_mask.bool()],
        )
