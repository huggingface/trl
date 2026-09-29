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

import json
import os
import types
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, call, patch

import pytest
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
)

from trl import BEMACallback, LogCompletionsCallback, WeaveCallback
from trl.trainer.callbacks import _generate_completions

from .testing_utils import TrlTestCase, require_comet, require_wandb


class TestLogCompletionsCallback(TrlTestCase):
    def setup_method(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # `Trainer` realigns the configs at train time, so mirror the pad token as the TRL trainers do at init.
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")
        dataset["train"] = dataset["train"].select(range(8))

        def tokenize_function(examples):
            out = self.tokenizer(examples["prompt"], padding="max_length", max_length=16, truncation=True)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(tokenize_function, batched=True)

        self.generation_config = GenerationConfig(max_length=32)

    @require_wandb
    def test_basic_wandb(self):
        import wandb

        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=2,  # evaluate every 2 steps
            per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
            per_device_eval_batch_size=2,
            report_to="wandb",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer,
        )
        completions_callback = LogCompletionsCallback(trainer, self.generation_config, num_prompts=2)
        trainer.add_callback(completions_callback)
        trainer.train()

        # Get the current run
        completions_path = wandb.run.summary.completions["path"]
        json_path = os.path.join(wandb.run.dir, completions_path)
        with open(json_path) as f:
            completions = json.load(f)

        # Check that the columns are correct
        assert "step" in completions["columns"]
        assert "prompt" in completions["columns"]
        assert "completion" in completions["columns"]

        # Check that the prompt is in the log
        assert self.dataset["test"][0]["prompt"] in completions["data"][0]

    @require_comet
    def test_basic_comet(self):
        import comet_ml

        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            eval_strategy="steps",
            eval_steps=2,  # evaluate every 2 steps
            per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
            per_device_eval_batch_size=2,
            report_to="comet_ml",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            processing_class=self.tokenizer,
        )
        completions_callback = LogCompletionsCallback(trainer, self.generation_config, num_prompts=2)
        trainer.add_callback(completions_callback)
        trainer.train()

        # close experiment to make sure all pending data are flushed
        experiment = comet_ml.get_running_experiment()
        assert experiment is not None
        experiment.end()

        # get experiment assets and check that all required tables was logged
        steps = len(self.dataset["train"]) + len(self.dataset["test"])
        tables_logged = int(steps / 2) + 1  # +1 to include zero step

        api_experiment = comet_ml.APIExperiment(previous_experiment=experiment.id)
        tables = api_experiment.get_asset_list("dataframe")
        assert tables is not None
        assert len(tables) == tables_logged
        assert all(table["fileName"] == "completions.csv" for table in tables)

    def test_padding_side_restored(self):
        # Same repro as https://github.com/huggingface/trl/issues/6663
        tokenizer = Mock()
        tokenizer.padding_side = "right"
        trainer = types.SimpleNamespace(
            eval_dataset={"prompt": ["prompt"]},
            accelerator=types.SimpleNamespace(
                is_main_process=False,
                split_between_processes=lambda prompts: nullcontext(prompts),
            ),
            model_wrapped=Mock(),
        )
        args = types.SimpleNamespace(per_device_eval_batch_size=1, report_to=[])
        state = types.SimpleNamespace(global_step=1, eval_steps=1)
        callback = LogCompletionsCallback(trainer, freq=1)

        with patch("trl.trainer.callbacks._generate_completions", return_value=["completion"]):
            callback.on_step_end(args, state, None, processing_class=tokenizer)
        assert tokenizer.padding_side == "right"

        state.global_step = 2
        with patch("trl.trainer.callbacks._generate_completions", side_effect=RuntimeError("generation failed")):
            with pytest.raises(RuntimeError, match="generation failed"):
                callback.on_step_end(args, state, None, processing_class=tokenizer)
        assert tokenizer.padding_side == "right"

        weave_callback = WeaveCallback(trainer)
        weave_callback._weave_initialized = True
        state.global_step = 3
        with patch("trl.trainer.callbacks._generate_completions", return_value=["completion"]):
            weave_callback.on_evaluate(args, state, None, processing_class=tokenizer)
        assert tokenizer.padding_side == "right"

    def test_generate_completions_restores_padding_side(self):
        sides = []

        class Batch(dict):
            def to(self, device):
                return self

        class _Tokenizer(PreTrainedTokenizerBase):
            def __init__(self):
                self.padding_side = "right"

            def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
                sides.append(self.padding_side)
                batch = Batch(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]))
                batch.input_ids = batch["input_ids"]
                return batch

            def decode(self, generation, skip_special_tokens=True):
                return "ok"

        tokenizer = _Tokenizer()
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 7], [4, 5, 0, 8]])

        @contextmanager
        def unwrap(model, accelerator, *args, **kwargs):
            yield model

        with patch("trl.trainer.callbacks.unwrap_model_for_generation", unwrap):
            completions = _generate_completions(
                ["a", "bb"], model, tokenizer, accelerator=Mock(), generation_config=None, batch_size=2
            )
        assert completions == ["ok", "ok"]
        assert sides == ["left"]
        assert tokenizer.padding_side == "right"

        model.generate.side_effect = RuntimeError("generation failed")
        with patch("trl.trainer.callbacks.unwrap_model_for_generation", unwrap):
            with pytest.raises(RuntimeError, match="generation failed"):
                _generate_completions(["a"], model, tokenizer, accelerator=Mock(), generation_config=None)
        assert tokenizer.padding_side == "right"

    def test_generate_completions_restores_processor_padding_side(self):
        # ProcessorMixin has no padding_side; it lives on the nested tokenizer.
        sides = []

        class Batch(dict):
            def to(self, device):
                return self

        class _Tokenizer(PreTrainedTokenizerBase):
            def __init__(self):
                self.padding_side = "right"

            def decode(self, generation, skip_special_tokens=True):
                return "ok"

        class _Processor(ProcessorMixin):
            def __init__(self):
                self.tokenizer = _Tokenizer()

            def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
                sides.append(self.tokenizer.padding_side)
                batch = Batch(input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]))
                batch.input_ids = batch["input_ids"]
                return batch

            def decode(self, generation, skip_special_tokens=True):
                return self.tokenizer.decode(generation, skip_special_tokens=skip_special_tokens)

        processor = _Processor()
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 7], [4, 5, 0, 8]])

        @contextmanager
        def unwrap(model, accelerator, *args, **kwargs):
            yield model

        with patch("trl.trainer.callbacks.unwrap_model_for_generation", unwrap):
            completions = _generate_completions(
                ["a", "bb"], model, processor, accelerator=Mock(), generation_config=None, batch_size=2
            )
        assert completions == ["ok", "ok"]
        assert sides == ["left"]
        assert processor.tokenizer.padding_side == "right"
        assert not hasattr(processor, "padding_side")

        model.generate.side_effect = RuntimeError("generation failed")
        with patch("trl.trainer.callbacks.unwrap_model_for_generation", unwrap):
            with pytest.raises(RuntimeError, match="generation failed"):
                _generate_completions(["a"], model, processor, accelerator=Mock(), generation_config=None)
        assert processor.tokenizer.padding_side == "right"

    def test_generate_completions_rejects_unknown_processing_class(self):
        class _Bare:
            def __init__(self):
                self.padding_side = "right"
                self.tokenizer = types.SimpleNamespace(padding_side="right")

        bare = _Bare()
        with pytest.raises(TypeError, match="PreTrainedTokenizerBase"):
            _generate_completions(["a"], Mock(), bare, accelerator=Mock(), generation_config=None)
        assert bare.padding_side == "right"
        assert bare.tokenizer.padding_side == "right"

    def test_train_batches_stay_right_padded_after_callback(self):
        # Uneven lengths, so DataCollatorForLanguageModeling actually pads via tokenizer.padding_side.
        self.tokenizer.padding_side = "right"
        token_id = self.tokenizer.eos_token_id
        lengths = [1, 2, 3, 4]
        train_dataset = Dataset.from_dict(
            {
                "input_ids": [[token_id] * length for length in lengths],
                "attention_mask": [[1] * length for length in lengths],
            }
        )
        eval_dataset = Dataset.from_dict({"prompt": ["Say hi.", "Say bye."]})
        masks = []

        class RecordingCollator(DataCollatorForLanguageModeling):
            def __call__(self, features):
                batch = super().__call__(features)
                masks.append(batch["attention_mask"].detach().cpu().clone())
                return batch

        training_args = TrainingArguments(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            max_steps=2,
            learning_rate=1e-5,
            save_strategy="no",
            report_to="none",
            seed=0,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=RecordingCollator(self.tokenizer, mlm=False),
        )
        trainer.add_callback(LogCompletionsCallback(trainer, self.generation_config, freq=1))
        trainer.train()

        assert self.tokenizer.padding_side == "right"
        # One batch is collated after on_step_end. Left padding would put a 0 in column 0.
        assert len(masks) >= 2
        for mask in masks:
            assert (mask == 0).any()
            assert torch.all(mask[:, 0] == 1)


class TestBEMACallback(TrlTestCase):
    def setup_method(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        # `Trainer` realigns the configs at train time, so mirror the pad token as the TRL trainers do at init.
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

        def tokenize_function(examples, tokenizer):
            out = tokenizer(examples["text"], padding="max_length", max_length=17)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(
            tokenize_function, fn_kwargs={"tokenizer": self.tokenizer}, remove_columns=["text"], batched=True
        )

    def test_model_saved(self):
        """Test that BEMACallback saves the BEMA model."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            processing_class=self.tokenizer,
            callbacks=[bema_callback],
        )
        trainer.train()

        # Check that the BEMA model was saved and can be loaded
        bema_path = os.path.join(self.tmp_dir, "bema")
        assert os.path.isdir(bema_path), "BEMA directory was not created"
        AutoModelForCausalLM.from_pretrained(bema_path)

    def test_update_frequency_0(self):
        """Test that BEMA callback respects the update frequency."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2)

        with patch.object(bema_callback, "_update_bema_weights") as mock_update:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                processing_class=self.tokenizer,
                callbacks=[bema_callback],
            )

            trainer.train()

            # Total 9 steps (17 samples, batch size 8, 3 epochs).
            # BEMA starts after step 0 and updates every 2 steps → updates at 2, 4, 5, 8
            assert mock_update.call_args_list == [call(2), call(4), call(6), call(8)]

    def test_update_frequency_1(self):
        """Test that BEMA callback respects the update frequency."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=3)

        with patch.object(bema_callback, "_update_bema_weights") as mock_update:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                processing_class=self.tokenizer,
                callbacks=[bema_callback],
            )

            trainer.train()

            # Total 9 steps (17 samples, batch size 8, 3 epochs).
            # BEMA starts after step 0 and updates every 3 steps → updates at 3, 6, 9
            assert mock_update.call_args_list == [call(3), call(6), call(9)]

    def test_update_frequency_2(self):
        """Test that BEMA callback respects the update frequency."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, update_after=3)

        with patch.object(bema_callback, "_update_bema_weights") as mock_update:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                processing_class=self.tokenizer,
                callbacks=[bema_callback],
            )

            trainer.train()

            # Total 9 steps (17 samples, batch size 8, 3 epochs).
            # BEMA starts after step 3 and updates every 2 steps → updates at 5, 7, 9
            assert mock_update.call_args_list == [call(5), call(7), call(9)]

    def test_bias_power_zero(self):
        """Test that BEMACallback works with bias_power=0.0 (maximum, undecayed bias-correction)."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, bias_power=0.0)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            processing_class=self.tokenizer,
            callbacks=[bema_callback],
        )
        trainer.train()

    def test_no_ema(self):
        """Test that BEMACallback works without EMA updates."""
        training_args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        bema_callback = BEMACallback(update_freq=2, ema_power=0.0)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            processing_class=self.tokenizer,
            callbacks=[bema_callback],
        )
        trainer.train()
