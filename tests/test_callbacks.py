# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from unittest.mock import call, patch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments

from trl import BEMACallback, LogCompletionsCallback

from .testing_utils import TrlTestCase, require_comet, require_wandb


class TestLogCompletionsCallback(TrlTestCase):
    def setup_method(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer.pad_token = self.tokenizer.eos_token
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


class TestBEMACallback(TrlTestCase):
    def setup_method(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer.pad_token = self.tokenizer.eos_token
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

    def test_no_bema(self):
        """Test that BEMACallback works without BEMA updates."""
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
