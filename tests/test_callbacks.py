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
from unittest.mock import call, patch

import pytest
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from trl import BEMACallback, LogCompletionsCallback
from trl.trainer.callbacks import _generate_completions

from .testing_utils import TrlTestCase, require_comet, require_wandb


class TestCompletionGenerationMode(TrlTestCase):
    def setup_method(self):
        vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3, "world": 4}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]", padding_side="left"
        )
        self.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocab),
                n_positions=16,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.8,
                embd_pdrop=0.8,
                attn_pdrop=0.8,
                pad_token_id=0,
                eos_token_id=2,
                bos_token_id=3,
            )
        )
        self.generation_config = GenerationConfig(max_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=2)

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_generation_disables_dropout_and_restores_mode(self, training, batch_size):
        self.model.train(training)
        dropout_modes = []
        with self.model.transformer.h[0].mlp.dropout.register_forward_pre_hook(
            lambda layer, inputs: dropout_modes.append(layer.training)
        ):
            completions = _generate_completions(
                ["hello world", "hello"],
                self.model,
                self.tokenizer,
                Accelerator(cpu=True),
                self.generation_config,
                batch_size=batch_size,
            )
        assert len(completions) == 2
        assert dropout_modes and not any(dropout_modes)
        assert self.model.training == training

    @pytest.mark.parametrize("training", [True, False])
    def test_generation_error_restores_mode(self, training):
        self.model.train(training)

        def fail_generation(*args, **kwargs):
            assert not self.model.training
            raise RuntimeError("generation failed")

        with patch.object(self.model, "generate", side_effect=fail_generation):
            with pytest.raises(RuntimeError, match="generation failed"):
                _generate_completions(
                    ["hello world"], self.model, self.tokenizer, Accelerator(cpu=True), self.generation_config
                )
        assert self.model.training == training

    def test_logging_after_training_step_uses_eval_mode(self):
        train_dataset = Dataset.from_dict(
            {"input_ids": [[3, 4, 3]], "attention_mask": [[1, 1, 1]], "labels": [[3, 4, 3]]}
        )
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=1,
                report_to="none",
                save_strategy="no",
                disable_tqdm=True,
            ),
            train_dataset=train_dataset,
            eval_dataset=Dataset.from_dict({"prompt": ["hello world"]}),
            processing_class=self.tokenizer,
        )
        callback = LogCompletionsCallback(trainer, self.generation_config, freq=1)
        trainer.add_callback(callback)
        dropout_modes = []
        with self.model.transformer.h[0].mlp.dropout.register_forward_pre_hook(
            lambda layer, inputs: dropout_modes.append(layer.training)
        ):
            trainer.train()
        # The training forward uses dropout, but completion generation must not.
        assert len(dropout_modes) > 1
        assert dropout_modes[0]
        assert not any(dropout_modes[1:])
        assert self.model.training
        assert len(callback.table) == 1


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
