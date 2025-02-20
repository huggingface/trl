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

import json
import os
import sys
import tempfile
import unittest

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments
from transformers.testing_utils import require_peft, require_wandb
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_peft_available

from tests.testing_utils import require_comet, require_mergekit
from trl import BasePairwiseJudge, DPOConfig, DPOTrainer, LogCompletionsCallback, MergeModelCallback, WinRateCallback
from trl.mergekit_utils import MergeConfig


if is_peft_available():
    from peft import LoraConfig


class HalfPairwiseJudge(BasePairwiseJudge):
    """Naive pairwise judge that always returns [1, 0] for two prompts"""

    def judge(self, prompts, completions, shuffle_order=True, return_scores=False):
        # just check that the batch size is 2
        assert len(prompts) == 2
        if return_scores:
            return [0.3, 0.9]
        return [1, 0]


class TrainerWithRefModel(Trainer):
    # This is a dummy class to test the callback. Compared to the Trainer class, it only has an additional
    # ref_model attribute
    def __init__(self, model, ref_model, args, train_dataset, eval_dataset, processing_class):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
        )
        self.ref_model = ref_model


class WinRateCallbackTester(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")
        dataset["train"] = dataset["train"].select(range(8))
        self.expected_winrates = [
            {"eval_win_rate": 0.5, "epoch": 0.0, "step": 0},
            {"eval_win_rate": 0.5, "epoch": 0.5, "step": 2},
            {"eval_win_rate": 0.5, "epoch": 1.0, "step": 4},
            {"eval_win_rate": 0.5, "epoch": 1.5, "step": 6},
            {"eval_win_rate": 0.5, "epoch": 2.0, "step": 8},
            {"eval_win_rate": 0.5, "epoch": 2.5, "step": 10},
            {"eval_win_rate": 0.5, "epoch": 3.0, "step": 12},
        ]

        def tokenize_function(examples):
            out = self.tokenizer(examples["prompt"], padding="max_length", max_length=16, truncation=True)
            out["labels"] = out["input_ids"].copy()
            return out

        self.dataset = dataset.map(tokenize_function, batched=True)

        self.generation_config = GenerationConfig(max_length=32)
        self.judge = HalfPairwiseJudge()

    def test_basic(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = TrainerWithRefModel(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()
            winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
            self.assertListEqual(winrate_history, self.expected_winrates)

    def test_without_ref_model(self):
        # Same as before, but without the ref_model attribute. It should use the model attribute instead
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()
            winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
            self.assertListEqual(winrate_history, self.expected_winrates)

    def test_soft_judge(self):
        """Test that the soft judge functionality works correctly"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = TrainerWithRefModel(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config, use_soft_judge=True
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()

            # Expected values based on judge returning [0.3, 0.9] for each pair
            expected_soft_winrates = [
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 0.0, "step": 0},
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 0.5, "step": 2},
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 1.0, "step": 4},
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 1.5, "step": 6},
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 2.0, "step": 8},
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 2.5, "step": 10},
                {"eval_avg_win_prob": 0.4, "eval_win_rate": 0.5, "epoch": 3.0, "step": 12},
            ]

            winrate_history = [
                {k: h[k] for k in ["eval_avg_win_prob", "eval_win_rate", "epoch", "step"]}
                for h in trainer.state.log_history
                if "eval_avg_win_prob" in h
            ]
            self.assertListEqual(winrate_history, expected_soft_winrates)

    @require_peft
    def test_lora(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.add_adapter(peft_config)
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                eval_strategy="steps",
                eval_steps=2,  # evaluate every 2 steps
                per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
                per_device_eval_batch_size=2,
                report_to="none",
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset["train"],
                eval_dataset=self.dataset["test"],
                processing_class=self.tokenizer,
            )
            win_rate_callback = WinRateCallback(
                judge=self.judge, trainer=trainer, generation_config=self.generation_config
            )
            trainer.add_callback(win_rate_callback)
            trainer.train()
            winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
            self.assertListEqual(winrate_history, self.expected_winrates)


class LogCompletionsCallbackTester(unittest.TestCase):
    def setUp(self):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
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
            self.assertIn("step", completions["columns"])
            self.assertIn("prompt", completions["columns"])
            self.assertIn("completion", completions["columns"])

            # Check that the prompt is in the log
            self.assertIn(self.dataset["test"][0]["prompt"], completions["data"][0])

    @require_comet
    def test_basic_comet(self):
        import comet_ml

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
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


# On Windows, temporary directory cleanup fails when using the MergeModelCallback.
# This is not an issue with the functionality of the code itself, but it can cause the test to fail
# due to unhandled cleanup errors. Python 3.10 introduces the `ignore_cleanup_errors` argument to
# mitigate this. As a result, this test is skipped for Python versions below 3.10.
@require_mergekit
@unittest.skipIf(
    sys.version_info < (3, 10),
    "Test fails on Python versions lower than 3.10, but its only related to cleanup errors with temp dir.",
)
class MergeModelCallbackTester(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

    def test_callback(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                report_to="none",
                save_strategy="steps",
                save_steps=1,
            )
            config = MergeConfig()
            merge_callback = MergeModelCallback(config)
            trainer = DPOTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                processing_class=self.tokenizer,
                callbacks=[merge_callback],
            )
            trainer.train()
            last_checkpoint = get_last_checkpoint(tmp_dir)
            merged_path = os.path.join(last_checkpoint, "merged")
            self.assertTrue(os.path.isdir(merged_path), "Merged folder does not exist in the last checkpoint.")

    def test_every_checkpoint(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                report_to="none",
                save_strategy="steps",
                save_steps=1,
            )
            config = MergeConfig()
            merge_callback = MergeModelCallback(config, merge_at_every_checkpoint=True)
            trainer = DPOTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                processing_class=self.tokenizer,
                callbacks=[merge_callback],
            )
            trainer.train()

            checkpoints = sorted(
                [os.path.join(tmp_dir, cp) for cp in os.listdir(tmp_dir) if cp.startswith("checkpoint-")]
            )

            for checkpoint in checkpoints:
                merged_path = os.path.join(checkpoint, "merged")
                self.assertTrue(
                    os.path.isdir(merged_path), f"Merged folder does not exist in checkpoint {checkpoint}."
                )
