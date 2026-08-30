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

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments

from trl import BEMACallback, LogCompletionsCallback
from trl.trainer.callbacks import SyncRefModelCallback

from .testing_utils import TrlTestCase, require_comet, require_peft, require_wandb


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


@require_peft
class TestSyncRefModelCallbackAdapterPairing(TrlTestCase):
    """`_sync_ref_adapter` pairs a "default" parameter with its "ref" counterpart by name."""

    @staticmethod
    def _peft_model_with_base_parameter_named_default():
        from peft import LoraConfig, get_peft_model

        # A base model owning both a parameter and a submodule literally called "default". PEFT reserves neither, so
        # the adapter parameters here read `...default.proj.lora_A.default.weight`, with a base component and an
        # adapter component of the same name, and the base parameter reads `...default` with no component after it.
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.proj(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.default = Inner()
                self.register_parameter("default_bias", torch.nn.Parameter(torch.ones(2)))

            def forward(self, x):
                return self.default(x)

        config = LoraConfig(target_modules=["proj"])
        model = get_peft_model(Model(), config)
        model.add_adapter("ref", config)
        return model

    def test_base_module_and_parameter_named_default_are_not_paired(self):
        model = self._peft_model_with_base_parameter_named_default()
        adapter_names = [n for n, _ in model.named_parameters() if n.split(".").count("default") == 2]
        assert adapter_names, "the fixture must produce a path with both a base and an adapter 'default' component"

        SyncRefModelCallback._sync_ref_adapter(model, alpha=1.0)  # must not raise

        # Only the adapter component may be rewritten; the base module keeps its name.
        for name in adapter_names:
            parts = name.split(".")
            index = len(parts) - 1 - parts[::-1].index("default")
            model.get_parameter(".".join(parts[:index] + ["ref"] + parts[index + 1 :]))

    def test_adapter_parameters_follow_the_ema_equation(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        config = LoraConfig(r=4, target_modules=["q_proj"], trainable_token_indices=[0, 1])
        model = get_peft_model(
            AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"), config
        )
        model.add_adapter("ref", config)

        alpha = 0.25
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "default" in name.split("."):
                    param.fill_(4.0)
                elif "ref" in name.split("."):
                    param.fill_(8.0)

        SyncRefModelCallback._sync_ref_adapter(model, alpha=alpha)

        expected = (1.0 - alpha) * 8.0 + alpha * 4.0
        checked = 0
        for name, param in model.named_parameters():
            if "ref" in name.split("."):
                assert torch.allclose(param, torch.full_like(param, expected)), name
                checked += 1
        assert checked, "no reference parameter was examined"

    def test_terminal_adapter_parameter_is_paired(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        # `trainable_token_indices` deltas live in a `ParameterDict` keyed by adapter, so the adapter name is the
        # final path component with nothing after it.
        config = LoraConfig(r=4, target_modules=["q_proj"], trainable_token_indices=[0, 1])
        model = get_peft_model(
            AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"), config
        )
        model.add_adapter("ref", config)

        delta = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
        with torch.no_grad():
            model.get_parameter(f"{delta}.default").fill_(7.0)
            model.get_parameter(f"{delta}.ref").fill_(0.0)

        SyncRefModelCallback._sync_ref_adapter(model, alpha=1.0)

        assert torch.equal(
            model.get_parameter(f"{delta}.ref"), torch.full_like(model.get_parameter(f"{delta}.ref"), 7.0)
        )

    def test_modules_to_save_with_a_child_named_default_is_paired(self):
        from peft import LoraConfig, get_peft_model

        # `modules_to_save` wraps the saved module in a `ModuleDict` keyed by adapter, so a saved module that itself
        # contains a child called "default" produces two "default" components and the adapter key is the FIRST one.
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.default = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.default(x)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.block = Block()

            def forward(self, x):
                return self.block(self.proj(x))

        config = LoraConfig(target_modules=["proj"], modules_to_save=["block"])
        model = get_peft_model(Model(), config)
        model.add_adapter("ref", config)

        policy = "base_model.model.block.modules_to_save.default.default.weight"
        reference = "base_model.model.block.modules_to_save.ref.default.weight"
        with torch.no_grad():
            model.get_parameter(policy).fill_(5.0)
            model.get_parameter(reference).fill_(0.0)

        SyncRefModelCallback._sync_ref_adapter(model, alpha=1.0)

        assert torch.equal(model.get_parameter(reference), torch.full_like(model.get_parameter(reference), 5.0)), (
            "a modules_to_save parameter was skipped because the adapter key was not the last 'default' component"
        )

    def test_base_module_dict_without_a_ref_key_is_not_paired(self):
        from peft import LoraConfig, get_peft_model

        # A `ModuleDict` in the base model may hold a "default" key that PEFT knows nothing about, so there is no
        # "ref" counterpart to pair it with.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)
                self.heads = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 4)})

            def forward(self, x):
                return self.heads["default"](self.proj(x))

        config = LoraConfig(target_modules=["proj"])
        model = get_peft_model(Model(), config)
        model.add_adapter("ref", config)
        head = model.get_parameter("base_model.model.heads.default.weight").clone()

        SyncRefModelCallback._sync_ref_adapter(model, alpha=1.0)  # must not raise

        assert torch.equal(model.get_parameter("base_model.model.heads.default.weight"), head)
