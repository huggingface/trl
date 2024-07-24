# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

import pytest
import torch
from datasets import Dataset, features
from parameterized import parameterized
from PIL import Image
from pytest import mark
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

from trl import DPOConfig, DPOTrainer, FDivergenceType

from .testing_utils import require_bitsandbytes, require_no_wandb, require_peft


class DPOTrainerTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.ref_model = AutoModelForCausalLM.from_pretrained(cls.model_id)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token

        # get t5 as seq2seq example:
        model_id = "trl-internal-testing/T5ForConditionalGeneration-correct-vocab-calibrated"
        cls.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_ref_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        cls.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # get idefics2 model
        model_id = "trl-internal-testing/tiny-random-idefics2"
        cls.idefics2_model = AutoModelForVision2Seq.from_pretrained(model_id)
        cls.idefics2_ref_model = AutoModelForVision2Seq.from_pretrained(model_id)
        cls.idefics2_processor = AutoProcessor.from_pretrained(model_id)

    def _init_dummy_dataset(self):
        # fmt: off
        dummy_dataset_dict = {
            "prompt": [
                "hello",
                "how are you",
                "What is your name?",
                "What is your name?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "Which is the best programming language?",
                "[INST] How is the stock price? [/INST]",
                "[INST] How is the stock price? [/INST] ",
            ],
            "chosen": [
                "hi nice to meet you",
                "I am fine",
                "My name is Mary",
                "My name is Mary",
                "Python",
                "Python",
                "Python",
                "$46 as of 10am EST",
                "46 as of 10am EST",
            ],
            "rejected": [
                "leave me alone",
                "I am not fine",
                "Whats it to you?",
                "I dont have a name",
                "Javascript",
                "C++",
                "Java",
                " $46 as of 10am EST",
                " 46 as of 10am EST",
            ],
        }
        # fmt: on
        return Dataset.from_dict(dummy_dataset_dict)

    def _init_dummy_image_dataset(self):
        # fmt: off
        dummy_dataset_dict = {
            "images": [
                [Image.new("RGB", (100, 50), color="black")],
                # None,
                # [Image.new("RGB", (100, 100), color="blue"), Image.new("RGB", (150, 50), color="red")],
                [Image.new("RGB", (200, 100), color="green")],
                # [Image.new("RGB", (150, 150), color="yellow"), Image.new("RGB", (50, 150), color="purple")],
                [Image.new("RGB", (80, 120), color="gray")],
                [Image.new("RGB", (120, 80), color="pink")],
            ],
            "prompt": [
                "<image> Hello",
                # "How are you?",
                # "<image><image> Let's chat",
                "<image> Good morning",
                # "<image><image> What's up?",
                "Can you see this? <image>",
                "Here is something interesting: <image>",
            ],
            "chosen": [
                "Hi nice to meet you!",
                # "I'm doing well, thank you!",
                # "Sure, let's talk!",
                "Good morning to you too!",
                # "Not much, just working.",
                "Yes, I can see it clearly.",
                "That's quite interesting indeed.",
            ],
            "rejected": [
                "Leave me alone!",
                # "I'm not interested.",
                # "I don't want to chat.",
                "I'm still sleepy.",
                # "Busy right now, talk later.",
                "No, I can't see it.",
                "I'm not sure what that is.",
            ],
        }
        # fmt: on
        f = features.Features(
            {
                "images": features.Sequence(features.Image(decode=True)),  # datasets handles badly sequence of images
                "prompt": features.Value("string"),
                "chosen": features.Value("string"),
                "rejected": features.Value("string"),
            }
        )
        return Dataset.from_dict(dummy_dataset_dict, features=f)

    @parameterized.expand(
        [
            ["gpt2", "sigmoid", True],
            ["t5", "hinge", False],
            ["gpt2", "ipo", False],
            ["t5", "ipo", True],
            ["gpt2", "aot_pair", True],
            ["t5", "aot_pair", False],
            ["gpt2", "aot", True],
            ["t5", "aot", False],
            ["gpt2", "bco_pair", False],
            ["t5", "bco_pair", True],
            ["gpt2", "sppo_hard", False],
            ["t5", "sppo_hard", True],
            ["gpt2", "nca_pair", False],
            ["t5", "nca_pair", True],
            ["gpt2", "robust", True],
            ["gpt2", "exo_pair", False],
            ["t5", "exo_pair", True],
        ]
    )
    def test_dpo_trainer(self, name, loss_type, pre_compute):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute,
            )

            dummy_dataset = self._init_dummy_dataset()

            if name == "gpt2":
                model = self.model
                ref_model = self.ref_model
                tokenizer = self.tokenizer
            elif name == "t5":
                model = self.t5_model
                ref_model = self.t5_ref_model
                tokenizer = self.t5_tokenizer

            if name == "t5":
                self.skipTest("For some reason t5 does not compute gradients properly on tiny models")

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    @parameterized.expand(
        [
            ["sigmoid", True],
        ]
    )
    def test_vdpo_trainer(self, loss_type, pre_compute):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute,
            )

            dummy_dataset = self._init_dummy_image_dataset()

            model = self.idefics2_model
            ref_model = self.idefics2_ref_model
            processor = self.idefics2_processor

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                tokenizer=processor,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)

    def test_dpo_trainer_without_providing_ref_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=True,
                rpo_alpha=0.5,
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = DPOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.model.get_parameter(n)
                # check the params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.equal(param, new_param)

    @require_peft
    @mark.peft_test
    def test_dpo_trainer_without_providing_ref_model_with_lora(self):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=True,
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = DPOTrainer(
                model=self.model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # check the params have changed
            for n, param in previous_trainable_params.items():
                if "lora" in n:
                    new_param = trainer.model.get_parameter(n)
                    # check the params have changed - ignore 0 biases
                    if param.sum() != 0:
                        assert not torch.equal(param, new_param)

    def test_dpo_trainer_padding_token_is_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = None

            with self.assertRaisesRegex(
                ValueError,
                expected_regex=r"Padding is enabled, but the tokenizer is not configured with a padding token."
                r" Explicitly set `tokenizer.pad_token` \(e.g. `tokenizer.pad_token = tokenizer.eos_token`\)"
                r" before calling the trainer.",
            ):
                trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=dummy_dataset,
                    eval_dataset=dummy_dataset,
                )

                trainer.train()

    def test_dpo_trainer_w_dataset_num_proc(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                dataset_num_proc=5,
            )

            dummy_dataset = self._init_dummy_dataset()

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = None

            with self.assertRaisesRegex(
                ValueError,
                expected_regex=r"Padding is enabled, but the tokenizer is not configured with a padding token."
                r" Explicitly set `tokenizer.pad_token` \(e.g. `tokenizer.pad_token = tokenizer.eos_token`\)"
                r" before calling the trainer.",
            ):
                trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=dummy_dataset,
                    eval_dataset=dummy_dataset,
                )

                trainer.train()

    def test_tr_dpo_trainer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                precompute_ref_log_probs=False,
                sync_ref_model=True,
                ref_model_mixup_alpha=0.5,
                ref_model_sync_steps=1,
            )

            dummy_dataset = self._init_dummy_dataset()

            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.model,
                beta=0.1,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            # params of the ref model as its the same as the model
            previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

            trainer.train()

            assert trainer.state.log_history[-1]["train_loss"] is not None

            # check the params have changed
            for n, param in previous_trainable_params.items():
                new_param = trainer.ref_model.get_parameter(n)
                # check the ref model's params have changed - ignore 0 biases
                if param.sum() != 0:
                    assert not torch.equal(param, new_param)

    @require_no_wandb
    def test_dpo_trainer_generate_during_eval_no_wandb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=1,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                generate_during_eval=True,
            )

            dummy_dataset = self._init_dummy_dataset()

            with self.assertRaisesRegex(
                ValueError,
                expected_regex="`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve.",
            ):
                DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset,
                    eval_dataset=dummy_dataset,
                )

    @require_peft
    @mark.peft_test
    def test_dpo_lora_save(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model_peft = get_peft_model(model, lora_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                precompute_ref_log_probs=True,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model_peft,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

            # assert that the model is loaded without giving OSError
            try:
                AutoModelForCausalLM.from_pretrained(tmp_dir)
            except OSError:
                self.fail("Loading the saved peft adapter failed")

    @require_peft
    @require_bitsandbytes
    @mark.peft_test
    def test_dpo_lora_bf16_autocast_llama(self):
        # Note this test only works on compute capability > 7 GPU devices
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                bf16=True,
                beta=0.1,
                generate_during_eval=True,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

    @parameterized.expand(
        [
            ["gpt2", "sigmoid", False, False],
            ["gpt2", "sigmoid", False, True],
            ["gpt2", "sigmoid", True, False],
            ["gpt2", "sigmoid", True, True],
            ["gpt2", "ipo", False, False],
            ["gpt2", "ipo", False, True],
            ["gpt2", "ipo", True, False],
            ["gpt2", "ipo", True, True],
            ["gpt2", "aot_pair", False, False],
            ["gpt2", "aot_pair", False, True],
            ["gpt2", "aot_pair", True, False],
            ["gpt2", "aot_pair", True, True],
            ["gpt2", "aot", False, False],
            ["gpt2", "aot", False, True],
            ["gpt2", "aot", True, False],
            ["gpt2", "aot", True, True],
            ["gpt2", "bco_pair", False, False],
            ["gpt2", "bco_pair", False, True],
            ["gpt2", "bco_pair", True, False],
            ["gpt2", "bco_pair", True, True],
            ["gpt2", "robust", False, False],
            ["gpt2", "robust", False, True],
            ["gpt2", "robust", True, False],
            ["gpt2", "robust", True, True],
        ]
    )
    @require_bitsandbytes
    @require_peft
    @mark.peft_test
    @unittest.skip("You need a GPU with bf16 support in order to run these tests")
    def test_dpo_lora_bf16_autocast(self, name, loss_type, pre_compute, gen_during_eval):
        # Note this test only works on compute capability > 7 GPU devices
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(self.model_id, load_in_4bit=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                bf16=True,
                beta=0.1,
                generate_during_eval=gen_during_eval,
                loss_type=loss_type,
                precompute_ref_log_probs=pre_compute,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

            # save peft adapter
            trainer.save_model()

    @require_peft
    def test_dpo_lora_tags(self):
        from peft import LoraConfig

        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            assert trainer.model.model_tags == trainer._tag_names

    @require_peft
    def test_dpo_tags(self):
        model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            assert trainer.model.model_tags == trainer._tag_names

    @require_peft
    @mark.peft_test
    def test_dpo_lora_force_use_ref(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # lora model
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model_peft = get_peft_model(model, lora_config)

        ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
            )

            dummy_dataset = self._init_dummy_dataset()

            with self.assertRaises(ValueError):
                # passing a peft_model as model and ref_model should error out,
                # unless you pass `force_use_ref_model`
                trainer = DPOTrainer(
                    model=model_peft,
                    ref_model=ref_model,
                    args=training_args,
                    tokenizer=self.tokenizer,
                    train_dataset=dummy_dataset,
                    eval_dataset=dummy_dataset,
                    peft_config=lora_config,
                )

            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                beta=0.1,
                force_use_ref_model=True,
            )

            trainer = DPOTrainer(
                model=model_peft,
                ref_model=ref_model,
                args=training_args,
                tokenizer=self.tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
                peft_config=lora_config,
            )

            # train the model
            trainer.train()

    def test_dpo_trainer_torch_dtype(self):
        # See https://github.com/huggingface/trl/issues/1751
        dummy_dataset = self._init_dummy_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dpo_config = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                model_init_kwargs={"torch_dtype": "float16"},
                ref_model_init_kwargs={"torch_dtype": "float16"},
            )

            trainer = DPOTrainer(
                model=self.model_id,
                ref_model=self.model_id,
                tokenizer=self.tokenizer,
                args=dpo_config,
                train_dataset=dummy_dataset,
            )
            assert trainer.model.config.torch_dtype == torch.float16
            assert trainer.ref_model.config.torch_dtype == torch.float16

        # Now test when `torch_dtype` is provided but is wrong to either the model or the ref_model
        with tempfile.TemporaryDirectory() as tmp_dir:
            dpo_config = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                model_init_kwargs={"torch_dtype": -1},
            )

            with pytest.raises(
                ValueError,
                match="Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got -1.",
            ):
                _ = DPOTrainer(
                    model=self.model_id,
                    tokenizer=self.tokenizer,
                    args=dpo_config,
                    train_dataset=dummy_dataset,
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dpo_config = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=1,
                ref_model_init_kwargs={"torch_dtype": -1},
            )

            with pytest.raises(
                ValueError,
                match="Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got -1.",
            ):
                _ = DPOTrainer(
                    model=self.model_id,
                    ref_model=self.model_id,
                    tokenizer=self.tokenizer,
                    args=dpo_config,
                    train_dataset=dummy_dataset,
                )

    def test_dpo_loss_alpha_div_f(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                f_divergence_type=FDivergenceType.ALPHA_DIVERGENCE.value,
                f_alpha_divergence_coef=0.5,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            # Fake chosen and rejected log probs
            policy_chosen_logps = torch.FloatTensor([410.0, 0.1])
            policy_rejected_logps = torch.FloatTensor([810.5, 0.2])
            reference_chosen_logps = torch.FloatTensor([-610.0, -0.1])
            reference_rejected_logps = torch.FloatTensor([110.6, 0.5])
            losses, _, _ = trainer.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            assert torch.isfinite(losses).cpu().numpy().all()

    def test_dpo_loss_js_div_f(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # lora model
        model = AutoModelForCausalLM.from_pretrained(model_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = DPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                learning_rate=9e-1,
                eval_strategy="steps",
                f_divergence_type=FDivergenceType.JS_DIVERGENCE.value,
                f_alpha_divergence_coef=0.5,
            )

            dummy_dataset = self._init_dummy_dataset()

            # dpo train lora model with a lora config
            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=dummy_dataset,
                eval_dataset=dummy_dataset,
            )

            # Fake chosen and rejected log probs
            policy_chosen_logps = torch.FloatTensor([410.0, 0.1])
            policy_rejected_logps = torch.FloatTensor([95.5, 0.2])
            reference_chosen_logps = torch.FloatTensor([-610.0, -0.1])
            reference_rejected_logps = torch.FloatTensor([5.5, 0.5])
            losses, _, _ = trainer.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            assert torch.isfinite(losses).cpu().numpy().all()


if __name__ == "__main__":
    unittest.main()
