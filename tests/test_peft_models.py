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
import os
import tempfile
import unittest

import torch
from pytest import mark
from transformers import AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, is_peft_available


if is_peft_available():
    from peft import get_peft_model, LoraConfig

from .testing_utils import require_bitsandbytes, require_peft


@require_peft
@mark.peft_test
class PeftModelTester(unittest.TestCase):
    def setUp(self):
        self.causal_lm_model_id = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def test_create_peft_model(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        _ = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

    def test_peft_requires_grad(self):
        r"""
        Check that the value head of the returned model has requires_grad=True.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        # Check that the value head has requires_grad=True
        self.assertTrue(model.v_head.summary.weight.requires_grad)

    def test_check_peft_model_nb_trainable_params(self):
        r"""
        Check that the number of trainable parameters is correct.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 10273)

        # Check that the number of trainable param for the non-peft model is correct
        non_peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)
        nb_trainable_params = sum(p.numel() for p in non_peft_model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 99578)

    def test_create_peft_model_from_config(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.causal_lm_model_id, peft_config=self.lora_config
        )
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 10273)

        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 10273)

    @require_bitsandbytes
    def test_create_bnb_peft_model_from_config(self):
        r"""
        Simply creates a peft model and checks that it can be loaded.
        """
        from bitsandbytes.nn import Linear8bitLt

        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.causal_lm_model_id, peft_config=self.lora_config, load_in_8bit=True
        )
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 10273)
        self.assertTrue(
            trl_model.pretrained_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h.__class__ == Linear8bitLt
        )

        causal_lm_model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id, load_in_8bit=True, device_map="auto"
        )
        trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(causal_lm_model, peft_config=self.lora_config)
        # Check that the number of trainable parameters is correct
        nb_trainable_params = sum(p.numel() for p in trl_model.parameters() if p.requires_grad)
        self.assertEqual(nb_trainable_params, 10273)
        self.assertTrue(
            trl_model.pretrained_model.model.gpt_neox.layers[0].mlp.dense_h_to_4h.__class__ == Linear8bitLt
        )

    def test_save_pretrained_peft(self):
        r"""
        Check that the model can be saved and loaded properly.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/adapter_model.bin"),
                msg=f"{tmp_dir}/adapter_model.bin does not exist",
            )
            self.assertTrue(
                os.path.exists(f"{tmp_dir}/adapter_config.json"),
                msg=f"{tmp_dir}/adapter_config.json does not exist",
            )
            # check also for `pytorch_model.bin` and make sure it only contains `v_head` weights
            self.assertTrue(
                os.path.exists(f"{tmp_dir}/pytorch_model.bin"),
                msg=f"{tmp_dir}/pytorch_model.bin does not exist",
            )
            maybe_v_head = torch.load(f"{tmp_dir}/pytorch_model.bin")
            # check that only keys that starts with `v_head` are in the dict
            self.assertTrue(
                all(k.startswith("v_head") for k in maybe_v_head.keys()),
                msg=f"keys in {tmp_dir}/pytorch_model.bin do not start with `v_head`",
            )

            model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir)

            # check all the weights are the same
            for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters()):
                self.assertTrue(torch.allclose(p1[1], p2[1]), msg=f"{p1[0]} != {p2[0]}")

    def test_load_pretrained_peft(self):
        r"""
        Check that the model saved with peft class interface can be loaded properly.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pretrained_model.save_pretrained(tmp_dir)
            model_from_pretrained = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir)

            # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
            self.assertTrue(
                os.path.isfile(f"{tmp_dir}/adapter_model.bin"),
                msg=f"{tmp_dir}/adapter_model.bin does not exist",
            )
            self.assertTrue(
                os.path.exists(f"{tmp_dir}/adapter_config.json"),
                msg=f"{tmp_dir}/adapter_config.json does not exist",
            )

            # check all the weights are the same
            for p1, p2 in zip(model.named_parameters(), model_from_pretrained.named_parameters()):
                if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                    self.assertTrue(torch.allclose(p1[1], p2[1]), msg=f"{p1[0]} != {p2[0]}")

    def test_continue_training_peft_model(self):
        r"""
        Load peft and checks that it can continue training.
        """
        causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        pretrained_model = get_peft_model(causal_lm_model, self.lora_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pretrained_model.save_pretrained(tmp_dir)
            # set is_trainable to True
            model = AutoModelForCausalLMWithValueHead.from_pretrained(tmp_dir, is_trainable=True)
            # Check that the number of trainable parameters is correct
            nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(nb_trainable_params, 10273)
