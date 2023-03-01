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
import unittest

from pytest import mark
from transformers import AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, is_peft_available


if is_peft_available():
    from peft import get_peft_model, LoraConfig

from .testing_utils import require_peft


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
