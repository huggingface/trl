# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import tempfile
import torch

from transformers import AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead


ALL_CAUSAL_LM_MODELS = [
    "trl-internal-testing/tiny-random-CodeGenForCausalLM",
    "trl-internal-testing/tiny-random-GPTJForCausalLM",
    "trl-internal-testing/tiny-random-GPTNeoForCausalLM",
    "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "trl-internal-testing/tiny-random-OPTForCausalLM",
    "trl-internal-testing/tiny-random-BloomForCausalLM",
    "trl-internal-testing/tiny-random-GPT2LMHeadModel",
]

class BaseModelTester:
    all_model_names = None
    trl_model_class = None

    def test_from_save(self):
        """
        Test if the model can be saved and loaded from a directory and get the same weights
        """
        for model_name in self.all_model_names:
            torch.manual_seed(0)
            model = self.trl_model_class.from_pretrained(model_name)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                
                torch.manual_seed(0)
                model_from_save = self.trl_model_class.from_pretrained(tmp_dir)
            
            # Check if the weights are the same 
            for key in model_from_save.state_dict():
                self.assertTrue(torch.allclose(model_from_save.state_dict()[key], model.state_dict()[key]))
        
    def test_from_save_transformers(self):
        """
        Test if the model can be saved and loaded using transformers and get the same weights
        """
        for model_name in self.all_model_names:
            transformers_model = self.trl_model_class.transformers_parent_class.from_pretrained(model_name)

            trl_model = self.trl_model_class.from_pretrained(model_name)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                trl_model.save_pretrained(tmp_dir)
                transformers_model_from_save = self.trl_model_class.transformers_parent_class.from_pretrained(tmp_dir)
            
            # Check if the weights are the same 
            for key in transformers_model.state_dict():
                self.assertTrue(torch.allclose(transformers_model_from_save.state_dict()[key], transformers_model.state_dict()[key]))

class VHeadModelTester(BaseModelTester, unittest.TestCase):
    """
    Testing suite for v-head models.   
    """
    all_model_names = ALL_CAUSAL_LM_MODELS
    trl_model_class = AutoModelForCausalLMWithValueHead

    def test_vhead(self):
        r"""
        Test if the v-head is added to the model succesfully
        """
        for model_name in self.all_model_names:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
            self.assertTrue(hasattr(model, "v_head"))
    
    def test_vhead_not_str(self):
        r"""
        Test if the v-head is added to the model succesfully
        """
        for model_name in self.all_model_names:
            pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)
            self.assertTrue(hasattr(model, "v_head"))

    def test_inference(self):
        r"""
        Test if the model can be used for inference and outputs 3 values 
        - logits, loss, and value states   
        """
        EXPECTED_OUTPUT_SIZE = 3

        for model_name in self.all_model_names:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            outputs = model(input_ids)

            # Check if the outputs are of the right size - here 
            # we always output 3 values - logits, loss, and value states
            self.assertEqual(len(outputs), EXPECTED_OUTPUT_SIZE)
    
    def test_dropout_config(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the vhead
        """
        for model_name in self.all_model_names:
            pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
            pretrained_model.config.summary_dropout_prob = 0.5
            model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, pretrained_model.config.summary_dropout_prob)
    
    def test_dropout_kwargs(self):
        r"""
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the vhead
        """
        for model_name in self.all_model_names:
            v_head_kwargs = {"summary_dropout_prob": 0.5}

            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **v_head_kwargs)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, 0.5)

            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, summary_dropout_prob=0.5)

            # Check if v head of the model has the same dropout as the config
            self.assertEqual(model.v_head.dropout.p, 0.5)
    
    def test_generate(self):
        r"""
        Test if `generate` works for every model 
        """
        for model_name in self.all_model_names:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
            input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

            # Just check if the generation works
            _ = model.generate(input_ids)

    def test_raise_error_not_causallm(self):
        # Test with a model without a LM head
        model_id = "hf-internal-testing/tiny-random-GPT2Model"
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            pretrained_model = AutoModelForCausalLM.from_pretrained(model_id)
            _ = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model.transformer)