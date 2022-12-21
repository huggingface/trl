import unittest
import torch
from transformers import AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead

from .utils import BaseModelTester

ALL_CAUSAL_LM_MODELS = [
    "trl-internal-testing/tiny-random-CodeGenForCausalLM",
    "trl-internal-testing/tiny-random-GPTJForCausalLM",
    "trl-internal-testing/tiny-random-GPTNeoForCausalLM",
    "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "trl-internal-testing/tiny-random-OPTForCausalLM",
    "trl-internal-testing/tiny-random-BloomForCausalLM",
    "trl-internal-testing/tiny-random-GPT2LMHeadModel",
]

class VHeadModelTester(BaseModelTester, unittest.TestCase):
    r"""
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