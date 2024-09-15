import unittest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig
from trl.models.modeling_base import GeometricMixtureWrapper, create_reference_model

class TestGeometricMixtureWrapper(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.ref_model = create_reference_model(self.model)
        self.generation_config = GenerationConfig.from_pretrained("gpt2")
        self.mixture_coeff = 0.5
        self.wrapper = GeometricMixtureWrapper(
            self.model,
            self.ref_model,
            self.generation_config,
            mixture_coeff=self.mixture_coeff
        )

    def test_forward(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        output = self.wrapper(input_ids=input_ids, attention_mask=attention_mask)

        self.assertIsNotNone(output)
        self.assertTrue(hasattr(output, 'logits'))
        self.assertEqual(output.logits.shape, (1, 5, self.model.config.vocab_size))

    def test_mixture_coefficient(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            ref_model_output = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            wrapper_output = self.wrapper(input_ids=input_ids, attention_mask=attention_mask)

        expected_logits = torch.nn.functional.log_softmax(
            self.mixture_coeff * ref_model_output.logits + (1 - self.mixture_coeff) * model_output.logits,
            dim=-1
        )

        self.assertTrue(torch.allclose(wrapper_output.logits, expected_logits, atol=1e-5))

    def test_prepare_inputs_for_generation(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        inputs = self.wrapper.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )

        self.assertIn('input_ids', inputs)
        self.assertIn('attention_mask', inputs)
        self.assertFalse(inputs.get('use_cache', False))