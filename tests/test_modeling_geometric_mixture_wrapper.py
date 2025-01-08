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

import unittest

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from trl.models.modeling_base import GeometricMixtureWrapper, create_reference_model


class TestGeometricMixtureWrapper(unittest.TestCase):
    def setUp(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.ref_model = create_reference_model(self.model)
        self.generation_config = GenerationConfig.from_pretrained(model_id)
        self.mixture_coef = 0.5
        self.wrapper = GeometricMixtureWrapper(
            self.model, self.ref_model, self.generation_config, mixture_coef=self.mixture_coef
        )

    def test_forward(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        output = self.wrapper(input_ids=input_ids, attention_mask=attention_mask)

        self.assertIsNotNone(output)
        self.assertTrue(hasattr(output, "logits"))
        self.assertEqual(output.logits.shape, (1, 5, self.model.config.vocab_size))

    def test_mixture_coefficient(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            ref_model_output = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
            wrapper_output = self.wrapper(input_ids=input_ids, attention_mask=attention_mask)

        expected_logits = torch.nn.functional.log_softmax(
            self.mixture_coef * ref_model_output.logits + (1 - self.mixture_coef) * model_output.logits, dim=-1
        )

        self.assertTrue(torch.allclose(wrapper_output.logits, expected_logits, atol=1e-5))

    def test_prepare_inputs_for_generation(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones_like(input_ids)

        inputs = self.wrapper.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask, use_cache=True)

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertFalse(inputs.get("use_cache", False))
