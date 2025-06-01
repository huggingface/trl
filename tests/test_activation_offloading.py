# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.testing_utils import require_peft, require_torch_accelerator, torch_device
from transformers.utils import is_peft_available

from trl.models.activation_offloading import NoOpManager, OffloadActivations


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestActivationOffloading(unittest.TestCase):
    @require_torch_accelerator
    @require_peft
    def test_offloading_with_peft_models(self) -> None:
        """Test that activation offloading works with PEFT models."""
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        inp = torch.randint(0, 100, (2, 10), device=torch_device)

        # First forward-backward pass without offloading
        torch.manual_seed(42)
        loss = model(inp, labels=inp).loss
        loss.backward()

        # Store gradients - only from trainable parameters
        grads_original = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads_original.append((name, param.grad.clone()))

        # Reset gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        # Second forward-backward pass with offloading
        torch.manual_seed(42)
        with OffloadActivations():
            loss_c = model(inp, labels=inp).loss
        loss_c.backward()

        # Compare gradients - only trainable parameters
        for name_orig, grad_orig in grads_original:
            for name_param, param in model.named_parameters():
                if name_param == name_orig and param.requires_grad and param.grad is not None:
                    self.assertTrue(
                        torch.allclose(grad_orig, param.grad, rtol=1e-4, atol=1e-5),
                        f"Gradient mismatch for {name_orig}",
                    )

    @require_torch_accelerator
    def test_noop_manager_with_offloading(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
        inp = torch.randint(0, 100, (2, 10), device=torch_device)

        # Run with offloading but disable for specific section
        with OffloadActivations():
            # First forward-backward with normal offloading
            torch.manual_seed(42)
            out1 = model(inp, labels=inp)
            out1.loss.backward()
            grads1 = [p.grad.clone() for p in model.parameters()]

            # Reset grads
            for p in model.parameters():
                p.grad = None

            # Second forward-backward with NoOpManager
            with NoOpManager():
                torch.manual_seed(42)
                out2 = model(inp, labels=inp)
                out2.loss.backward()

            grads2 = [p.grad.clone() for p in model.parameters()]

        # Gradients should match as NoOpManager should have prevented offloading
        for g1, g2 in zip(grads1, grads2):
            self.assertTrue(torch.allclose(g1, g2, rtol=1e-4, atol=1e-5))

    @require_torch_accelerator
    def test_min_offload_size(self):
        """Test that tensors smaller than min_offload_size aren't offloaded"""
        model = nn.Sequential(
            nn.Linear(5, 5),  # Small layer that shouldn't be offloaded
            nn.Linear(5, 1000),  # Large layer that should be offloaded
        ).to(torch_device)

        inp = torch.randn(2, 5, device=torch_device)

        with OffloadActivations(min_offload_size=1000):
            out = model(inp)
            out.sum().backward()

        # The test passes if no errors occur, as we're mainly testing
        # that the logic handles both offloaded and non-offloaded tensors

    @require_torch_accelerator
    def test_real_hf_model(self):
        """Test with an actual HuggingFace model"""
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

        # Create small input
        inp = torch.randint(0, 100, (2, 10), device=torch_device)

        # Baseline without offloading
        torch.manual_seed(42)
        out1 = model(inp, labels=inp).loss
        out1.backward()
        grads1 = [p.grad.clone() for p in model.parameters()]

        # Reset grads
        for p in model.parameters():
            p.grad = None

        # With offloading
        with OffloadActivations():
            torch.manual_seed(42)
            out2 = model(inp, labels=inp).loss
            out2.backward()

        grads2 = [p.grad.clone() for p in model.parameters()]

        # Check outputs and gradients match
        self.assertTrue(torch.allclose(out1, out2, rtol=1e-5))
        for g1, g2 in zip(grads1, grads2):
            self.assertTrue(torch.allclose(g1, g2, rtol=1e-5))
