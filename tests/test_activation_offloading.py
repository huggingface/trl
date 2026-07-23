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

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device
from transformers.utils import is_peft_available

from trl.models import activation_offloading as activation_offloading_module
from trl.models.activation_offloading import NoOpManager, OffloadActivations

from .testing_utils import TrlTestCase, require_peft, require_torch_accelerator


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestActivationOffloading(TrlTestCase):
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
                    (
                        torch.testing.assert_close(grad_orig, param.grad, rtol=1e-4, atol=1e-5),
                        (f"Gradient mismatch for {name_orig}"),
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
        for g1, g2 in zip(grads1, grads2, strict=True):
            torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-5)

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
        torch.testing.assert_close(out1, out2)
        for g1, g2 in zip(grads1, grads2, strict=True):
            torch.testing.assert_close(g1, g2)

    @require_torch_accelerator
    def test_tensor_deduplication(self):
        """Test that deduplication works correctly for tensors sharing storage"""

        class ModelWithViews(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 100)

            def forward(self, x):
                out = self.linear(x)
                view1 = out.view(-1)
                view2 = out.transpose(0, 1)
                return view1.sum() + view2.sum()

        model = ModelWithViews().to(torch_device)
        offload_ctx = OffloadActivations(min_offload_size=1)
        offload_ctx.update_model_params(model)

        x = torch.randn(10, 100, device=torch_device, requires_grad=True)
        with offload_ctx:
            loss = model(x)

        total_tensor_ids = offload_ctx.tensor_id
        assert total_tensor_ids > 0, "Should have created tensor IDs"

        # modified=True means offloaded to CPU, modified=False means kept on GPU (deduplicated)
        deduplicated_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if not modified)
        offloaded_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if modified)

        assert offloaded_count > 0, "Should have offloaded at least one tensor"
        assert deduplicated_count > 0, "Should have deduplicated at least one tensor (view)"

        unique_storages_offloaded = len(offload_ctx.storage_to_tensor_id)
        assert unique_storages_offloaded < total_tensor_ids, (
            f"Deduplication should result in fewer storages ({unique_storages_offloaded}) "
            f"than total tensors ({total_tensor_ids})"
        )

        loss.backward()

    @require_torch_accelerator
    def test_reused_storage_key_after_stash_reap_is_not_deduplicated(self):
        """A reused allocator address should not be treated as a live view."""

        class SaveTensor(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                ctx.save_for_backward(tensor)
                return tensor.sum()

            @staticmethod
            def backward(ctx, grad_output):
                (tensor,) = ctx.saved_tensors
                return torch.ones_like(tensor) * grad_output

        first = torch.randn(4, 4, device=torch_device, requires_grad=True)
        filler = torch.randn(4, 4, device=torch_device, requires_grad=True)
        reused = torch.randn(4, 4, device=torch_device, requires_grad=True)

        def fake_storage_key(tensor):
            if id(tensor) in {id(first), id(reused)}:
                return ("reused-storage-key", tensor.dtype)
            return (id(tensor), tensor.dtype)

        offload_ctx = OffloadActivations(
            use_pin_memory=False,
            use_streams=True,
            min_offload_size=1,
            max_fwd_stash_size=1,
        )

        original_get_unique_tensor_key = activation_offloading_module._get_unique_tensor_key
        activation_offloading_module._get_unique_tensor_key = fake_storage_key
        try:
            with offload_ctx:
                loss = SaveTensor.apply(first) + SaveTensor.apply(filler) + SaveTensor.apply(reused)
        finally:
            activation_offloading_module._get_unique_tensor_key = original_get_unique_tensor_key

        offloaded_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if modified)
        deduplicated_count = sum(1 for _, modified, _, _, _ in offload_ctx.tracker.values() if not modified)

        assert offloaded_count == 3
        assert deduplicated_count == 0

        loss.backward()

    @require_torch_accelerator
    def test_stale_tracker_state_is_cleared_between_forwards(self):
        """Test that tensors from unused graph branches don't accumulate across steps."""

        class ModelWithUnusedBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.used = nn.Linear(8, 8)
                self.unused = nn.Linear(8, 8)

            def forward(self, x):
                return self.used(x).sum(), self.unused(x).sum()

        model = ModelWithUnusedBranch().to(torch_device)
        offload_ctx = OffloadActivations(use_pin_memory=False, use_streams=False, min_offload_size=1)
        offload_ctx.update_model_params(model)
        inp = torch.randn(4, 8, device=torch_device)

        tracker_counts = []
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            with offload_ctx:
                loss, _ = model(inp)
            loss.backward()
            tracker_counts.append(len(offload_ctx.tracker))

        assert tracker_counts == [tracker_counts[0]] * len(tracker_counts)

    @require_torch_accelerator
    def test_parameter_filtering(self):
        """Test that model parameters are filtered during offloading"""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10)).to(torch_device)
        offload_ctx = OffloadActivations()
        offload_ctx.update_model_params(model)

        assert len(offload_ctx.param_storages) > 0, "Should have tracked parameter storages"

        param_ptrs = {p.data.untyped_storage().data_ptr() for p in model.parameters()}
        assert offload_ctx.param_storages == param_ptrs, "Tracked storages should match parameter storages"


class _SdpaCounter(TorchDispatchMode):
    """Counts how many times an attention (scaled-dot-product) op executes at the dispatcher level."""

    def __init__(self) -> None:
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "scaled_dot_product" in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))


class TestSelectiveActivationCheckpointing(TrlTestCase):
    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def _forward_backward(self, model):
        torch.manual_seed(42)
        inp = torch.randint(0, model.config.vocab_size, (2, 32), device=torch_device)
        counter = _SdpaCounter()
        with counter:
            model(input_ids=inp, labels=inp).loss.backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        model.zero_grad(set_to_none=True)
        return grads, counter.count

    def test_matches_full_checkpointing_and_skips_attention_recompute(self):
        """SAC must produce identical gradients to full checkpointing while not recomputing attention."""
        # Full (non-selective) gradient checkpointing baseline.
        model_full = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        model_full.train()
        model_full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        grads_full, sdpa_full = self._forward_backward(model_full)

        # Selective activation checkpointing: our wrapper injects the SAC context function.
        model_sac = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        model_sac.train()
        activation_offloading_module.enable_selective_activation_checkpointing(model_sac)
        model_sac.gradient_checkpointing_enable()
        grads_sac, sdpa_sac = self._forward_backward(model_sac)

        # Gradients must match the full-checkpointing baseline.
        torch.testing.assert_close(grads_sac, grads_full, rtol=1e-4, atol=1e-5)

        # Full checkpointing recomputes attention in the backward pass; SAC saves it, so it runs fewer times.
        assert sdpa_sac < sdpa_full
