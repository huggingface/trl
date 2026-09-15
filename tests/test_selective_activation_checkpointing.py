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
from torch.utils.checkpoint import CheckpointPolicy
from transformers import AutoModelForCausalLM
from transformers.testing_utils import torch_device

from trl.models.selective_activation_checkpointing import (
    _aten_attention_ops,
    _build_policy_fn,
    enable_selective_activation_checkpointing,
)

from .testing_utils import TrlTestCase, require_torch_accelerator


class TestSelectiveActivationCheckpointingPolicy(TrlTestCase):
    """Exercises the SAC policy directly, against real and fake ops, so it needs no accelerator or model."""

    def test_saves_aten_sdpa_ops(self):
        aten_attention_ops = _aten_attention_ops()
        assert aten_attention_ops, "At least one SDPA backend should be registered under torch.ops.aten"
        policy_fn = _build_policy_fn(aten_attention_ops)
        for op in aten_attention_ops:
            assert policy_fn(None, op) == CheckpointPolicy.MUST_SAVE

    def test_saves_flash_attn_ops_by_name(self):
        # flash-attn's custom kernels live under hashed namespaces (e.g. `_flash_attn2_cuda_f12afc9::varlen_fwd`),
        # so they can only be recognized by substring, not by identity like the aten ops above.
        policy_fn = _build_policy_fn(set())
        assert policy_fn(None, "flash_attn2_cuda::varlen_fwd") == CheckpointPolicy.MUST_SAVE

    def test_recomputes_everything_else(self):
        policy_fn = _build_policy_fn(_aten_attention_ops())
        assert policy_fn(None, torch.ops.aten.mm.default) == CheckpointPolicy.PREFER_RECOMPUTE


@require_torch_accelerator
class TestSelectiveActivationCheckpointing(TrlTestCase):
    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def test_matches_full_checkpointing(self):
        """SAC must produce the same gradients as full checkpointing."""
        model_full = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        model_full.train()
        model_full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        model_sac = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        model_sac.train()
        enable_selective_activation_checkpointing(model_sac)
        model_sac.gradient_checkpointing_enable()

        torch.manual_seed(42)
        inp = torch.randint(0, model_full.config.vocab_size, (2, 32), device=torch_device)
        model_full(input_ids=inp, labels=inp).loss.backward()
        model_sac(input_ids=inp, labels=inp).loss.backward()

        for p_full, p_sac in zip(model_full.parameters(), model_sac.parameters(), strict=True):
            torch.testing.assert_close(p_sac.grad, p_full.grad, rtol=1e-4, atol=1e-5)

    def test_idempotent(self):
        """Enabling SAC twice on the same model must not stack wrappers."""
        model = AutoModelForCausalLM.from_pretrained(self.model_id, attn_implementation="sdpa").to(torch_device)
        enable_selective_activation_checkpointing(model)
        once_wrapped = model.gradient_checkpointing_enable
        enable_selective_activation_checkpointing(model)
        assert model.gradient_checkpointing_enable is once_wrapped
