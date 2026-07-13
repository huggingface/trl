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

"""Numerical parity and dispatch tests for the custom Triton GRPO loss paths.

`use_liger_kernel=True` routes through the Liger fork's non-chunked `triton_grpo_loss` on
materialized logits; `use_liger_chunked_loss=True` additionally selects the fused-linear
`chunked_triton_grpo_loss`. Both must agree numerically with TRL's reference torch loss.
"""

from unittest.mock import patch

import pytest
import torch
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer

from .testing_utils import TrlTestCase, require_liger_kernel, require_torch_accelerator


def make_trainer(tmp_dir, loss_type="dapo", **config_kwargs):
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
    training_args = GRPOConfig(
        output_dir=tmp_dir,
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=16,
        loss_type=loss_type,
        report_to="none",
        logging_strategy="no",
        **config_kwargs,
    )
    return GRPOTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        args=training_args,
        train_dataset=dataset,
    )


def make_loss_inputs(trainer, device, seed=0):
    """Build a fixed synthetic batch shaped like `_generate_and_score_completions` output."""
    generator = torch.Generator().manual_seed(seed)
    batch, prompt_len, completion_len = 4, 6, 12
    vocab_size = trainer.model.config.vocab_size
    prompt_ids = torch.randint(0, vocab_size, (batch, prompt_len), generator=generator).to(device)
    completion_ids = torch.randint(0, vocab_size, (batch, completion_len), generator=generator).to(device)
    prompt_mask = torch.ones_like(prompt_ids)
    completion_mask = torch.ones_like(completion_ids)
    # Zero out a tail to exercise masking
    completion_mask[0, -3:] = 0
    completion_mask[2, -1:] = 0
    advantages = torch.tensor([0.75, -0.25, 1.5, -1.0], device=device)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "num_items_in_batch": completion_mask.sum(),
    }


def compute_loss_and_grad(trainer, inputs, probe_param="lm_head.weight"):
    trainer.model.zero_grad(set_to_none=True)
    loss = trainer.compute_loss(trainer.model, dict(inputs))
    loss.backward()
    grad = trainer.model.get_parameter(probe_param).grad.detach().clone()
    trainer.model.zero_grad(set_to_none=True)
    return loss.detach().clone(), grad


@require_liger_kernel
@require_torch_accelerator
class TestTritonGRPOLossParity(TrlTestCase):
    @pytest.mark.parametrize("loss_type", ["dapo", "grpo", "bnpo"])
    def test_triton_loss_matches_torch_loss(self, loss_type):
        device = torch.accelerator.current_accelerator()
        torch_trainer = make_trainer(self.tmp_dir, loss_type=loss_type, use_liger_kernel=False)
        torch_trainer.model.to(device)
        inputs = make_loss_inputs(torch_trainer, device)
        torch_loss, torch_grad = compute_loss_and_grad(torch_trainer, inputs)

        liger_trainer = make_trainer(self.tmp_dir, loss_type=loss_type, use_liger_kernel=True)
        liger_trainer.model.load_state_dict(torch_trainer.model.state_dict())
        liger_trainer.model.to(device)
        liger_loss, liger_grad = compute_loss_and_grad(liger_trainer, inputs)

        assert torch.isfinite(liger_loss).all()
        torch.testing.assert_close(liger_loss, torch_loss, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(liger_grad, torch_grad, atol=1e-3, rtol=1e-2)

    def test_chunked_loss_matches_non_chunked(self):
        device = torch.accelerator.current_accelerator()
        base_trainer = make_trainer(self.tmp_dir, use_liger_kernel=True)
        base_trainer.model.to(device)
        inputs = make_loss_inputs(base_trainer, device)
        base_loss, base_grad = compute_loss_and_grad(base_trainer, inputs)

        chunked_trainer = make_trainer(self.tmp_dir, use_liger_kernel=True, use_liger_chunked_loss=True)
        chunked_trainer.model.load_state_dict(base_trainer.model.state_dict())
        chunked_trainer.model.to(device)
        chunked_loss, chunked_grad = compute_loss_and_grad(chunked_trainer, inputs)

        assert torch.isfinite(chunked_loss).all()
        torch.testing.assert_close(chunked_loss, base_loss, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(chunked_grad, base_grad, atol=1e-3, rtol=1e-2)


@require_liger_kernel
class TestTritonGRPOLossDispatch(TrlTestCase):
    def test_default_routes_to_non_chunked_triton_loss(self):
        trainer = make_trainer(self.tmp_dir, use_liger_kernel=True)
        assert trainer.use_liger_kernel
        assert not trainer.use_liger_chunked_loss
        with patch.object(trainer, "compute_liger_loss", return_value=torch.tensor(0.0)) as mock_loss:
            trainer.compute_loss(trainer.model, {})
        mock_loss.assert_called_once()

    def test_chunked_flag_routes_to_chunked_loss(self):
        trainer = make_trainer(self.tmp_dir, use_liger_kernel=True, use_liger_chunked_loss=True)
        assert trainer.use_liger_chunked_loss
        with patch.object(trainer, "compute_chunked_liger_loss", return_value=torch.tensor(0.0)) as mock_loss:
            trainer.compute_loss(trainer.model, {})
        mock_loss.assert_called_once()

    def test_liger_disabled_ignores_chunked_flag(self):
        trainer = make_trainer(self.tmp_dir, use_liger_kernel=False, use_liger_chunked_loss=True)
        assert not trainer.use_liger_chunked_loss

    def test_vespo_with_chunked_loss_raises(self):
        with pytest.raises(ValueError, match="vespo"):
            make_trainer(self.tmp_dir, loss_type="vespo", use_liger_kernel=True, use_liger_chunked_loss=True)
