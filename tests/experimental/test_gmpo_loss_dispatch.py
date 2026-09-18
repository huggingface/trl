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

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from trl.experimental.gmpo import GMPOTrainer


@pytest.mark.parametrize("use_liger_kernel", [False, True])
def test_compute_loss_preserves_gmpo_objective_and_gradient(use_liger_kernel):
    trainer = object.__new__(GMPOTrainer)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.use_liger_kernel = use_liger_kernel
    trainer.model = torch.nn.Linear(1, 1)
    trainer.top_entropy_quantile = 1.0
    trainer.epsilon_low = trainer.epsilon_high = 0.4
    trainer.beta = 0.0
    trainer.current_gradient_accumulation_steps = 1
    trainer._metrics = {"train": defaultdict(list)}
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        reduce=lambda value, reduction: value,
        gather=lambda value: value,
        unwrap_model=lambda model: model,
    )

    def unexpected_fused_loss(*args, **kwargs):
        raise AssertionError("GMPO must not dispatch to the fused GRPO objective")

    trainer._forward_redirection = unexpected_fused_loss
    trainer.compute_liger_loss = unexpected_fused_loss
    logp = torch.tensor([[0.1, 0.6, -0.2], [-0.1, -0.6, 0.2]], dtype=torch.float64, requires_grad=True)
    trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (logp, torch.zeros_like(logp), None)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    advantages = torch.tensor([1.0, -0.5], dtype=torch.float64)
    inputs = {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long),
        "prompt_mask": torch.ones(2, 1, dtype=torch.long),
        "completion_ids": torch.ones(2, 3, dtype=torch.long),
        "completion_mask": mask,
        "old_per_token_logps": torch.zeros_like(logp),
        "advantages": advantages,
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    gradient = torch.autograd.grad(loss, logp)[0]

    reference_logp = logp.detach().clone().requires_grad_()
    positive = torch.minimum(reference_logp[0], reference_logp.new_tensor(0.4))
    negative = torch.maximum(reference_logp[1], reference_logp.new_tensor(-0.4))
    expected = (-positive.mean().exp() + 0.5 * negative[:2].mean().exp()) / 2
    expected_gradient = torch.autograd.grad(expected, reference_logp)[0]
    torch.testing.assert_close(loss, expected, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-7, atol=1e-9)


def test_compute_loss_rejects_return_outputs():
    trainer = object.__new__(GMPOTrainer)
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(report_to=[])
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    with pytest.raises(ValueError, match="GMPOTrainer does not support returning outputs"):
        trainer.compute_loss(None, {}, return_outputs=True)
