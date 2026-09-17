# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from trl.experimental.bco.bco_trainer import BCOTrainer, RunningMoments
from trl.trainer.base_trainer import _BaseTrainer


def test_accumulation_uses_one_updated_baseline(monkeypatch):
    monkeypatch.setattr(
        _BaseTrainer, "get_batch_samples", lambda self, iterator, num_batches, device: (list(iterator), None)
    )
    torch.manual_seed(13)
    chosen = torch.randn(4, 3, dtype=torch.double)
    rejected = torch.randn(4, 3, dtype=torch.double)
    weight = torch.randn(1, 3, dtype=torch.double)
    gradients = []
    means = []

    for sizes in [(4,), (1, 1, 1, 1), (1, 3)]:
        trainer = object.__new__(BCOTrainer)
        trainer.args = SimpleNamespace(gradient_accumulation_steps=1 if len(sizes) == 1 else 4)
        trainer.model_wrapped = torch.nn.Linear(3, 1, bias=False, dtype=torch.double)
        trainer.model_wrapped.weight.data.copy_(weight)
        trainer.model_wrapped.register_buffer("counter", torch.tensor(0))
        trainer.ref_model = None
        trainer.beta = 0.1
        trainer.match_underlying_distribution = False
        trainer.running = RunningMoments(
            accelerator=SimpleNamespace(use_distributed=False), count=16, mean=0.2, var=0.25
        )
        trainer._window_loss_calls = 0
        trainer._collecting_window_rewards = False
        trainer._window_rewards = []
        trainer._prepare_inputs = lambda batch: batch
        trainer.compute_loss_context_manager = nullcontext

        def get_loss(model, batch, do_train=False, current_trainer=trainer):
            model.counter.add_(1)
            torch.rand(1)
            random.random()
            np.random.rand()
            ch = model(batch[0]).flatten()
            rej = model(batch[1]).flatten()
            return current_trainer.bco_loss(ch, rej, torch.zeros_like(ch), torch.zeros_like(rej), None, None, do_train)

        trainer.get_batch_loss_metrics = get_loss
        batches = list(zip(chosen.split(sizes), rejected.split(sizes), strict=True))
        torch_rng = torch.get_rng_state()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        samples, _ = trainer.get_batch_samples(iter(batches), len(batches), "cpu")

        assert trainer.model_wrapped.counter.item() == 0
        assert torch.equal(torch_rng, torch.get_rng_state())
        assert python_rng == random.getstate()
        np.testing.assert_equal(numpy_rng, np.random.get_state())

        for batch in samples:
            losses, *_ = get_loss(trainer.model_wrapped, batch, do_train=True)
            (losses.sum() / 8).backward()
        assert trainer._window_loss_calls == 0
        assert trainer.running.count == 24
        means.append(trainer.running.mean)
        gradients.append(trainer.model_wrapped.weight.grad)

        before = (trainer.running.count, trainer.running.mean)
        get_loss(trainer.model_wrapped, samples[0], do_train=False)
        assert before == (trainer.running.count, trainer.running.mean)

    for grad in gradients[1:]:
        torch.testing.assert_close(grad, gradients[0], atol=1e-8, rtol=1e-6)
    assert max(means) - min(means) < 1e-7
