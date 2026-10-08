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

import math
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig, Trainer, TrainingArguments

from trl.experimental.async_distillation import AsyncDistillationTrainer
from trl.experimental.async_distillation.async_distillation_trainer import (
    DataCollatorForRollout as DistillationCollator,
)
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout as GRPOCollator
from trl.models.utils import _ForwardRedirection


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(2, 2)
        torch.nn.init.ones_(self.embed.weight)

    def forward(self, input_ids, position_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _TokenModel(torch.nn.Module):
    """A token-local model keeps the reference independent of how samples are packed."""

    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.base_model = _Backbone()
        self.lm_head = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.lm_head.weight)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, position_ids, labels=None, completion_mask=None, use_cache=False):
        hidden = self.base_model(input_ids, position_ids).last_hidden_state
        log_probs = self.lm_head(hidden[:, :-1]).log_softmax(-1)
        return {
            "log_probs": log_probs.gather(-1, input_ids[:, 1:, None]).squeeze(-1),
            "entropy": -(log_probs.exp() * log_probs).sum(-1),
            "aux_loss": self.lm_head.weight.sum() + 2.0,
        }


@pytest.fixture(params=[AsyncGRPOTrainer, AsyncDistillationTrainer], ids=["grpo", "distillation"])
def trainer(request, tmp_path):
    # Use the real Trainer and Accelerator paths on CPU. The async constructors require a rollout server and FA3.
    trainer = request.param.__new__(request.param)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        gradient_accumulation_steps=2,
        bf16=False,
        fp16=False,
    )
    Trainer.__init__(trainer, model=_TokenModel(), args=args, compute_loss_func="non-None value to disable scaling")
    trainer.model_accepts_loss_kwargs = False
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.aux_loss_enabled = False
    trainer.router_aux_loss_coef = 0.1
    trainer._metrics = {"train": defaultdict(list)}
    trainer._teacher_ids = ["teacher"]
    trainer._forward_redirection = _ForwardRedirection()
    trainer.args.beta = 0.0
    trainer.args.teacher_temperature = 1.0
    trainer.args.add_tail_bucket = False
    return trainer


def _sample(n_tokens, advantage, teacher_probability, sample_id=0):
    # Two prompt tokens and an internal context token must not enter the denominator.
    mask = [0, 0] + [1] * (n_tokens // 2) + [0] + [1] * (n_tokens - n_tokens // 2)
    return {
        "input_ids": [1] * len(mask),
        "completion_mask": mask,
        "old_log_probs": [-math.log(2)] * len(mask),
        "advantage": advantage,
        "teacher_topk_ids": [[0, 1]] * len(mask),
        "teacher_topk_logprobs": [[math.log(teacher_probability), math.log(1 - teacher_probability)]] * len(mask),
        "group_id": sample_id,
        "prompt_id": sample_id,
        "metrics": {},
    }


def _collate(trainer, groups):
    if isinstance(trainer, AsyncGRPOTrainer):
        collator = GRPOCollator(pad_token_id=0, num_processes=len(groups))
    else:
        collator = DistillationCollator(pad_token_id=0, teacher_top_k=1, num_processes=len(groups))
    return collator([groups])


def _step(trainer, batches, num_batches):
    trainer.model.zero_grad()
    trainer.args.gradient_accumulation_steps = num_batches
    trainer._step_forward_tokens = trainer._step_trained_tokens = 0
    trainer._step_seq_len_weighted = trainer._step_samples = trainer._step_forward_s = 0
    trainer._step_microbatches = trainer._current_train_step_time = 0
    batches, num_items = trainer.get_batch_samples(iter(batches), num_batches, trainer.args.device)
    # This is how Trainer handles the final, shorter accumulation window.
    trainer.current_gradient_accumulation_steps = len(batches)
    loss = sum(trainer.training_step(trainer.model, batch, num_items) for batch in batches)
    grads = {name: param.grad.clone() for name, param in trainer.model.named_parameters()}
    return loss, grads


@pytest.mark.parametrize("token_counts", [(100, 900), (100, 100), (0, 9), (0, 0)])
@pytest.mark.parametrize("num_batches", [2, 4])
def test_accumulated_gradients_match_full_batch(trainer, token_counts, num_batches):
    samples = [_sample(token_counts[0], -2.0, 0.9), _sample(token_counts[1], -0.5, 0.6, sample_id=1)]
    reference_loss, reference_grads = _step(trainer, [_collate(trainer, [samples])], 1)
    loss, grads = _step(trainer, [_collate(trainer, [[sample]]) for sample in samples], num_batches)

    torch.testing.assert_close(loss, reference_loss)
    for name in grads:
        torch.testing.assert_close(grads[name], reference_grads[name])
    assert trainer._step_trained_tokens == sum(token_counts)
    assert trainer._step_microbatches == 2


def test_window_token_count(trainer):
    # Each rank receives the same global count; these values must not be summed across ranks again.
    batches = [{"global_n_tokens": torch.tensor([n], dtype=torch.float32)} for n in [100, 900, 50]]
    iterator = iter(batches)
    window, count = trainer.get_batch_samples(iterator, 2, trainer.args.device)
    assert len(window) == 2
    torch.testing.assert_close(count, torch.tensor(1000.0))
    window, count = trainer.get_batch_samples(iterator, 2, trainer.args.device)
    assert len(window) == 1
    torch.testing.assert_close(count, torch.tensor(50.0))
    window, count = trainer.get_batch_samples(iterator, 2, trainer.args.device)
    assert window == []
    assert count is None


def test_aux_loss_keeps_its_scale(trainer):
    if not isinstance(trainer, AsyncGRPOTrainer):
        pytest.skip("Only GRPO has a router auxiliary loss.")
    trainer.aux_loss_enabled = True
    samples = [_sample(1, -2.0, 0.9), _sample(9, -0.5, 0.6, sample_id=1)]
    reference_loss, reference_grads = _step(trainer, [_collate(trainer, [samples])], 1)
    loss, grads = _step(trainer, [_collate(trainer, [[sample]]) for sample in samples], 2)
    torch.testing.assert_close(loss, reference_loss)
    for name in grads:
        torch.testing.assert_close(grads[name], reference_grads[name])


@pytest.mark.parametrize("token_counts", [((1, 3), (4, 12)), ((0, 0), (0, 1))])
def test_rank_average_matches_full_batch(trainer, monkeypatch, token_counts):
    # Model gradients are computed per rank, then averaged here to check the DDP loss scale on CPU.
    samples = [
        [_sample(count, -2.0 if i == 0 else -0.5, 0.9 if i == 0 else 0.6) for count in counts]
        for i, counts in enumerate(token_counts)
    ]
    all_samples = [sample for batch in samples for sample in batch]
    reference_loss, reference_grads = _step(trainer, [_collate(trainer, [all_samples])], 1)
    global_batches = [_collate(trainer, [[sample] for sample in batch]) for batch in samples]
    monkeypatch.setattr(type(trainer.accelerator), "num_processes", property(lambda self: 2))
    rank_results = []
    for rank in range(2):
        local_batches = [{key: value[rank : rank + 1] for key, value in batch.items()} for batch in global_batches]
        rank_results.append(_step(trainer, local_batches, 2))
    torch.testing.assert_close(sum(loss for loss, _ in rank_results) / 2, reference_loss)
    for name in reference_grads:
        torch.testing.assert_close(sum(grads[name] for _, grads in rank_results) / 2, reference_grads[name])


def test_trainer_optimizer_step_matches_full_batch(trainer, monkeypatch):
    samples = [_sample(1, -2.0, 0.9), _sample(9, -0.5, 0.6, sample_id=1)]
    reference_loss, reference_grads = _step(trainer, [_collate(trainer, [samples])], 1)
    expected = {name: param.detach() - 0.1 * reference_grads[name] for name, param in trainer.model.named_parameters()}
    trainer.model.zero_grad()
    batches = [_collate(trainer, [[sample]]) for sample in samples]
    trainer.args.gradient_accumulation_steps = 2
    trainer.args.max_steps = 1
    trainer.args.max_grad_norm = 0.0
    trainer.args.save_strategy = "no"
    trainer.args.logging_strategy = "no"
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)
    # Keep Trainer's full optimizer loop, but omit the async worker lifecycle and rollout metrics.
    monkeypatch.setattr(trainer, "_inner_training_loop", Trainer._inner_training_loop.__get__(trainer))
    monkeypatch.setattr(trainer, "log", Trainer.log.__get__(trainer))
    monkeypatch.setattr(trainer, "get_train_dataloader", lambda: torch.utils.data.DataLoader(batches, batch_size=None))
    result = trainer.train()
    assert result.global_step == 1
    assert result.training_loss == pytest.approx(reference_loss.item())
    for name, param in trainer.model.named_parameters():
        torch.testing.assert_close(param, expected[name])
