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

"""
Worker for `test_distillation_trainer_dtensor_head.py`: runs the multi-teacher loss with a sharded `DTensor` student
head on two gloo ranks that route to a *different number of teachers* (`[a, b]` on rank 0, `[a, a]` on rank 1).

This isolates the loss path rather than running a full FSDP2 trainer: FSDP2 is what makes the student head a
`DTensor`, and the head is the only student parameter the chunked loss projects through, so a head built by hand on
the mesh exercises exactly the collectives the trainer would issue. The student head's `full_tensor()` must therefore
run once per microbatch, not once per teacher group, or the ranks issue different numbers of collectives and
`TORCH_DISTRIBUTED_DEBUG=DETAIL` reports a mismatch (or the run deadlocks). Forward, the cross-rank statistic
reduction `compute_loss` performs, and backward are all driven.
"""

import argparse
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from trl.trainer.distillation_trainer import DistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--equal-groups", action="store_true", help="Route both ranks to [a, b] instead of [a, b]/[a, a]."
    )
    script_args = parser.parse_args()

    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    # Asserted here so a single-process fallback can never be read as a two-rank result.
    assert dist.get_world_size() == 2, dist.get_world_size()
    torch.manual_seed(42)

    vocab_size, hidden_size, rows, tokens = 8, 4, 2, 2
    mesh = init_device_mesh("cpu", (2,))
    # What FSDP2 hands the loss: the student `lm_head` weight sharded over the mesh's single dimension.
    weight = distribute_tensor(torch.randn(vocab_size, hidden_size), mesh, [Shard(0)]).requires_grad_()
    student_head = SimpleNamespace(weight=weight, bias=None)
    config = SimpleNamespace(get_text_config=lambda: SimpleNamespace())
    student = SimpleNamespace(config=config, get_output_embeddings=lambda: student_head)
    teachers = {
        teacher_id: SimpleNamespace(
            config=config, get_output_embeddings=lambda: torch.nn.Linear(hidden_size, vocab_size, bias=False)
        )
        for teacher_id in ["a", "b"]
    }

    # Unequal routing: rank 0 holds two teacher groups, rank 1 holds one. The head gather must not depend on this.
    teacher_index = torch.tensor([0, 1] if script_args.equal_groups or rank == 0 else [0, 0])
    targets = {
        int(index): torch.randn(int((teacher_index == index).sum()), tokens, hidden_size)
        for index in teacher_index.unique()
    }
    hidden_states = torch.randn(rows, tokens, hidden_size, requires_grad=True)
    trainer = SimpleNamespace(
        model=SimpleNamespace(training=True),
        teacher_models=teachers,
        _teacher_ids=["a", "b"],
        _teacher_head=None,
        _teacher_targets={"train": {0: targets}},
        _teacher_targets_index=0,
        beta=0.5,
        temperature=1.0,
        _get_last_hidden_state=lambda *args, **kwargs: hidden_states,
    )
    inputs = {
        "prompt_ids": torch.ones(rows, tokens, dtype=torch.long),
        "completion_ids": torch.ones(rows, tokens, dtype=torch.long),
        "prompt_mask": torch.ones(rows, tokens),
        "completion_mask": torch.ones(rows, tokens),
        "teacher_index": teacher_index,
    }

    try:
        loss, entropy_sum, n_valid, teacher_stats = DistillationTrainer._compute_loss(trainer, student, inputs, None)
        # The reduction `compute_loss` runs on the `[2, num_teachers]` statistics, right after the loss returns: the
        # collective the mismatched head gathers used to collide with.
        dist.all_reduce(teacher_stats)
        loss.backward()
        assert weight.grad is not None
        assert torch.isfinite(loss)
        assert n_valid.item() == rows * tokens
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
