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

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from trl import TargetPOTrainer


WORLD_SIZE = 2
NUM_GENERATIONS = 4  # one group of 4 split across 2 ranks (2 per rank)
LOCAL_SEQ_LOGPS = [
    torch.tensor([0.1, -0.3]),  # rank 0
    torch.tensor([0.5, -0.2]),  # rank 1
]
LOCAL_TARGETS = [
    torch.tensor([0.1, 0.2]),  # rank 0
    torch.tensor([0.4, 0.3]),  # rank 1; global sums to 1.0
]


def _tpo_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        local = LOCAL_SEQ_LOGPS[rank].clone().requires_grad_(True)

        gathered = TargetPOTrainer._gather_tensor_with_grad(local)
        logps = torch.log_softmax(gathered.view(-1, NUM_GENERATIONS), dim=1).view(-1)

        process_slice = slice(rank * local.size(0), (rank + 1) * local.size(0))
        local_logps = logps[process_slice]
        local_targets = LOCAL_TARGETS[rank]

        loss = -(local_targets * local_logps).sum() * NUM_GENERATIONS / local_targets.numel()
        loss.backward()

        global_logps = torch.cat(LOCAL_SEQ_LOGPS)
        global_targets = torch.cat(LOCAL_TARGETS)
        global_softmax = torch.softmax(global_logps, dim=0)
        scale = NUM_GENERATIONS / local_targets.numel()
        expected = scale * (global_softmax[process_slice] - global_targets[process_slice])

        torch.testing.assert_close(local.grad, expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed not available")
def test_tpo_gradient_across_ranks_with_group_spanning_ranks():
    """
    A TPO prompt group of size 4 split 2/2 across DP ranks. The group's log-softmax normalizer depends on
    all four completions, so the autograd-aware all_gather must route gradient from each rank's loss back
    to the owning rank's local tensor. Expected local gradient is scale * (softmax - target) at local positions.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        init_file = os.path.join(tmp_dir, "rendezvous")
        mp.spawn(_tpo_worker, args=(WORLD_SIZE, init_file), nprocs=WORLD_SIZE, join=True)
