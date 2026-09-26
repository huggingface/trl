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
import sys

import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch import nn
from transformers.integrations.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled

from trl.trainer.distillation_trainer import _chunked_divergence_loss
from trl.trainer.sft_trainer import _chunked_cross_entropy_loss


class ChunkedLossHeads(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.student = nn.Linear(8, 32768, bias=False)
        if mode == "distillation":
            self.teacher = nn.Linear(8, 32768, bias=False)


def main():
    mode = sys.argv[1]
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    config = {"train_batch_size": 2, "zero_optimization": {"stage": 3}}
    zero3_config = HfDeepSpeedConfig(config)
    assert is_deepspeed_zero3_enabled()
    with deepspeed.zero.Init(config_dict_or_path=config):
        model = ChunkedLossHeads(mode)
    model.student.weight.requires_grad_(False)
    assert model.student.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
    if mode == "distillation":
        model.teacher.weight.requires_grad_(False)
        assert model.teacher.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE

    for valid_counts in ((4, 8), (0, 8)):
        local_valid = valid_counts[rank]
        hidden = torch.randn(1, 8, 8, device=device, requires_grad=True)
        if mode == "sft":
            targets = torch.full((1, 8), -100, device=device, dtype=torch.long)
            targets[:, :local_valid] = 1
            loss, _, _, n_valid = _chunked_cross_entropy_loss(hidden, model.student.weight, 4, shift_labels=targets)
        else:
            targets = torch.zeros((1, 8), device=device)
            targets[:, :local_valid] = 1
            loss, _, n_valid = _chunked_divergence_loss(
                hidden,
                torch.randn_like(hidden),
                model.student.weight,
                model.teacher.weight,
                targets,
                beta=0.5,
                chunk_size=4,
            )

        assert n_valid.item() == local_valid
        assert torch.isfinite(loss).item()
        if local_valid == 0:
            assert loss.item() == 0
        loss.backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all().item()
        assert model.student.weight.ds_status == ZeroParamStatus.NOT_AVAILABLE
        torch.distributed.barrier()

    assert zero3_config is not None
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
