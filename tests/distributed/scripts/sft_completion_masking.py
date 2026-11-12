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

"""Run with torchrun to test completion-only updates and failure propagation from an unsupervised rank."""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset

from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--invalid-rank", type=int, default=-1)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    use_cpu = not torch.cuda.is_available()
    dataset = Dataset.from_list(
        [{"input_ids": [1, 2, 3], "labels": [-100, -100, -100] if rank == args.invalid_rank else [-100, 2, 3]}] * 4
    )
    trainer = SFTTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            use_cpu=use_cpu,
            ddp_backend="gloo" if use_cpu else "nccl",
            max_steps=1,
            per_device_train_batch_size=1,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    before = trainer.model.get_input_embeddings().weight.detach().clone()
    result = trainer.train()
    assert 0 < result.training_loss < float("inf")
    assert not torch.equal(before, trainer.model.get_input_embeddings().weight)
    assert torch.isfinite(trainer.model.get_input_embeddings().weight).all()
    sys.stdout.write(json.dumps({"rank": rank, "updated": True, "loss": result.training_loss}) + "\n")


if __name__ == "__main__":
    main()
