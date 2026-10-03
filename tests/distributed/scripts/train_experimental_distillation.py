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

"""One-step GKD/GOLD training entry point for the distributed regression tests."""

import argparse

from datasets import load_dataset
from torch.distributed.fsdp import FSDPModule
from transformers import AutoTokenizer

from trl.experimental.gkd import GKDConfig, GKDTrainer
from trl.experimental.gold import GOLDConfig, GOLDTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=["gkd", "gold"], required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("trl-internal-testing/zen", "conversational_language_modeling", split="train")
    config_kwargs = dict(
        output_dir=args.output_dir,
        max_steps=1,
        per_device_train_batch_size=1,
        max_length=64,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
    )
    if args.trainer == "gkd":
        trainer_class = GKDTrainer
        config = GKDConfig(max_new_tokens=8, **config_kwargs)
    else:
        trainer_class = GOLDTrainer
        config = GOLDConfig(max_completion_length=8, **config_kwargs)

    trainer = trainer_class(
        model=model_id,
        teacher_model=model_id,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if trainer.is_fsdp_enabled:
        assert isinstance(trainer.teacher_model, FSDPModule)
    trainer.train()
    assert trainer.state.global_step == 1
