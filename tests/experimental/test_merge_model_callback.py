# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl import DPOConfig, DPOTrainer
from trl.experimental.merge_model_callback import MergeConfig, MergeModelCallback

from ..testing_utils import TrlTestCase, require_mergekit


@require_mergekit
class TestMergeModelCallback(TrlTestCase):
    def setup_method(self):
        self.model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        self.dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

    def test_callback(self):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            num_train_epochs=1,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
        )
        config = MergeConfig()
        merge_callback = MergeModelCallback(config)
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            callbacks=[merge_callback],
        )
        trainer.train()
        last_checkpoint = get_last_checkpoint(self.tmp_dir)
        merged_path = os.path.join(last_checkpoint, "merged")
        assert os.path.isdir(merged_path), "Merged folder does not exist in the last checkpoint."

    def test_every_checkpoint(self):
        training_args = DPOConfig(
            output_dir=self.tmp_dir,
            num_train_epochs=1,
            report_to="none",
            save_strategy="steps",
            save_steps=1,
        )
        config = MergeConfig()
        merge_callback = MergeModelCallback(config, merge_at_every_checkpoint=True)
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
            callbacks=[merge_callback],
        )
        trainer.train()

        checkpoints = sorted(
            [os.path.join(self.tmp_dir, cp) for cp in os.listdir(self.tmp_dir) if cp.startswith("checkpoint-")]
        )

        for checkpoint in checkpoints:
            merged_path = os.path.join(checkpoint, "merged")
            assert os.path.isdir(merged_path), f"Merged folder does not exist in checkpoint {checkpoint}."
