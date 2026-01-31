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

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.sdft import SDFTConfig, SDFTTrainer

from ..testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def build_dataset():
    return Dataset.from_dict(
        {
            "prompt": ["Write a short poem about the sea."],
            "teacher_prompt": ["Write a short poem about the sea."],
        }
    )


class TestSDFTTrainer(TrlTestCase):
    def _build_args(self):
        return SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            num_generations=1,
            report_to="none",
        )

    def _build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def test_init_creates_default_teacher(self):
        args = self._build_args()
        tokenizer = self._build_tokenizer()
        trainer = SDFTTrainer(
            model=MODEL_ID,
            args=args,
            processing_class=tokenizer,
            train_dataset=build_dataset(),
        )
        assert trainer.ref_model is not None
        assert trainer.ref_model is not trainer.model

    def test_init_with_ref_model_id(self):
        args = self._build_args()
        tokenizer = self._build_tokenizer()
        trainer = SDFTTrainer(
            model=MODEL_ID,
            ref_model=MODEL_ID,
            args=args,
            processing_class=tokenizer,
            train_dataset=build_dataset(),
        )
        assert trainer.ref_model is not None

    def test_missing_teacher_prompt_raises(self):
        args = self._build_args()
        tokenizer = self._build_tokenizer()
        bad_dataset = Dataset.from_dict({"prompt": ["Hello"]})
        with pytest.raises(ValueError, match="teacher_prompt"):
            SDFTTrainer(
                model=MODEL_ID,
                args=args,
                processing_class=tokenizer,
                train_dataset=bad_dataset,
            )

    @pytest.mark.low_priority
    def test_train_updates_student_and_freezes_teacher(self):
        args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            num_generations=1,
            max_completion_length=8,
            max_steps=1,
            logging_steps=1,
            report_to="none",
            save_strategy="no",
            eval_strategy="no",
        )
        tokenizer = self._build_tokenizer()
        trainer = SDFTTrainer(
            model=MODEL_ID,
            args=args,
            processing_class=tokenizer,
            train_dataset=build_dataset(),
        )

        student_before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
        teacher_before = {n: p.detach().clone() for n, p in trainer.ref_model.named_parameters()}

        trainer.train()

        # Student params should change
        student_changed = False
        for name, before in student_before.items():
            after = trainer.model.get_parameter(name).detach()
            if not torch.allclose(before, after):
                student_changed = True
                break
        assert student_changed, "Student parameters did not update after training"

        # Teacher params should remain frozen
        for name, before in teacher_before.items():
            after = trainer.ref_model.get_parameter(name).detach()
            assert torch.allclose(before, after), f"Teacher parameter {name} changed during training"
