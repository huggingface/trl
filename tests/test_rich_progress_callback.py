# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import tempfile
import unittest

import torch
import torch.nn as nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from trl.trainer.callbacks import RichProgressCallback


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.a * x


class TestRichProgressCallback(unittest.TestCase):
    def setUp(self):
        self.dummy_model = DummyModel()
        self.dummy_train_dataset = Dataset.from_list([{"x": 1.0, "y": 2.0}] * 5)
        self.dummy_val_dataset = Dataset.from_list([{"x": 1.0, "y": 2.0}] * 101)

    def test_rich_progress_callback_logging(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
                per_device_eval_batch_size=2,
                per_device_train_batch_size=2,
                num_train_epochs=4,
                eval_strategy="steps",
                eval_steps=1,
                logging_strategy="steps",
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                disable_tqdm=True,
            )
            callbacks = [RichProgressCallback()]
            trainer = Trainer(
                model=self.dummy_model,
                train_dataset=self.dummy_train_dataset,
                eval_dataset=self.dummy_val_dataset,
                args=training_args,
                callbacks=callbacks,
            )

            trainer.train()
            trainer.train()
