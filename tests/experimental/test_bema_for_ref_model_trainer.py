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
from datasets import DatasetDict, IterableDatasetDict, load_dataset

from trl import DPOConfig
from trl.experimental.bema_for_ref_model import DPOTrainer

from ..testing_utils import TrlTestCase


class TestBEMAForRefModelDPOTrainer(TrlTestCase):
    @pytest.mark.parametrize(
        "eval_dataset_type",
        [
            "dataset",
            "iterable_dataset",
            "dataset_dict",
            "iterable_dataset_dict",
            "dict_of_dataset",
            "dict_of_iterable_dataset",
            "none",
        ],
    )
    def test_init_with_eval_dataset(self, eval_dataset_type):
        train_dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

        if eval_dataset_type == "none":
            eval_dataset = None
        else:
            streaming = "iterable" in eval_dataset_type
            eval_split = load_dataset(
                "trl-internal-testing/zen", "standard_preference", split="test", streaming=streaming
            )
            if eval_dataset_type in ("dataset", "iterable_dataset"):
                eval_dataset = eval_split
            elif eval_dataset_type in ("dataset_dict", "iterable_dataset_dict"):
                dataset_dict_cls = IterableDatasetDict if streaming else DatasetDict
                eval_dataset = dataset_dict_cls({"data1": eval_split, "data2": eval_split})
            else:  # "dict_of_dataset" or "dict_of_iterable_dataset"
                eval_dataset = {"data1": eval_split, "data2": eval_split}

        training_args = DPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = DPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
            # Each split was tokenized independently.
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data1"]))
            assert "prompt_ids" in next(iter(trainer.eval_dataset["data2"]))
        else:
            assert "prompt_ids" in next(iter(trainer.eval_dataset))
