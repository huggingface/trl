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

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.mark.parametrize("padding_free", [False, True])
@pytest.mark.parametrize("labels", [[-100, -100, -100], [1, -100, -100], [1]])
def test_collator_rejects_no_next_token_targets(padding_free, labels):
    collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=padding_free)
    with pytest.raises(ValueError, match="no supervised next-token targets"):
        collator([{"input_ids": list(range(1, len(labels) + 1)), "labels": labels}])


@pytest.mark.parametrize("padding_free", [False, True])
def test_collator_accepts_mixed_supervision(padding_free):
    collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=padding_free)
    batch = collator(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, -100]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]
    )
    assert batch["labels"][:, 1:].ne(-100).sum().item() == 1
    assert batch["labels"][batch["labels"] != -100].tolist() == [5]


def test_packed_document_starts_are_not_targets():
    collator = DataCollatorForLanguageModeling(pad_token_id=0, padding_free=True)
    with pytest.raises(ValueError, match="no supervised next-token targets"):
        collator([{"input_ids": [1, 2, 3, 4], "labels": [1, -100, 3, -100], "seq_lengths": [2, 2]}])


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("valid", [False, True])
def test_collator_in_dataloader_workers(workers, valid):
    options = {"multiprocessing_context": "spawn"} if workers else {}
    labels = [-100, 2, 3] if valid else [-100, -100, -100]
    loader = DataLoader(
        [{"input_ids": [1, 2, 3], "labels": labels}],
        batch_size=1,
        num_workers=workers,
        collate_fn=DataCollatorForLanguageModeling(pad_token_id=0),
        **options,
    )
    iterator = iter(loader)
    try:
        if valid:
            assert next(iterator)["labels"][:, 1:].ne(-100).sum().item() == 2
        else:
            with pytest.raises(ValueError, match="no supervised next-token targets"):
                next(iterator)
        assert next(iterator, None) is None
    finally:
        del iterator


@pytest.fixture
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_ID)


@pytest.mark.parametrize("use_fast", [False, True])
def test_qwen_newline_marker_does_not_need_isolated_token_matching(model, tmp_path, use_fast):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=use_fast)
    prompt = "Question\n\n### Assistant:"
    completion = " The answer is Paris."
    ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    marker = tokenizer.encode("\n### Assistant:", add_special_tokens=False)
    assert not any(ids[i : i + len(marker)] == marker for i in range(len(ids)))
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list([{"prompt": prompt, "completion": completion}]),
        args=SFTConfig(
            output_dir=str(tmp_path),
            completion_only_loss=True,
            max_length=None,
            use_cpu=True,
            report_to="none",
            max_steps=1,
            save_strategy="no",
            disable_tqdm=True,
        ),
    )
    row = trainer.train_dataset[0]
    assert len(row["input_ids"]) == len(row["labels"])
    assert row["input_ids"] == tokenizer.encode(prompt + completion + tokenizer.eos_token)
    prompt_length = len(tokenizer.encode(prompt))
    assert row["labels"][:prompt_length] == [-100] * prompt_length
    assert tokenizer.decode([label for label in row["labels"] if label != -100]) == completion + tokenizer.eos_token
    before = model.get_input_embeddings().weight.detach().clone()
    result = trainer.train()
    assert math.isfinite(result.training_loss) and result.training_loss > 0
    assert not torch.equal(before, model.get_input_embeddings().weight)


@pytest.mark.parametrize("skip_prepare", [False, True])
def test_trainer_rejects_unsupervised_batch_without_truncation(model, tmp_path, skip_prepare):
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [-100, -100, -100]}]),
        args=SFTConfig(
            output_dir=str(tmp_path),
            max_length=None,
            use_cpu=True,
            report_to="none",
            max_steps=1,
            save_strategy="no",
            dataset_kwargs={"skip_prepare_dataset": skip_prepare},
        ),
    )
    with pytest.raises(ValueError, match="no supervised next-token targets"):
        trainer.train()


def test_truncation_drops_first_token_only_supervision(model, tmp_path):
    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(
            [
                {"input_ids": [1, 2, 3, 4], "labels": [1, -100, -100, 4]},
                {"input_ids": [5, 6, 7, 8], "labels": [-100, 6, -100, 8]},
            ]
        ),
        args=SFTConfig(output_dir=str(tmp_path), max_length=3, use_cpu=True, report_to="none"),
    )
    assert trainer.train_dataset[:]["labels"] == [[-100, 6, -100]]


def test_assistant_and_completion_masks_intersect(tmp_path):
    trainer = SFTTrainer(
        model="trl-internal-testing/tiny-Qwen3ForCausalLM",
        train_dataset=Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "Name a fruit."}],
                    "completion": [
                        {"role": "assistant", "content": "Apple."},
                        {"role": "user", "content": "UNSUPERVISED_USER_TEXT"},
                        {"role": "assistant", "content": "Pear."},
                    ],
                }
            ]
        ),
        args=SFTConfig(
            output_dir=str(tmp_path),
            assistant_only_loss=True,
            completion_only_loss=True,
            max_length=None,
            use_cpu=True,
            report_to="none",
        ),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][batch["labels"] != -100]
    text = trainer.processing_class.decode(labels)
    assert "Apple." in text and "Pear." in text
    assert "UNSUPERVISED_USER_TEXT" not in text
