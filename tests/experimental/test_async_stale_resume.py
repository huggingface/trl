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

import json
import os
import queue
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from accelerate.state import PartialState
from datasets import Dataset

from trl.experimental.async_distillation import AsyncDistillationTrainer
from trl.experimental.async_distillation.async_distillation_trainer import (
    RolloutQueueDataset as DistillationRolloutQueueDataset,
)
from trl.experimental.async_distillation.async_rollout_worker import AsyncRolloutWorker as DistillationRolloutWorker
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import RolloutQueueDataset as GRPORolloutQueueDataset
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker as GRPORolloutWorker
from trl.trainer.base_trainer import _BaseTrainer


PartialState()


class _QueueDrained(Exception):
    pass


def _raise_queue_drained(_timeout):
    raise _QueueDrained


def _stale_dataset(dataset_cls, id_name, id_value, dropped):
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.queue = queue.Queue()
    dataset.queue.put(SimpleNamespace(**{id_name: id_value, "model_version": 0, "enqueued_at": None}))
    dataset.model_version_fn = lambda: 5
    dataset.check_health_fn = _raise_queue_drained
    dataset.stale_after_s = 1.0
    dataset.metrics = defaultdict(list)
    dataset.wait_s = 0.0
    dataset.max_staleness = 4
    dataset.poll_interval_s = 0.0
    dataset.report_to = []
    dataset._trace_buf = []
    dataset._traces_per_log = 8
    dataset._trace_log_interval = 8
    dataset._last_traced_version = -1
    if id_name == "group_id":
        dataset.dropped_groups = dropped
    else:
        dataset.dropped_prompts = dropped
    with pytest.raises(_QueueDrained):
        next(iter(dataset))
    assert dataset.metrics["sample/dropped_stale_total"] == [1.0]


@pytest.mark.parametrize(
    ("dataset_cls", "id_name", "id_value"),
    [
        (GRPORolloutQueueDataset, "group_id", 7),
        (DistillationRolloutQueueDataset, "prompt_id", 7),
    ],
)
def test_stale_drop_is_recorded_as_resolved_prompt(dataset_cls, id_name, id_value):
    dropped = set()
    _stale_dataset(dataset_cls, id_name, id_value, dropped)
    assert dropped == {id_value}


def _save_checkpoint(trainer_cls, worker_cls, out_dir, trained, dropped, before_resume, dataset_start_index):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.model_version = 4
    trainer.rollout_worker = MagicMock(spec=worker_cls)
    trainer.rollout_worker._loop_kwargs = {"dataset_start_index": dataset_start_index}
    trainer.state = SimpleNamespace(global_step=7)
    trainer._get_output_dir = lambda trial: out_dir
    if trainer_cls is AsyncGRPOTrainer:
        trainer._trained_groups = trained
        trainer._dropped_groups = dropped
        trainer._groups_before_resume = before_resume
    else:
        trainer._trained_prompts = trained
        trainer._dropped_prompts = dropped
        trainer._prompts_before_resume = before_resume
    with patch.object(_BaseTrainer, "_save_checkpoint"):
        trainer._save_checkpoint(MagicMock(), None)
    checkpoint = os.path.join(out_dir, "checkpoint-7", "rollout_state.json")
    with open(checkpoint, encoding="utf-8") as stream:
        return json.load(stream)


@pytest.mark.parametrize(
    ("trainer_cls", "worker_cls"),
    [
        (AsyncGRPOTrainer, GRPORolloutWorker),
        (AsyncDistillationTrainer, DistillationRolloutWorker),
    ],
)
def test_checkpoint_cursor_skips_stale_hole_without_counting_it_as_trained(trainer_cls, worker_cls, tmp_path):
    # Groups/prompts 0, 1, and 3–5 trained; 2 was stale-dropped. Resume starts at 6, while the epoch counter
    # records only the five training signals (plus any count from an earlier resume).
    state = _save_checkpoint(
        trainer_cls,
        worker_cls,
        str(tmp_path),
        trained={0, 1, 3, 4, 5},
        dropped={2},
        before_resume=10,
        dataset_start_index=20,
    )
    assert state["prompt_index"] == 26
    assert state["model_version"] == 4
    count_key = "groups_trained" if trainer_cls is AsyncGRPOTrainer else "prompts_trained"
    assert state[count_key] == 15


def _resume(trainer_cls, worker_cls, checkpoint_dir):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.rollout_worker = MagicMock(spec=worker_cls)
    trainer.rollout_worker._loop_kwargs = {}
    trainer.train_dataset = Dataset.from_dict({"prompt": ["a"]})
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.model_version = 0
    if trainer_cls is AsyncGRPOTrainer:
        trainer._groups_before_resume = 0
        trainer._trained_groups = {99}
        trainer._dropped_groups = {99}
    else:
        trainer._prompts_before_resume = 0
        trainer._trained_prompts = {99}
        trainer._dropped_prompts = {99}
    with patch.object(_BaseTrainer, "_inner_training_loop", return_value=None):
        trainer._inner_training_loop(resume_from_checkpoint=checkpoint_dir)
    return trainer


@pytest.mark.parametrize(
    ("trainer_cls", "worker_cls", "count_key", "count_attr", "dropped_attr"),
    [
        (AsyncGRPOTrainer, GRPORolloutWorker, "groups_trained", "_groups_before_resume", "_dropped_groups"),
        (
            AsyncDistillationTrainer,
            DistillationRolloutWorker,
            "prompts_trained",
            "_prompts_before_resume",
            "_dropped_prompts",
        ),
    ],
)
def test_resume_restores_cursor_and_actual_training_count(
    trainer_cls, worker_cls, count_key, count_attr, dropped_attr, tmp_path
):
    checkpoint_dir = tmp_path / "checkpoint-7"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "rollout_state.json").write_text(
        json.dumps({"prompt_index": 26, count_key: 15, "model_version": 4}), encoding="utf-8"
    )
    trainer = _resume(trainer_cls, worker_cls, str(checkpoint_dir))
    assert trainer.rollout_worker._loop_kwargs["dataset_start_index"] == 26
    assert getattr(trainer, count_attr) == 15
    assert getattr(trainer, dropped_attr) == set()
    if trainer_cls is AsyncGRPOTrainer:
        assert trainer._trained_groups == set()
    assert trainer.model_version == 4
    trainer.rollout_worker.update_model_version.assert_called_once_with(4)


def test_resume_uses_cursor_as_legacy_training_count(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint-legacy"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "rollout_state.json").write_text(
        json.dumps({"prompt_index": 26, "model_version": 4}), encoding="utf-8"
    )
    trainer = _resume(AsyncGRPOTrainer, GRPORolloutWorker, str(checkpoint_dir))
    assert trainer.rollout_worker._loop_kwargs["dataset_start_index"] == 26
    assert trainer._groups_before_resume == 26


@pytest.mark.parametrize(
    ("trainer_cls", "module_name", "dropped_attr", "dropped_kwarg"),
    [
        (
            AsyncGRPOTrainer,
            "trl.experimental.async_grpo.async_grpo_trainer",
            "_dropped_groups",
            "dropped_groups",
        ),
        (
            AsyncDistillationTrainer,
            "trl.experimental.async_distillation.async_distillation_trainer",
            "_dropped_prompts",
            "dropped_prompts",
        ),
    ],
)
def test_train_dataloader_shares_dropped_id_tracker(trainer_cls, module_name, dropped_attr, dropped_kwarg):
    import importlib

    module = importlib.import_module(module_name)
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.accelerator = SimpleNamespace(num_processes=1, is_main_process=True, prepare=lambda value: value)
    trainer.rollout_queue = object()
    trainer.model_version = 0
    trainer.rollout_worker = SimpleNamespace(check_health=lambda _timeout: None)
    trainer.args = SimpleNamespace(
        heartbeat_stale_after_s=1.0,
        max_staleness=4,
        report_to=[],
        token_budget=0,
        per_device_train_batch_size=1,
    )
    trainer._metrics = {"train": defaultdict(list)}
    trainer.processing_class = SimpleNamespace(pad_token_id=0)
    dropped = set()
    setattr(trainer, dropped_attr, dropped)
    if trainer_cls is AsyncGRPOTrainer:
        trainer._trained_groups = set()
    else:
        trainer._trained_prompts = set()
        trainer._teacher_ids = {"teacher"}
        trainer._teacher_id_to_idx = {"teacher": 0}

    with (
        patch.object(module, "RolloutQueueDataset", return_value=object()) as queue_dataset,
        patch.object(module, "FixedCountBatcher", return_value=object()),
        patch.object(module, "DataLoader", return_value=object()),
    ):
        trainer.get_train_dataloader()

    assert queue_dataset.call_args.kwargs[dropped_kwarg] is dropped
