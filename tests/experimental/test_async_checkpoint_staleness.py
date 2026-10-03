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
import queue
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from accelerate import PartialState
from datasets import Dataset

from trl.experimental.async_distillation import async_distillation_trainer as distillation
from trl.experimental.async_grpo import async_grpo_trainer as grpo
from trl.trainer.base_trainer import _BaseTrainer


@pytest.fixture(params=[grpo, distillation], ids=["grpo", "distillation"])
def trainer_state(request, tmp_path):
    PartialState(cpu=True)
    module = request.param
    is_grpo = module is grpo
    trainer_cls = module.AsyncGRPOTrainer if is_grpo else module.AsyncDistillationTrainer
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.accelerator = MagicMock()
    trainer.accelerator.is_main_process = True
    trainer.rollout_worker = MagicMock(spec=module.AsyncRolloutWorker)
    trainer.rollout_worker._loop_kwargs = {"dataset_start_index": 10}
    trainer.state = SimpleNamespace(global_step=5)
    trainer.model_version = 7
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer.train_dataset = Dataset.from_dict({"prompt": list(range(100))})
    trained = {0, 1, 3, 4, 5}
    dropped = {2}
    if is_grpo:
        trainer._trained_groups = trained
        trainer._dropped_groups = dropped
        trainer._groups_before_resume = 8
    else:
        trainer._trained_prompts = trained
        trainer._dropped_prompts = dropped
        trainer._prompts_before_resume = 8
    return module, trainer, trained, dropped


@pytest.mark.parametrize("missing_group", [False, True])
def test_checkpoint_advances_past_stale_drops_but_not_inflight(trainer_state, tmp_path, missing_group):
    module, trainer, trained, dropped = trainer_state
    if missing_group:
        trained.remove(4)
    with patch.object(_BaseTrainer, "_save_checkpoint"):
        trainer._save_checkpoint(None, None)
    state = json.loads((tmp_path / "checkpoint-5" / "rollout_state.json").read_text())
    assert state["prompt_index"] == (14 if missing_group else 16)
    assert state["trained_prompt_count"] == 8 + len(trained)


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_resume_restores_training_count_separately_from_cursor(trainer_state, tmp_path, legacy_checkpoint):
    module, trainer, trained, dropped = trainer_state
    state = {"prompt_index": 16, "model_version": 7}
    if not legacy_checkpoint:
        state["trained_prompt_count"] = 13
    (tmp_path / "rollout_state.json").write_text(json.dumps(state))
    trainer.accelerator.is_main_process = False
    with patch.object(_BaseTrainer, "_inner_training_loop"):
        trainer._inner_training_loop(resume_from_checkpoint=str(tmp_path))
    assert trainer.rollout_worker._loop_kwargs["dataset_start_index"] == 16
    before_resume = trainer._groups_before_resume if module is grpo else trainer._prompts_before_resume
    assert before_resume == (16 if legacy_checkpoint else 13)
    assert dropped == set()
    assert trained == set()


def test_queue_stale_drop_is_saved_without_counting_as_training(trainer_state, tmp_path):
    module, trainer, trained, dropped = trainer_state
    trained.clear()
    dropped.clear()
    sample_queue = queue.Queue()
    for prompt_id, version in [(0, 0), (1, 5)]:
        sample_queue.put(
            SimpleNamespace(
                input_ids=[1, 2],
                completion_mask=[0, 1],
                old_log_probs=[0.0, -0.5],
                advantage=1.0,
                model_version=version,
                group_id=prompt_id,
                prompt_id=prompt_id,
                metrics={},
                enqueued_at=None,
                teacher_topk_ids=[[], [2]],
                teacher_topk_logprobs=[[], [-0.5]],
                teacher_id="teacher",
            )
        )
    metrics = defaultdict(list)
    dataset = module.RolloutQueueDataset(
        rollout_queue=sample_queue,
        model_version_fn=lambda: 5,
        check_health_fn=lambda _: None,
        stale_after_s=60,
        metrics=metrics,
        dropped_prompts=dropped,
        max_staleness=3,
    )
    example = next(iter(dataset))
    assert dropped == {0}
    assert metrics["sample/dropped_stale_total"] == [1.0]
    assert trained == set()
    if module is grpo:
        collator = module.DataCollatorForRollout(0, groups_trained=trained)
    else:
        collator = module.DataCollatorForRollout(0, teacher_top_k=1, prompts_trained=trained)
    collator([[[example]]])
    assert trained == {1}
    with patch.object(_BaseTrainer, "_save_checkpoint"):
        trainer._save_checkpoint(None, None)
    state = json.loads((tmp_path / "checkpoint-5" / "rollout_state.json").read_text())
    assert state["prompt_index"] == 12
    assert state["trained_prompt_count"] == 9
