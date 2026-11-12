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
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import SFTConfig, SFTTrainer

MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


class GradForcingLossTrainer(SFTTrainer):
    """Trainer whose `compute_loss` returns a tensor that requires grad, mimicking an integration (e.g. a checkpointing
    output hook) that re-enables `requires_grad_` on evaluation losses even under `torch.no_grad()`."""

    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):  # return_outputs=True
            loss = (loss[0].requires_grad_(True), loss[1])
        else:
            loss = loss.requires_grad_(True)
        return loss


class TestPredictionStepDetachesLoss:
    @pytest.mark.parametrize("prediction_loss_only", [True, False])
    def test_prediction_step_loss_detached(self, prediction_loss_only):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="float32")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        def collate(features):
            batch = tokenizer([f["prompt"] for f in features], return_tensors="pt", padding=True)
            batch["labels"] = batch["input_ids"].clone()
            return batch

        trainer = GradForcingLossTrainer(
            model=model,
            args=SFTConfig(output_dir="tmp_trl_test", report_to="none"),
            train_dataset=Dataset.from_list([{"prompt": "Hello, my dog is", "completion": " cute"}]),
            data_collator=collate,
        )
        inputs = collate([{"prompt": "Hello, my dog is cute"}])
        loss, logits, labels = trainer.prediction_step(
            trainer.model, trainer._prepare_inputs(inputs), prediction_loss_only=prediction_loss_only
        )
        assert loss is not None
        # Regression check: the base trainer detached evaluation losses, so should we (see #8246)
        assert not loss.requires_grad
        if prediction_loss_only:
            assert logits is None
        else:
            assert logits is None or not logits.requires_grad
