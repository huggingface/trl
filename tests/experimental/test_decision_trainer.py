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

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed


pytest.importorskip("transformers.decision")

from trl.experimental.decision import DecisionConfig, DecisionTrainer

from ..testing_utils import TrlTestCase, require_peft


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def dataset(tokenizer):
    encoded = tokenizer(["A refund", "A delivery", "A complaint", "Another refund", "Another delivery"])
    return Dataset.from_dict({**encoded, "labels": [0, 1, 2, 0, 1]})


def make_model():
    set_seed(42)
    return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3, dtype=torch.float32)


class TestDecisionTrainer(TrlTestCase):
    @pytest.mark.parametrize("use_peft", [False, pytest.param(True, marks=require_peft)])
    def test_train_and_predict(self, tokenizer, dataset, use_peft):
        peft_config = None
        if use_peft:
            from peft import LoraConfig

            peft_config = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")

        def compute_metrics(prediction):
            return {"accuracy": float((prediction.predictions.argmax(-1) == prediction.label_ids).mean())}

        trainer = DecisionTrainer(
            model=make_model(),
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_steps=1,
                report_to="none",
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            peft_config=peft_config,
        )
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert any(not torch.equal(before[name], parameter) for name, parameter in trainer.model.named_parameters())
        prediction = trainer.predict(dataset)
        assert prediction.predictions.shape == (len(dataset), 3)
        np.testing.assert_allclose(prediction.predictions.sum(-1), 1, atol=1e-6)
        np.testing.assert_array_equal(prediction.label_ids, dataset["labels"])
        metrics = trainer.evaluate()
        assert metrics["eval_choice_accuracy"] == pytest.approx(metrics["eval_accuracy"])
        assert "eval_coverage_loss" not in metrics
        assert "eval_null_precision" not in metrics
        trainer.save_model()

    def test_accumulation(self, tokenizer, dataset):
        models = []
        for batch_size, accumulation in ((5, 1), (3, 2)):
            model = make_model()
            trainer = DecisionTrainer(
                model=model,
                args=DecisionConfig(
                    output_dir=self.tmp_dir,
                    use_cpu=True,
                    bf16=False,
                    max_steps=1,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    max_grad_norm=0,
                    lr_scheduler_type="constant",
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
            )
            trainer.train()
            models.append(model)
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
