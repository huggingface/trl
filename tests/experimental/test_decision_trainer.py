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
from functools import partial

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from packaging.version import Version


pytest.importorskip("transformers.decision")

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    JevForDecision,
    set_seed,
)
from transformers.decision import DecisionConfig as TransformersDecisionConfig
from transformers.decision import parse_questions
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from trl.experimental.decision import DataCollatorForDecision, DecisionConfig, DecisionTrainer
from trl.experimental.decision.data_collator import _normalize_target

from ..testing_utils import TrlTestCase, require_peft


HAS_JEVA = "jeva" in CONFIG_MAPPING
HAS_CLM = "clm" in CONFIG_MAPPING
HAS_JEV_PREPARATION = "JevForDecision" in LOSS_MAPPING and hasattr(JevForDecision, "prepare_inputs_for_decision")

if HAS_JEVA:
    from transformers import AutoModelForDecision, JevaForDecision
    from transformers import JevaConfig as JevaModelConfig


# Transformers 5.23 renamed the TrainingArguments checkpointing fields.
CHECKPOINTING_ARG = (
    "activation_checkpointing" if Version(transformers.__version__) >= Version("5.23.0") else "gradient_checkpointing"
)


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def classification_dataset(tokenizer):
    encoded = tokenizer(["A refund", "A delivery", "A complaint", "Another refund", "Another delivery"])
    return Dataset.from_dict({**encoded, "labels": [0, 1, 2, 0, 1]})


def make_classification_model():
    set_seed(42)
    return AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3, dtype=torch.float32)


class TestDecisionTrainer(TrlTestCase):
    @pytest.mark.parametrize("use_peft", [False, pytest.param(True, marks=require_peft)])
    def test_train_and_predict(self, tokenizer, classification_dataset, use_peft):
        peft_config = None
        if use_peft:
            from peft import LoraConfig

            peft_config = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")

        def compute_metrics(prediction):
            return {"accuracy": float((prediction.predictions.argmax(-1) == prediction.label_ids).mean())}

        trainer = DecisionTrainer(
            model=make_classification_model(),
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
            train_dataset=classification_dataset,
            eval_dataset=classification_dataset,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            peft_config=peft_config,
        )
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert any(not torch.equal(before[name], parameter) for name, parameter in trainer.model.named_parameters())
        prediction = trainer.predict(classification_dataset)
        assert prediction.predictions.shape == (len(classification_dataset), 3)
        np.testing.assert_allclose(prediction.predictions.sum(-1), 1, atol=1e-6)
        np.testing.assert_array_equal(prediction.label_ids, classification_dataset["labels"])
        metrics = trainer.evaluate()
        assert metrics["eval_choice_accuracy"] == pytest.approx(metrics["eval_accuracy"])
        assert "eval_coverage_loss" not in metrics
        assert "eval_null_precision" not in metrics
        trainer.save_model()

    def test_accumulation(self, tokenizer, classification_dataset):
        models = []
        for batch_size, accumulation in ((5, 1), (3, 2)):
            model = make_classification_model()
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
                train_dataset=classification_dataset,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                optimizers=(torch.optim.SGD(model.parameters(), lr=0.01), None),
            )
            trainer.train()
            models.append(model)
        for left, right in zip(models[0].parameters(), models[1].parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.fixture
def dataset():
    records = [
        (
            "A billing question",
            {"type": "choice", "instructions": "Choose a team", "criteria": {"billing": None, "technical": None}},
            [0, 1, 0],
        ),
        (
            "No service applies",
            {"type": "choice", "instructions": "Choose a service", "criteria": {"shipping": None}, "allow_null": True},
            [1, 0],
        ),
        ("The answer is mostly true", {"type": "noul", "instructions": "Is the answer true?"}, [0, 0.8, 0.2]),
        (
            "A good answer",
            {"type": "score", "instructions": "Rate the answer", "criteria": ["bad", "good", "great"]},
            [0, 0, 0.5, 0.5],
        ),
        (
            "A refund",
            {
                "type": "choice",
                "instructions": "Choose a team",
                "criteria": {"yes": None, "no": None},
                "allow_null": True,
            },
            [0.1, 0.7, 0.2],
        ),
    ]
    return Dataset.from_dict(
        {
            "request": [
                json.dumps({"state": state, "questions": {"decision": question}}) for state, question, _ in records
            ],
            "target_probs": [target for _, _, target in records],
        }
    )


def make_model(model_id=MODEL_ID, dtype=torch.float32, mode="head", base_path=None):
    set_seed(42)
    encoder = AutoModel.from_pretrained(model_id, dtype=dtype)
    model = JevaForDecision(
        JevaModelConfig(text_config=encoder.config, projection_dim=16, decision_num_heads=2), model=encoder
    )
    model.base_model.requires_grad_(mode == "full")
    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        model.save_pretrained(base_path)
        model.config._name_or_path = str(base_path)
        config = LoraConfig(
            r=2,
            target_modules=r"model\..*\.(q_proj|v_proj)",
            task_type="FEATURE_EXTRACTION",
            modules_to_save=["context_proj", "option_proj", "core"],
        )
        model = get_peft_model(model, config)
    return model


@pytest.fixture
def collator(tokenizer):
    model = make_model()
    return DataCollatorForDecision(
        tokenizer,
        model.decision_config,
        prepare_inputs_for_decision=partial(
            model.prepare_inputs_for_decision, pad_token_id=tokenizer.pad_token_id, option_chunk_size=1
        ),
    )


@pytest.mark.skipif(not HAS_JEVA, reason="Requires Transformers #50614")
class TestDecisionTrainerJeva(TrlTestCase):
    @pytest.mark.parametrize("mode", ["head", "full", pytest.param("lora", marks=require_peft)])
    def test_train(self, mode, tokenizer, dataset, collator):
        model = make_model(mode=mode, base_path=os.path.join(self.tmp_dir, "base_model"))
        trainer = DecisionTrainer(
            model=model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=2,
                logging_steps=1,
                report_to="none",
                **{CHECKPOINTING_ARG: mode != "head"},
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        logs = trainer.state.log_history[0]
        assert logs["loss"] == pytest.approx(logs["candidate_loss"] + logs["coverage_loss"], abs=1e-4)
        changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
        for component in ("context_proj", "option_proj", "core"):
            assert any(f"{component}." in name for name in changed)
        assert all(parameter.requires_grad for name, parameter in model.named_parameters() if name in changed)
        if mode == "head":
            assert not any(name.startswith("model.") for name in changed)
        elif mode == "lora":
            assert any("lora_" in name for name in changed)
            assert all("lora_" in name for name in changed if ".model.model." in name)
        else:
            assert any(name.startswith("model.") for name in changed)

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_loss_and_predictions(self, tokenizer, dataset, batch_size, collator):
        model = make_model().eval()
        batch = collator(list(dataset))
        with torch.no_grad():
            outputs = model(
                **{
                    key: batch[key]
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "option_input_ids",
                        "option_attention_mask",
                        "num_options",
                        "option_mask",
                    )
                }
            )
        log_q = outputs.logits.masked_fill(~batch["option_mask"], 0)
        expected = -(batch["labels"][:, 1:] * log_q).sum(-1)
        expected += (
            F.binary_cross_entropy_with_logits(outputs.coverage_logits, 1 - batch["labels"][:, 0], reduction="none")
            * batch["allow_null"]
        )

        def compute_metrics(prediction):
            probs, labels = prediction.predictions, prediction.label_ids
            np.testing.assert_allclose(np.where(probs == -100, 0, probs).sum(-1), 1, atol=1e-6)
            assert np.array_equal(probs == -100, labels == -100)
            return {"accuracy": float((probs.argmax(-1) == labels.argmax(-1)).mean())}

        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                report_to="none",
                per_device_eval_batch_size=batch_size,
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        prediction = trainer.predict(dataset)
        assert prediction.predictions.shape == (len(dataset), 4)
        assert prediction.metrics["test_loss"] == pytest.approx(expected.mean().item(), rel=1e-5)
        assert "eval_holdout_accuracy" in trainer.evaluate({"holdout": dataset})
        metrics = trainer.evaluate()
        probs = torch.tensor(np.where(prediction.predictions == -100, 0, prediction.predictions))
        predicted = probs.argmax(-1)
        target = batch["labels"].argmax(-1)
        hard_choice = batch["is_choice"] & (batch["labels"].max(-1).values == 1)
        predicted_null = (predicted == 0) & hard_choice
        target_null = (target == 0) & hard_choice
        true_null = (predicted_null & target_null).sum().item()
        assert metrics["eval_candidate_loss"] + metrics["eval_coverage_loss"] == pytest.approx(
            expected.mean().item(), rel=1e-5
        )
        assert metrics["eval_choice_accuracy"] == pytest.approx(
            ((predicted == target) & hard_choice).sum().item() / hard_choice.sum().item()
        )
        present = hard_choice & (target != 0)
        assert metrics["eval_candidate_accuracy"] == pytest.approx(
            ((outputs.logits.argmax(-1) + 1 == target) & present).sum().item() / present.sum().item()
        )
        assert metrics["eval_null_precision"] == pytest.approx(true_null / max(predicted_null.sum().item(), 1))
        assert metrics["eval_null_recall"] == pytest.approx(true_null / target_null.sum().item())
        assert metrics["eval_null_rate"] == pytest.approx(
            ((predicted == 0) & batch["allow_null"]).sum().item() / batch["allow_null"].sum().item()
        )
        assert metrics["eval_brier"] == pytest.approx(
            (probs - batch["labels"]).square().sum(-1).mean().item(), abs=1e-6
        )
        assert metrics["eval_null_brier"] == pytest.approx(
            (probs[:, 0] - batch["labels"][:, 0]).square()[batch["allow_null"]].mean().item(), abs=1e-6
        )
        assert metrics["eval_entropy"] == pytest.approx(
            -(probs * probs.clamp_min(1e-30).log()).sum(-1).mean().item(), abs=1e-6
        )

    def test_accumulation(self, tokenizer, dataset, collator):
        trained_models = []
        for batch_size, accumulation in ((5, 1), (3, 2)):
            model = make_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            trainer = DecisionTrainer(
                model,
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
                    **{CHECKPOINTING_ARG: False},
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                data_collator=collator,
                optimizers=(optimizer, None),
            )
            trainer.train()
            trained_models.append(model)
        for a, b in zip(trained_models[0].parameters(), trained_models[1].parameters(), strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    @require_peft
    def test_checkpoint_and_resume(self, tokenizer, dataset, collator):
        args = DecisionConfig(
            output_dir=self.tmp_dir,
            use_cpu=True,
            bf16=False,
            max_steps=2,
            save_steps=1,
            report_to="none",
            **{CHECKPOINTING_ARG: False},
        )
        trainer = DecisionTrainer(
            make_model(mode="lora", base_path=os.path.join(self.tmp_dir, "base_model")),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        trainer.train()
        checkpoint = os.path.join(self.tmp_dir, "checkpoint-2")
        from peft import PeftModel

        base_path = os.path.join(self.tmp_dir, "base_model")
        reloaded = PeftModel.from_pretrained(AutoModelForDecision.from_pretrained(base_path), checkpoint)
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])
        assert os.path.isfile(os.path.join(checkpoint, "optimizer.pt"))
        resumed = PeftModel.from_pretrained(
            AutoModelForDecision.from_pretrained(base_path), checkpoint, is_trainable=True
        )
        trainer = DecisionTrainer(
            resumed, args=args, train_dataset=dataset, processing_class=tokenizer, data_collator=collator
        )
        trainer.train(resume_from_checkpoint=os.path.join(self.tmp_dir, "checkpoint-1"))
        assert trainer.state.global_step == 2
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])

    @pytest.mark.parametrize(
        "model_id", ["trl-internal-testing/tiny-LlamaForCausalLM-3.2", "trl-internal-testing/tiny-Qwen3MoeForCausalLM"]
    )
    def test_backbones(self, model_id, dataset):
        model = make_model(model_id, mode="full")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForDecision(
            tokenizer,
            model.decision_config,
            prepare_inputs_for_decision=partial(
                model.prepare_inputs_for_decision, pad_token_id=tokenizer.pad_token_id
            ),
        )
        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=1,
                logging_steps=1,
                report_to="none",
                **{CHECKPOINTING_ARG: False},
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        assert model.get_output_embeddings() is None
        result = trainer.train()
        assert np.isfinite(result.training_loss)


class TestDecisionTargetNormalization(TrlTestCase):
    @pytest.mark.parametrize(
        "question, label, target",
        [
            ({"type": "choice", "criteria": {"a": None, "b": None}}, {"target_index": 1}, [0, 0, 1]),
            ({"type": "noul", "instructions": "Is it true?"}, {"p_true": 0.8}, [0, 0.8, 0.2]),
            (
                {"type": "choice", "criteria": {"a": None, "b": None, "c": None}},
                {"target_probs": {"b": 0.75, "a": 0.25}},
                [0, 0.25, 0.75, 0],
            ),
            (
                {"type": "choice", "criteria": {"c": None, "b": None, "a": None}},
                {"target_probs": {"b": 0.75, "a": 0.25}},
                [0, 0, 0.75, 0.25],
            ),
            pytest.param(
                {"type": "choice", "criteria": {"a": None}, "allow_null": True},
                {"target_index": None},
                [1, 0],
                marks=pytest.mark.skipif(not HAS_JEVA, reason="Requires nullable question parsing"),
            ),
            pytest.param(
                {"type": "score", "criteria": ["bad", "good"], "values": [1, 3]},
                {"score": 2},
                [0, 0.5, 0.5],
                marks=pytest.mark.skipif(not HAS_JEVA, reason="Requires custom rubric values"),
            ),
        ],
    )
    def test_decision_labels(self, question, label, target):
        parsed = parse_questions({"decision": question}, TransformersDecisionConfig())[0]
        assert _normalize_target(label, parsed) == pytest.approx(target)

    def test_unknown_target_option(self):
        parsed = parse_questions(
            {"decision": {"type": "choice", "criteria": {"a": None, "b": None}}},
            TransformersDecisionConfig(),
        )[0]
        with pytest.raises(ValueError, match="unknown option"):
            _normalize_target({"target_probs": {"typo": 1.0}}, parsed)


@pytest.mark.skipif(not HAS_CLM, reason="Requires Transformers #50649")
class TestDecisionTrainerClm(TrlTestCase):
    def test_contrastive_training(self, tokenizer):
        from transformers import ClmConfig, ClmForDecision

        encoder = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
        model = ClmForDecision(
            ClmConfig(text_config=encoder.config, hidden_size=24, projection_dim=8, num_hidden_layers=3), model=encoder
        )
        model.base_model.requires_grad_(False)
        collator = DataCollatorForDecision(
            tokenizer, model.decision_config, prepare_inputs_for_decision=model.prepare_inputs_for_decision
        )
        records = [
            ("An invoice", {"billing": None, "delivery": None}, {"billing": 0.75, "delivery": 0.25}),
            ("A different request", {"delivery": None, "other": None}, {"delivery": 0.2, "other": 0.8}),
            ("Another invoice", {"billing": None, "delivery": None}, {"billing": 1.0}),
        ]
        dataset = Dataset.from_list(
            [
                {
                    "request": json.dumps(
                        {"state": state, "questions": {"intent": {"type": "choice", "criteria": criteria}}}
                    ),
                    "target_probs": target,
                }
                for state, criteria, target in records
            ]
        )
        batch = collator(list(dataset))
        targets = torch.tensor([[0.75, 0.25, 0], [0, 0.2, 0.8], [1, 0, 0]])
        torch.testing.assert_close(batch["labels"], targets)
        before = {name: parameter.detach().clone() for name, parameter in model.base_model.named_parameters()}
        state_head_before = model.state_head.input_projection.weight.detach().clone()
        action_head_before = model.action_head.input_projection.weight.detach().clone()
        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=3,
                per_device_eval_batch_size=3,
                report_to="none",
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert not torch.equal(state_head_before, model.state_head.input_projection.weight)
        assert not torch.equal(action_head_before, model.action_head.input_projection.weight)
        for name, parameter in model.base_model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        prediction = trainer.predict(dataset)
        assert prediction.predictions.shape == (3, 3)
        np.testing.assert_array_equal(prediction.predictions == -100, ~batch["option_mask"].numpy())
        np.testing.assert_allclose(
            np.where(prediction.predictions == -100, 0, prediction.predictions).sum(-1), 1, atol=1e-6
        )


@pytest.mark.skipif(not HAS_JEV_PREPARATION, reason="Requires Transformers #51382")
class TestDecisionTrainerJev(TrlTestCase):
    def test_training_and_predictions(self, tokenizer):
        from transformers import JevConfig

        encoder = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
        model = JevForDecision(JevConfig(text_config=encoder.config, hidden_size=encoder.config.hidden_size))
        model.model = encoder
        collator = DataCollatorForDecision(
            tokenizer, model.decision_config, prepare_inputs_for_decision=model.prepare_inputs_for_decision
        )
        rows = [
            (
                "An invoice",
                {"type": "choice", "criteria": {"billing": None, "delivery": None}},
                {"target_probs": [0, 0.75, 0.25]},
            ),
            ("An urgent invoice", {"type": "noul", "instructions": "Is this urgent?"}, {"p_true": 0.8}),
            ("A good answer", {"type": "score", "criteria": ["bad", "good", "great"]}, {"score": 1.5}),
        ]
        dataset = Dataset.from_list(
            [
                {
                    "request": json.dumps({"state": state, "questions": {"decision": question}}),
                    "target_probs": None,
                    "p_true": None,
                    "score": None,
                    **target,
                }
                for state, question, target in rows
            ]
        )
        batch = collator([dataset[0]])
        with torch.no_grad():
            expected = model(**{key: value for key, value in batch.items() if key != "is_choice"}).loss.item()
        trainer = DecisionTrainer(
            model,
            args=DecisionConfig(
                output_dir=self.tmp_dir,
                use_cpu=True,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                report_to="none",
                **{CHECKPOINTING_ARG: False},
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        assert trainer.evaluate(dataset.select([0]))["eval_loss"] == pytest.approx(expected)
        before = model.proj.weight.detach().clone()
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        assert not torch.equal(before, model.proj.weight)
        prediction = trainer.predict(dataset)
        assert prediction.predictions.shape == (3, 3)
        np.testing.assert_allclose(
            np.where(prediction.predictions == -100, 0, prediction.predictions).sum(-1), 1, atol=1e-6
        )
