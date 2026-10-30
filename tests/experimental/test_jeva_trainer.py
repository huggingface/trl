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

import copy
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, set_seed

from trl.experimental.jeva import JevaConfig, JevaModel, JevaTrainer
from trl.experimental.jeva.data import DataCollatorForJeva

from ..testing_utils import TrlTestCase, require_peft


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def dataset():
    return Dataset.from_dict(
        {
            "state": [
                "A billing question",
                "No service applies",
                "The answer is mostly true",
                "A good answer",
                "A refund",
            ],
            "question": [
                "Choose a team",
                "Choose a service",
                "Is the answer true?",
                "Rate the answer",
                "Choose a team",
            ],
            "decision_type": ["choice", "choice", "noul", "score", "choice"],
            "options": [
                ["billing", "technical"],
                ["shipping"],
                ["false", "true"],
                ["bad", "good", "great"],
                ["yes", "no"],
            ],
            "allow_null": [False, True, False, False, True],
            "target_probs": [[0, 1, 0], [1, 0], [0, 0.2, 0.8], [0, 0, 0.5, 0.5], [0.1, 0.7, 0.2]],
        }
    )


def make_model(model_id=MODEL_ID):
    set_seed(42)
    return JevaModel.from_backbone_pretrained(
        model_id, model_config={"projection_dim": 16, "decision_num_heads": 2}, torch_dtype=torch.float32
    )


class TestJevaTrainer(TrlTestCase):
    @pytest.mark.parametrize("mode", ["head", "full", pytest.param("lora", marks=require_peft)])
    def test_train(self, mode, tokenizer, dataset):
        model = make_model()
        peft_config = None
        if mode == "lora":
            from peft import LoraConfig

            peft_config = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION")
        trainer = JevaTrainer(
            model=model,
            args=JevaConfig(
                output_dir=self.tmp_dir,
                mode=mode,
                use_cpu=True,
                bf16=False,
                gradient_checkpointing=mode != "head",
                max_steps=2,
                per_device_train_batch_size=2,
                option_chunk_size=1,
                report_to="none",
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
        assert any(name.startswith("core.") for name in changed)
        if mode == "head":
            assert not any(name.startswith("backbone.") for name in changed)
            assert not model.backbone.training
        elif mode == "lora":
            assert any("lora_" in name for name in changed)
            assert all("lora_" in name for name in changed if name.startswith("backbone."))
        else:
            assert any(name.startswith("backbone.") for name in changed)

    def test_loss_and_predictions(self, tokenizer, dataset):
        model = make_model().eval()
        batch = DataCollatorForJeva(tokenizer)(list(dataset))
        with torch.no_grad():
            outputs = model(
                **{key: batch[key] for key in ("state_inputs", "option_inputs", "option_counts", "option_mask")}
            )
        log_q = outputs.log_q.masked_fill(~batch["option_mask"], 0)
        expected = -(batch["labels"][:, 1:] * log_q).sum(-1)
        expected += (
            F.binary_cross_entropy_with_logits(outputs.coverage_logit, 1 - batch["labels"][:, 0], reduction="none")
            * batch["allow_null"]
        )

        def compute_metrics(prediction):
            probs, labels = prediction.predictions, prediction.label_ids
            np.testing.assert_allclose(np.where(probs == -100, 0, probs).sum(-1), 1, atol=1e-6)
            assert np.array_equal(probs == -100, labels == -100)
            return {"accuracy": float((probs.argmax(-1) == labels.argmax(-1)).mean())}

        trainer = JevaTrainer(
            model,
            args=JevaConfig(
                output_dir=self.tmp_dir, use_cpu=True, bf16=False, report_to="none", per_device_eval_batch_size=2
            ),
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        prediction = trainer.predict(dataset)
        assert prediction.predictions.shape == (len(dataset), 4)
        assert prediction.metrics["test_loss"] == pytest.approx(expected.mean().item(), rel=1e-5)
        assert "eval_holdout_accuracy" in trainer.evaluate({"holdout": dataset})

    @pytest.mark.parametrize("pooling", ["last", "mean", "cls"])
    def test_permutation_and_chunks(self, pooling, tokenizer, dataset):
        model = make_model().eval()
        model.config.pooling = pooling
        row = dict(dataset[3])
        permuted = {**row, "options": list(reversed(row["options"])), "target_probs": [0, 0.5, 0.5, 0]}
        collator = DataCollatorForJeva(tokenizer)
        batch = collator([row])
        reordered = collator([permuted])
        with torch.no_grad():
            outputs = model(
                **{key: batch[key] for key in ("state_inputs", "option_inputs", "option_counts", "option_mask")}
            )
            actual = model(
                **{key: reordered[key] for key in ("state_inputs", "option_inputs", "option_counts", "option_mask")},
                option_chunk_size=1,
            )
        torch.testing.assert_close(actual.log_q.flip(-1), outputs.log_q)
        torch.testing.assert_close(actual.coverage_logit, outputs.coverage_logit)

    def test_accumulation(self, tokenizer, dataset):
        trained_models = []
        for batch_size, accumulation in ((5, 1), (3, 2)):
            model = make_model()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            trainer = JevaTrainer(
                model,
                args=JevaConfig(
                    output_dir=self.tmp_dir,
                    use_cpu=True,
                    bf16=False,
                    gradient_checkpointing=False,
                    max_steps=1,
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    max_grad_norm=0,
                    lr_scheduler_type="constant",
                    report_to="none",
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                optimizers=(optimizer, None),
            )
            trainer.train()
            trained_models.append(model)
        for a, b in zip(trained_models[0].parameters(), trained_models[1].parameters(), strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    @require_peft
    def test_checkpoint_and_resume(self, tokenizer, dataset):
        from peft import LoraConfig

        args = JevaConfig(
            output_dir=self.tmp_dir,
            mode="lora",
            use_cpu=True,
            bf16=False,
            gradient_checkpointing=False,
            max_steps=2,
            save_steps=1,
            report_to="none",
        )
        trainer = JevaTrainer(
            make_model(),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(r=2, target_modules=["q_proj", "v_proj"], task_type="FEATURE_EXTRACTION"),
        )
        trainer.train()
        checkpoint = os.path.join(self.tmp_dir, "checkpoint-2")
        reloaded = JevaModel.from_pretrained(checkpoint)
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])
        assert os.path.isfile(os.path.join(checkpoint, "optimizer.pt"))
        trainer = JevaTrainer(checkpoint, args=args, train_dataset=dataset, processing_class=tokenizer)
        trainer.train(resume_from_checkpoint=os.path.join(self.tmp_dir, "checkpoint-1"))
        assert trainer.state.global_step == 2
        for name, param in trainer.model.named_parameters():
            torch.testing.assert_close(param, reloaded.state_dict()[name])

    @pytest.mark.parametrize(
        "model_id", ["trl-internal-testing/tiny-LlamaForCausalLM-3.2", "trl-internal-testing/tiny-Qwen3MoeForCausalLM"]
    )
    def test_backbones(self, model_id, dataset):
        model = make_model(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        trainer = JevaTrainer(
            model,
            args=JevaConfig(
                output_dir=self.tmp_dir,
                mode="full",
                use_cpu=True,
                bf16=False,
                gradient_checkpointing=False,
                router_aux_loss_coef=0.1 if "Moe" in model_id else None,
                max_steps=1,
                logging_steps=1,
                report_to="none",
                option_chunk_size=1,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        assert model.get_output_embeddings() is None
        result = trainer.train()
        assert np.isfinite(result.training_loss)
        if "Moe" in model_id:
            assert trainer.state.log_history[0]["aux_loss"] > 0

    def test_import_checkpoint(self, tmp_path, tokenizer, dataset):
        model = make_model().eval()
        model.backbone.save_pretrained(tmp_path / "backbone")
        config = model.config.to_dict()
        config["backbone"] = "backbone"
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"model": config}))
        torch.save(
            {
                "decision_arch": "matrix_stack",
                "context_proj": model.context_proj.state_dict(),
                "option_proj": model.option_proj.state_dict(),
                "core": model.core.state_dict(),
            },
            tmp_path / "head.pt",
        )
        imported = JevaModel.from_jeva_checkpoint(str(tmp_path)).eval()
        batch = DataCollatorForJeva(tokenizer)(list(dataset))
        inputs = {key: batch[key] for key in ("state_inputs", "option_inputs", "option_counts", "option_mask")}
        with torch.no_grad():
            expected, actual = model(**inputs), imported(**inputs)
        torch.testing.assert_close(actual.log_q, expected.log_q)
        torch.testing.assert_close(actual.coverage_logit, expected.coverage_logit)

    @pytest.mark.skipif("TRL_TEST_JEVA_OUTPUT_DIR" not in os.environ, reason="launched by tests/distributed")
    def test_distributed_body(self):
        training_args = JevaConfig(
            output_dir=os.environ["TRL_TEST_JEVA_OUTPUT_DIR"],
            mode="full",
            use_cpu=os.environ.get("TRL_TEST_JEVA_USE_CPU") == "1",
            bf16=False,
            gradient_checkpointing=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4 if os.environ.get("TRL_TEST_JEVA_CP") == "1" else 2,
            accelerator_config={"even_batches": False},
            option_chunk_size=2,
            max_steps=1,
            max_grad_norm=0,
            lr_scheduler_type="constant",
            save_strategy="no",
            report_to="none",
            ddp_find_unused_parameters=False,
        )
        set_seed(42)
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        model = JevaModel.from_backbone_pretrained(
            model_id,
            model_config={"projection_dim": 16, "decision_num_heads": 2},
            attn_implementation="sdpa",
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = Dataset.from_list(
            [
                {
                    "state": f"Request {i}",
                    "question": "Which option applies?",
                    "options": [f"Option {j}" for j in range(count)],
                    "target_probs": [0.2] + [0.8 / count] * count,
                    "allow_null": True,
                }
                for i, count in enumerate([1, 2, 3, 4, 2, 1, 3])
            ]
        )
        reference = copy.deepcopy(model).to(training_args.device).eval()
        batch = DataCollatorForJeva(tokenizer)(list(dataset))
        batch = {
            key: {name: value.to(training_args.device) for name, value in item.items()}
            if isinstance(item, dict)
            else item.to(training_args.device)
            for key, item in batch.items()
        }
        outputs = reference(
            **{key: batch[key] for key in ("state_inputs", "option_inputs", "option_counts", "option_mask")}
        )
        loss = -(batch["labels"][:, 1:] * outputs.log_q.masked_fill(~batch["option_mask"], 0)).sum(-1)
        loss += F.binary_cross_entropy_with_logits(outputs.coverage_logit, 1 - batch["labels"][:, 0], reduction="none")
        loss.mean().backward()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()

        trainer = JevaTrainer(
            model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": 0.01}),
        )
        trainer.train()
        state = trainer.accelerator.get_state_dict(trainer.model_wrapped)
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(state[name].to(parameter), parameter, rtol=1e-4, atol=1e-5)
        if os.environ.get("TRL_TEST_JEVA_CP") == "1":
            assert np.isfinite(trainer.evaluate(dataset.select(range(4)))["eval_loss"])
        else:
            prediction = trainer.predict(dataset.select(range(4)))
            if trainer.is_world_process_zero():
                assert prediction.predictions.shape == (4, 5)
                assert np.isfinite(prediction.metrics["test_loss"])
        trainer.save_model()
        trainer.accelerator.wait_for_everyone()
        if trainer.is_world_process_zero():
            reloaded = JevaModel.from_pretrained(os.environ["TRL_TEST_JEVA_OUTPUT_DIR"])
            for name, parameter in reloaded.named_parameters():
                torch.testing.assert_close(parameter, state[name].cpu())


@pytest.mark.parametrize(
    "row, target",
    [
        ({"target_index": 1, "options": ["a", "b"]}, [0, 0, 1]),
        ({"target_index": None, "options": ["a"], "allow_null": True}, [1, 0]),
        ({"decision_type": "noul", "p_true": 0.8}, [0, 0.2, 0.8]),
        ({"decision_type": "score", "options": ["bad", "good"], "values": [1, 3], "score": 2}, [0, 0.5, 0.5]),
    ],
)
def test_decision_labels(tokenizer, row, target):
    batch = DataCollatorForJeva(tokenizer)([{"state": "A state", "question": "Decide", **row}])
    torch.testing.assert_close(batch["labels"], torch.tensor([target], dtype=torch.float32))


def test_truncation(tokenizer):
    row = {"state": "A long state", "question": "Decide", "options": ["a", "b"], "target_index": 0}
    with pytest.raises(ValueError, match="token limit"):
        DataCollatorForJeva(tokenizer, max_state_length=2)([row])
    batch = DataCollatorForJeva(tokenizer, max_state_length=2, truncation="truncate")([row])
    assert batch["state_inputs"]["input_ids"].shape[1] == 2
