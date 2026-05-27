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

import logging
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, TrainerControl, TrainerState, TrainingArguments
from transformers.utils import is_peft_available

from trl.experimental.sdft import SDFTConfig, SDFTTrainer
from trl.experimental.sdpo import SDPOTrainer
from trl.experimental.self_distillation.generation import Generation, GenerationOutput
from trl.experimental.self_distillation.loss_utils import (
    aggregate_loss,
    apply_importance_sampling_clipping,
    compute_full_logit_self_distillation_loss,
    compute_sampled_token_self_distillation_loss,
    compute_topk_self_distillation_loss,
)

from ..testing_utils import TrlTestCase


if is_peft_available():
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    from trl.experimental.self_distillation.teacher_sync import PEFTAdapterEMACallback


class TestSelfDistillationTrainerBehavior(TrlTestCase):
    @staticmethod
    def _make_loss_test_trainer(**args_overrides):
        trainer = object.__new__(SDFTTrainer)
        args = {
            "distillation_mode": "sampled_token",
            "distillation_topk": None,
            "distillation_alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_is_clip": None,
        }
        args.update(args_overrides)
        trainer.args = SimpleNamespace(**args)
        trainer.loss_type = "dapo"
        trainer.max_completion_length = 2
        trainer.accelerator = SimpleNamespace(gather=lambda tensor: tensor)
        trainer._metrics = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }
        trainer._name = "SDFT"
        return trainer

    def test_full_logit_loss_matches_forward_kl(self):
        student_probs = torch.tensor([[[0.8, 0.2]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)

        loss = compute_full_logit_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            distillation_alpha=0.0,
        )

        expected_loss = teacher_probs[0, 0, 0] * (
            teacher_probs[0, 0, 0].log() - student_probs[0, 0, 0].log()
        ) + teacher_probs[0, 0, 1] * (teacher_probs[0, 0, 1].log() - student_probs[0, 0, 1].log())
        torch.testing.assert_close(loss, expected_loss.reshape(1, 1))

    def test_sampled_token_loss_uses_selected_completion_ids(self):
        student_probs = torch.tensor([[[0.1, 0.9], [0.7, 0.3]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.4, 0.6], [0.2, 0.8]]], dtype=torch.float32)
        completion_ids = torch.tensor([[1, 0]])

        loss = compute_sampled_token_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            completion_ids,
            distillation_alpha=1.0,
        )

        expected_student_logps = torch.tensor([[0.9, 0.7]], dtype=torch.float32).log()
        expected_teacher_logps = torch.tensor([[0.6, 0.2]], dtype=torch.float32).log()
        expected_loss = (expected_student_logps - expected_teacher_logps) * expected_student_logps
        torch.testing.assert_close(loss, expected_loss)

    def test_topk_loss_renormalizes_selected_student_support(self):
        student_probs = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.float32)

        loss = compute_topk_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            distillation_topk=2,
            distillation_alpha=0.0,
            distillation_add_tail=False,
        )

        student_topk = torch.tensor([0.5, 0.3], dtype=torch.float32)
        student_topk = student_topk / student_topk.sum()
        teacher_topk = torch.tensor([0.2, 0.6], dtype=torch.float32)
        teacher_topk = teacher_topk / teacher_topk.sum()
        expected_loss = (teacher_topk * (teacher_topk.log() - student_topk.log())).sum()
        torch.testing.assert_close(loss, expected_loss.reshape(1, 1))

    def test_topk_loss_can_include_tail_bucket(self):
        student_probs = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.float32)

        loss = compute_topk_self_distillation_loss(
            student_probs.log(),
            teacher_probs.log(),
            distillation_topk=2,
            distillation_alpha=0.0,
            distillation_add_tail=True,
        )

        student_with_tail = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
        teacher_with_tail = torch.tensor([0.2, 0.6, 0.2], dtype=torch.float32)
        expected_loss = (teacher_with_tail * (teacher_with_tail.log() - student_with_tail.log())).sum()
        torch.testing.assert_close(loss, expected_loss.reshape(1, 1))

    def test_aggregate_loss_masks_response_tokens(self):
        per_token_loss = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
        response_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        loss = aggregate_loss(per_token_loss, response_mask, loss_type="dapo", max_completion_length=2)

        torch.testing.assert_close(loss, torch.tensor(3.0))

    def test_aggregate_loss_rejects_unsupported_loss_modes(self):
        per_token_loss = torch.tensor([[1.0]])
        response_mask = torch.tensor([[1.0]])

        with pytest.raises(ValueError, match="Unsupported loss_type: luspo"):
            aggregate_loss(per_token_loss, response_mask, loss_type="luspo", max_completion_length=1)

    def test_importance_sampling_clipping_caps_token_ratio(self):
        per_token_loss = torch.tensor([[1.0, 2.0]])
        student_log_probs = torch.tensor([[0.4, 0.3]], dtype=torch.float32).log()
        old_log_probs = torch.tensor([[0.1, 0.2]], dtype=torch.float32).log()

        loss = apply_importance_sampling_clipping(
            per_token_loss,
            student_log_probs,
            old_log_probs,
            clip_coeff=2.0,
        )

        torch.testing.assert_close(loss, torch.tensor([[2.0, 3.0]]))

    @pytest.mark.parametrize("trainer_cls", [SDFTTrainer, SDPOTrainer])
    def test_trainer_syncs_vllm_weights_before_generation_on_new_step(self, trainer_cls):
        class FakeGenerationEngine:
            def __init__(self):
                self.sync_weights_call_count = 0
                self.generate_calls = []

            def sync_weights(self):
                self.sync_weights_call_count += 1

            def generate(self, prompt_ids, *, num_generations):
                self.generate_calls.append({"prompt_ids": prompt_ids, "num_generations": num_generations})
                return GenerationOutput(prompt_ids=prompt_ids, completion_ids=[[31], [32]])

        trainer = object.__new__(trainer_cls)
        trainer.args = SimpleNamespace(use_vllm=True)
        trainer.model = SimpleNamespace(training=True)
        trainer.state = SimpleNamespace(global_step=4)
        trainer._last_loaded_step = 3
        trainer.num_generations = 2
        trainer.num_generations_eval = 3
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer._tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99)
        trainer.mask_truncated_completions = False
        trainer.generation_engine = FakeGenerationEngine()
        trainer._tokenize_prompts = lambda prompts: [[11, 12] for _ in prompts]
        trainer._dispatch_self_distillation_callback = lambda event_name, **payload: None
        trainer._compute_rollout_logps = lambda **kwargs: None

        batch = trainer.sample_rollouts([{"prompt": "a"}, {"prompt": "b"}])

        assert trainer.generation_engine.sync_weights_call_count == 1
        assert trainer._last_loaded_step == 4
        assert trainer.generation_engine.generate_calls == [{"prompt_ids": [[11, 12], [11, 12]], "num_generations": 2}]
        torch.testing.assert_close(batch["completion_ids"], torch.tensor([[31], [32]]))

    def test_generation_engine_transformers_trims_after_first_eos(self, monkeypatch):
        class FakeModel:
            def generate(self, input_ids, attention_mask, generation_config):
                del attention_mask, generation_config
                completions = torch.tensor([[7, 2, 9], [8, 9, 10]], dtype=torch.long)
                return torch.cat([input_ids, completions], dim=1)

        @contextmanager
        def fake_unwrap_model_for_generation(*args, **kwargs):
            del args, kwargs
            yield FakeModel()

        import trl.experimental.self_distillation.generation as generation_module

        monkeypatch.setattr(generation_module, "unwrap_model_for_generation", fake_unwrap_model_for_generation)

        generator = object.__new__(Generation)
        generator.use_vllm = False
        generator.accelerator = SimpleNamespace(device=torch.device("cpu"))
        generator.model_wrapped = object()
        generator.is_fsdp_enabled = False
        generator.args = SimpleNamespace(ds3_gather_for_generation=False)
        generator.generation_kwargs = {}
        generator.generation_config = SimpleNamespace()
        generator._tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)

        output = generator.generate([[11, 12, 13], [21]], num_generations=1)

        assert output.prompt_ids == [[11, 12, 13], [21]]
        assert output.completion_ids == [[7, 2], [8, 9, 10]]

    def test_sdft_response_mask_keeps_loss_token_skip_local(self):
        trainer = object.__new__(SDFTTrainer)
        trainer.num_loss_tokens_to_skip = 2

        response_mask = trainer._build_self_distillation_response_mask(
            torch.tensor([[1, 1, 1, 0]]),
            torch.tensor([1]),
        )

        torch.testing.assert_close(response_mask, torch.tensor([[0, 0, 1, 0]]))

    def test_teacher_model_kind_live_uses_student_model(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind="live",
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model is trainer.model

    @pytest.mark.skipif(not is_peft_available(), reason="PEFT is required for this test")
    def test_warns_when_initial_student_already_has_a_peft_adapter(self, caplog):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind="base",
            report_to="none",
        )
        model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model = get_peft_model(
            model,
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        with caplog.at_level(logging.WARNING, logger="trl.experimental.sdft.sdft_trainer"):
            SDFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )

        assert "already contains a PEFT adapter" in caplog.text
        assert "`teacher_model_kind='base'` may refer to the underlying base weights" in caplog.text

    @pytest.mark.skipif(not is_peft_available(), reason="PEFT is required for this test")
    def test_peft_adapter_ema_callback_updates_teacher_adapter(self):
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            device_map="cpu",
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
                r=8,
            ),
            adapter_name="default",
        )

        update_rate = 0.5
        callback = PEFTAdapterEMACallback(
            model=model,
            teacher_adapter_name="teacher",
            update_rate=update_rate,
            sync_steps=1,
        )
        args = TrainingArguments(output_dir=self.tmp_dir, report_to="none")
        state = TrainerState(global_step=0)
        control = TrainerControl()

        callback.on_train_begin(args, state, control)

        assert "teacher" in model.peft_config
        assert callback.shadow_weights is not None
        teacher_state = get_peft_model_state_dict(model, adapter_name="teacher")
        for key, param in teacher_state.items():
            assert torch.all(param == 0), f"Teacher param {key} should be zero-initialized"

        student_state = {
            key: value.clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()
        }
        assert set(callback.shadow_weights.keys()) == set(student_state.keys())

        state.global_step = 1
        callback.on_step_end(args, state, control)

        for key in callback.shadow_weights:
            expected = update_rate * student_state[key]
            torch.testing.assert_close(callback.shadow_weights[key], expected)

        teacher_state = get_peft_model_state_dict(model, adapter_name="teacher")
        for key in teacher_state:
            torch.testing.assert_close(teacher_state[key].float(), callback.shadow_weights[key])

    @pytest.mark.parametrize("teacher_model_kind", ["base", "ema"])
    def test_teacher_model_kind_base_and_ema_use_frozen_teacher_copy(self, teacher_model_kind):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        training_args = SDFTConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            teacher_model_kind=teacher_model_kind,
            report_to="none",
        )

        trainer = SDFTTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        assert trainer.teacher_model is not trainer.model
        assert trainer.teacher_model.training is False

        student_param = next(trainer.model.parameters())
        teacher_param = next(trainer.teacher_model.parameters())
        assert teacher_param.requires_grad is False
        assert teacher_param.data_ptr() != student_param.data_ptr()

    def test_compute_self_distillation_loss_ignores_masked_completion_tokens(self):
        trainer = self._make_loss_test_trainer(
            distillation_mode="full_logits",
            distillation_alpha=0.0,
        )
        model = SimpleNamespace(training=True)

        # Token 0 is active and has a known non-zero divergence.
        # Token 1 is intentionally very different but masked out, so it must not affect the loss.
        student_probs = torch.tensor([[[0.8, 0.2], [0.01, 0.99]]], dtype=torch.float32)
        teacher_probs = torch.tensor([[[0.5, 0.5], [0.99, 0.01]]], dtype=torch.float32)
        distillation_logits = SimpleNamespace(
            completion_ids=torch.tensor([[0, 1]], dtype=torch.long),
            completion_mask=torch.tensor([[1, 1]], dtype=torch.long),
            response_mask=torch.tensor([[1, 0]], dtype=torch.long),
            student_logits=student_probs.log(),
            teacher_logits=teacher_probs.log(),
        )

        loss = trainer._compute_self_distillation_loss(model, {}, distillation_logits)

        expected_active_token_loss = teacher_probs[0, 0, 0] * (
            teacher_probs[0, 0, 0].log() - student_probs[0, 0, 0].log()
        ) + teacher_probs[0, 0, 1] * (teacher_probs[0, 0, 1].log() - student_probs[0, 0, 1].log())
        torch.testing.assert_close(loss, expected_active_token_loss)
        torch.testing.assert_close(
            torch.tensor(trainer._metrics["train"]["self_distillation/distillation_loss"]),
            expected_active_token_loss.unsqueeze(0),
        )

    def test_compute_self_distillation_loss_applies_importance_sampling_clip(self):
        trainer = self._make_loss_test_trainer(distillation_is_clip=2.0)
        model = SimpleNamespace(training=True)

        student_token_probs = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
        teacher_token_probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        old_token_probs = torch.tensor([[0.05, 0.4]], dtype=torch.float32)
        clip_coeff = trainer.args.distillation_is_clip

        distillation_logits = SimpleNamespace(
            completion_ids=torch.tensor([[0, 1]], dtype=torch.long),
            completion_mask=torch.tensor([[1, 1]], dtype=torch.long),
            response_mask=torch.tensor([[1, 1]], dtype=torch.long),
            student_logits=torch.log(torch.tensor([[[0.2, 0.8], [0.6, 0.4]]], dtype=torch.float32)),
            teacher_logits=torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32)),
        )

        loss = trainer._compute_self_distillation_loss(
            model,
            {"old_per_token_logps": old_token_probs.log()},
            distillation_logits,
        )

        raw_per_token_loss = (student_token_probs.log() - teacher_token_probs.log()) * student_token_probs.log()
        clipped_ratio = torch.minimum(
            student_token_probs / old_token_probs, torch.full_like(student_token_probs, clip_coeff)
        )
        expected_loss = (raw_per_token_loss * clipped_ratio).mean()

        torch.testing.assert_close(loss, expected_loss)
        torch.testing.assert_close(
            torch.tensor(trainer._metrics["train"]["self_distillation/distillation_loss"]),
            expected_loss.unsqueeze(0),
        )
