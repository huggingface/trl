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

from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset
from transformers import TrainerCallback
from transformers.utils import is_peft_available

from trl.experimental.opsd import OPSDConfig, OPSDTrainer
from trl.experimental.opsd.loss_utils import (
    add_tail_bucket,
    compute_divergence,
    compute_topk_self_distillation_loss,
)

from ..testing_utils import TrlTestCase, require_liger_kernel, require_peft, require_torch_accelerator


if is_peft_available():
    from peft import LoraConfig


class SelfDistillationCaptureCallback(TrainerCallback):
    def __init__(self):
        self.captured_generation_prompts = None
        self.captured_old_per_token_logps = None
        self.captured_prompt_ids = None
        self.generation_batch_build_count = 0

    def on_generation_prompts_selected(self, generation_prompts=None, **kwargs):
        if self.captured_generation_prompts is None and generation_prompts is not None:
            self.captured_generation_prompts = generation_prompts

    def on_self_distillation_batch_prepared(self, old_per_token_logps=None, prompt_ids=None, **kwargs):
        if self.captured_old_per_token_logps is None and old_per_token_logps is not None:
            self.captured_old_per_token_logps = old_per_token_logps.detach().cpu()
        if self.captured_prompt_ids is None and prompt_ids is not None:
            self.captured_prompt_ids = prompt_ids.detach().cpu()

    def on_generation_batch_built(self, **kwargs):
        self.generation_batch_build_count += 1


class TestOPSDTrainer(TrlTestCase):
    @staticmethod
    def _trainable_param_snapshot(model):
        return {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    @staticmethod
    def _assert_any_trainable_param_changed(model, previous_trainable_params):
        assert any(
            not torch.allclose(previous_param, model.get_parameter(name), rtol=1e-12, atol=1e-12)
            for name, previous_param in previous_trainable_params.items()
        )

    def test_train(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_matches_non_liger_loss(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."], "privileged_context": ["Example answer: 4."]})
        common = dict(
            output_dir=self.tmp_dir,
            report_to="none",
            per_device_train_batch_size=1,
            max_completion_length=3,
            num_generations=1,
            distillation_mode="full_logits",
            distillation_is_clip=None,
            distillation_kl_clip=None,
        )

        ref_trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=OPSDConfig(use_liger_kernel=False, **common),
            train_dataset=dataset,
        )
        liger_trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=OPSDConfig(use_liger_kernel=True, **common),
            train_dataset=dataset,
        )

        liger_trainer.model.load_state_dict(ref_trainer.model.state_dict())
        torch.manual_seed(0)
        with torch.no_grad():
            for param in ref_trainer.teacher_model.parameters():
                param.add_(0.5 * torch.randn_like(param))
        liger_trainer.teacher_model.load_state_dict(ref_trainer.teacher_model.state_dict())

        device = next(ref_trainer.model.parameters()).device
        batch = {
            "prompt_ids": torch.tensor([[10, 11], [12, 13]], device=device),
            "prompt_mask": torch.tensor([[1, 1], [1, 1]], device=device),
            "completion_ids": torch.tensor([[14, 15, 16], [17, 18, 19]], device=device),
            "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], device=device),
            "teacher_input_ids": torch.tensor([[20, 21, 22, 14, 15, 16], [23, 24, 25, 17, 18, 19]], device=device),
            "teacher_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], device=device),
        }

        ref_trainer.model.eval()
        liger_trainer.model.eval()
        with torch.no_grad():
            ref_loss = ref_trainer.compute_loss(ref_trainer.model, batch).item()
            liger_loss = liger_trainer.compute_loss(liger_trainer.model, batch).item()

        torch.testing.assert_close(
            torch.tensor(liger_loss),
            torch.tensor(ref_loss),
            rtol=2e-2,
            atol=1e-6,
        )

    def test_train_rejects_none_privileged_context(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": [None],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        with pytest.raises(ValueError, match="`privileged_context` must not be None"):
            trainer.train()

    def test_train_with_chat_template_kwargs(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [{"role": "user", "content": "Solve 2+2."}],
                    [{"role": "user", "content": "Solve 3+3."}],
                ],
                "privileged_context": [
                    "Teacher hint: answer with 4.",
                    "Teacher hint: answer with 6.",
                ],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            chat_template_kwargs={"enable_thinking": False},
            report_to="none",
        )

        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_peft
    def test_train_with_peft_model(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )

        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    @require_peft
    def test_train_with_peft_model_and_ema_teacher_sync(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Name the capital of France."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: Paris.",
                ],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
            teacher_model_kind="ema",
            teacher_update_rate=0.05,
            teacher_sync_steps=1,
            report_to="none",
        )

        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj"],
            ),
        )
        previous_trainable_params = self._trainable_param_snapshot(trainer.model)

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        self._assert_any_trainable_param_changed(trainer.model, previous_trainable_params)

    def test_train_populates_old_log_probs_for_distillation_clipping_when_misaligned(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: 6.",
                ],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is not None

    def test_train_reuses_buffered_generation_batches(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2.", "Solve 3+3."],
                "privileged_context": [
                    "Example answer: 4.",
                    "Example answer: 6.",
                ],
            }
        )

        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            steps_per_generation=2,
            max_completion_length=8,
            max_steps=2,
            num_generations=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.generation_batch_build_count == 1


class TestOPSDConfigValidation(TrlTestCase):
    def test_sampled_token_requires_reverse_kl_alpha(self):
        with pytest.raises(ValueError, match="distillation_alpha=1.0"):
            OPSDConfig(
                output_dir=self.tmp_dir,
                distillation_mode="sampled_token",
                distillation_alpha=0.5,
                distillation_kl_clip=None,
            )

    def test_kl_clip_rejects_sampled_token(self):
        with pytest.raises(ValueError, match="distillation_kl_clip"):
            OPSDConfig(output_dir=self.tmp_dir, distillation_mode="sampled_token", distillation_alpha=1.0)

    def test_kl_clip_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="must be positive"):
            OPSDConfig(output_dir=self.tmp_dir, distillation_kl_clip=0.0)

    def test_teacher_prompt_template_requires_placeholders(self):
        with pytest.raises(ValueError, match="placeholders"):
            OPSDConfig(output_dir=self.tmp_dir, teacher_prompt_template="{prompt} only")


class TestOPSDLoss(TrlTestCase):
    def test_kl_clip_caps_divergence(self):
        torch.manual_seed(0)
        student = torch.log_softmax(torch.randn(2, 3, 11), dim=-1)
        teacher = torch.log_softmax(torch.randn(2, 3, 11), dim=-1)

        unclipped = compute_divergence(student, teacher, 0.0)
        clipped = compute_divergence(student, teacher, 0.0, kl_clip=1e-4)

        # Each vocabulary entry contributes at most the clip value.
        assert (clipped <= 11 * 1e-4 + 1e-6).all()
        assert (clipped <= unclipped + 1e-6).all()

    def test_topk_uses_teacher_support(self):
        # Teacher mass concentrated on tokens {0, 1}; the student prefers tokens {2, 3}.
        teacher_logits = torch.full((1, 1, 6), -10.0)
        teacher_logits[..., 0] = 5.0
        teacher_logits[..., 1] = 4.0
        student_logits = torch.full((1, 1, 6), -10.0)
        student_logits[..., 2] = 5.0
        student_logits[..., 3] = 4.0

        loss = compute_topk_self_distillation_loss(
            student_logits,
            teacher_logits,
            distillation_topk=2,
            distillation_alpha=0.0,
            distillation_add_tail=True,
            distillation_kl_clip=None,
        )

        # Expected: support = the teacher's top-2 indices {0, 1}, both renormalized with a tail bucket.
        topk_teacher = torch.log_softmax(teacher_logits, dim=-1)[..., :2]
        topk_student = torch.log_softmax(student_logits, dim=-1)[..., :2]
        expected = compute_divergence(add_tail_bucket(topk_student), add_tail_bucket(topk_teacher), 0.0)
        torch.testing.assert_close(loss, expected)


class TestOPSDTeacherPrompt(TrlTestCase):
    def test_teacher_prompt_contains_solution_and_transition(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "Solve 2+2."}]],
                "privileged_context": ["The answer is 4."],
            }
        )
        training_args = OPSDConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=1,
            max_completion_length=8,
            max_steps=1,
            num_generations=1,
            report_to="none",
        )

        captured = {}

        class Capture(TrainerCallback):
            def on_self_distillation_batch_prepared(self, teacher_input_ids=None, processing_class=None, **kwargs):
                if "text" not in captured and teacher_input_ids is not None:
                    captured["text"] = processing_class.decode(teacher_input_ids[0], skip_special_tokens=True)

        trainer = OPSDTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            args=training_args,
            train_dataset=dataset,
            callbacks=[Capture()],
        )
        trainer.train()

        assert "Solve 2+2." in captured["text"]
        assert "The answer is 4." in captured["text"]
        assert "Reference Solution Begin" in captured["text"]


class TestOPSDTeacherServerTopk(TrlTestCase):
    """Server-free unit tests for the `use_teacher_server` + `topk_logits` path (no vLLM server needed)."""

    def test_getter_shapes_and_padding_sentinels(self):
        # Sample 0 has a 1-token completion (positions 1-2 are padding); sample 1 is full length 3.
        ragged = {
            "actual_logprobs": [[[-2.3]], [[-1.1], [-0.4], [-3.0]]],
            "logprobs": [[[-2.3, -2.6]], [[-1.1, -1.5], [-0.4, -0.9], [-3.0, -3.4]]],
            "logprob_token_ids": [[[90, 91]], [[90, 91], [9217, 9218], [100, 101]]],
        }
        mock_self = MagicMock()
        mock_self.teacher_client.get_sequence_logprobs = MagicMock(return_value=ragged)
        mock_self.temperature = 1.0
        mock_self.args.distillation_mode = "topk_logits"
        mock_self.args.distillation_topk = 2

        inputs = {
            "teacher_input_ids": torch.tensor([[10, 11, 90, 0, 0], [10, 11, 90, 9217, 100]]),
            "teacher_attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        }
        out = OPSDTrainer._get_teacher_token_logprobs_from_server(mock_self, inputs, logits_to_keep=3)

        assert out["actual_logprobs"].shape == (2, 3)
        assert out["topk_logprobs"].shape == (2, 3, 2)
        assert out["topk_token_ids"].shape == (2, 3, 2)
        assert out["actual_logprobs"][0, 0].item() == pytest.approx(-2.3, rel=1e-5)
        assert out["actual_logprobs"][1, 2].item() == pytest.approx(-3.0, rel=1e-5)
        assert out["actual_logprobs"][0, 1].item() == float("-inf")
        assert out["topk_logprobs"][0, 2, 0].item() == float("-inf")
        assert torch.isfinite(out["actual_logprobs"][1]).all()
        assert mock_self.teacher_client.get_sequence_logprobs.call_args.kwargs["top_logprobs"] == 2

    def test_topk_mask_keeps_forward_and_backward_finite(self):
        # Sample 0 has completion length 1 (positions 1-2 padded with -inf); sample 1 is full length.
        teacher_topk = torch.tensor(
            [
                [[-2.3, -2.6], [float("-inf"), float("-inf")], [float("-inf"), float("-inf")]],
                [[-1.1, -1.5], [-0.4, -0.9], [-3.0, -3.4]],
            ],
            dtype=torch.float32,
        )
        loss_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        # Neutralize -inf at masked positions before the divergence (matches the server topk path).
        keep = loss_mask.bool().unsqueeze(-1)
        zero = torch.zeros((), dtype=teacher_topk.dtype)
        teacher_topk = torch.where(keep, teacher_topk, zero)
        raw_student = torch.randn(2, 3, 2, requires_grad=True)
        student_topk = torch.where(keep, torch.log_softmax(raw_student, dim=-1), zero)

        student_sparse = add_tail_bucket(student_topk)
        teacher_sparse = add_tail_bucket(teacher_topk)
        assert torch.isfinite(student_sparse).all()

        per_token_loss = compute_divergence(student_sparse, teacher_sparse, 0.5)
        loss = (per_token_loss * loss_mask).sum() / loss_mask.sum()
        assert torch.isfinite(loss)

        loss.backward()
        assert torch.isfinite(raw_student.grad).all()
