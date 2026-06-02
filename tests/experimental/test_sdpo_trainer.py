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

import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback

from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

from ..testing_utils import TrlTestCase, require_liger_kernel, require_torch_accelerator


class SelfDistillationCaptureCallback(TrainerCallback):
    def __init__(self):
        self.captured_teacher_input_text = None
        self.captured_teacher_input_texts = []
        self.captured_self_distillation_mask = None
        self.captured_teacher_attention_mask = None
        self.captured_completion_mask = None
        self.captured_old_per_token_logps = None

    def on_teacher_context_built(
        self,
        processing_class=None,
        teacher_input_ids=None,
        teacher_attention_mask=None,
        completion_mask=None,
        self_distillation_mask=None,
        **kwargs,
    ):
        if self.captured_teacher_input_text is None and teacher_input_ids is not None:
            self.captured_teacher_input_text = processing_class.decode(teacher_input_ids[0], skip_special_tokens=True)
        if teacher_input_ids is not None:
            self.captured_teacher_input_texts.extend(
                processing_class.decode(ids, skip_special_tokens=True) for ids in teacher_input_ids
            )
        if self.captured_teacher_attention_mask is None and teacher_attention_mask is not None:
            self.captured_teacher_attention_mask = teacher_attention_mask.detach().cpu()
        if self.captured_completion_mask is None and completion_mask is not None:
            self.captured_completion_mask = completion_mask.detach().cpu()
        if self.captured_self_distillation_mask is None and self_distillation_mask is not None:
            self.captured_self_distillation_mask = self_distillation_mask.detach().cpu()

    def on_self_distillation_batch_prepared(self, old_per_token_logps=None, **kwargs):
        if self.captured_old_per_token_logps is None and old_per_token_logps is not None:
            self.captured_old_per_token_logps = old_per_token_logps.detach().cpu()


class TestSDPOTrainer(TrlTestCase):
    def test_train_with_positional_config_argument(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve 2+2."],
                "privileged_context": ["Your earlier answer used the wrong format."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            include_environment_feedback=True,
            max_steps=1,
            report_to="none",
        )

        trainer = SDPOTrainer(
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            training_args,
            dataset,
        )

        trainer.train()

        assert trainer.args.output_dir == self.tmp_dir
        assert trainer.args.include_environment_feedback is True
        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_vllm_config_defaults_match_reference_trainers(self):
        config = SDPOConfig(output_dir=self.tmp_dir)

        assert config.vllm_mode == "colocate"
        assert config.vllm_model_impl == "vllm"

    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            distillation_mode="topk_logits",
            distillation_topk=5,
            distillation_is_clip=None,
            report_to="none",
        )
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if param.sum() != 0:
                assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @require_liger_kernel
    @require_torch_accelerator
    def test_liger_loss_matches_non_liger_loss(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})
        common = dict(
            output_dir=self.tmp_dir,
            report_to="none",
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=3,
            sdpo_policy_loss_mode="distillation_only",
            distillation_mode="full_logits",
            distillation_is_clip=None,
            distillation_weight=0.7,
        )

        ref_trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=SDPOConfig(use_liger_kernel=False, **common),
            train_dataset=dataset,
        )
        liger_trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=SDPOConfig(use_liger_kernel=True, **common),
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
            "self_distillation_mask": torch.tensor([1.0, 0.0], device=device),
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

    def test_train_without_successful_rollouts(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            distillation_is_clip=None,
            report_to="none",
        )

        def zero_reward(**kwargs):
            prompts = kwargs["prompts"]
            return [0.0] * len(prompts)

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_train_populates_old_log_probs_for_distillation_clipping_when_misaligned(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2.", "Solve 3+3."]})

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=3,
            steps_per_generation=2,
            num_generations=2,
            max_completion_length=8,
            max_steps=1,
            report_to="none",
        )

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=lambda **kwargs: [0.0] * len(kwargs["prompts"]),
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_old_per_token_logps is not None

    def test_evaluation_uses_num_generations_eval_for_teacher_grouping(self):
        eval_dataset = Dataset.from_dict({"prompt": ["Alpha prompt", "Beta prompt", "Gamma prompt", "Delta prompt"]})

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=4,
            generation_batch_size=3,
            num_generations=3,
            num_generations_eval=2,
            max_completion_length=8,
            success_reward_threshold=0.5,
            dont_reprompt_on_self_success=False,
            distillation_is_clip=None,
            max_steps=1,
            report_to="none",
        )

        def eval_rewards(**kwargs):
            prompts = kwargs["prompts"]
            if len(prompts) == 4 and prompts.count("Alpha prompt") == 2 and prompts.count("Beta prompt") == 2:
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0] * len(prompts)

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=eval_rewards,
            args=training_args,
            train_dataset=eval_dataset.select(range(1)),
            eval_dataset=eval_dataset,
            callbacks=[capture_callback],
        )

        trainer.evaluate()

        assert capture_callback.captured_teacher_input_texts
        alpha_teachers = [text for text in capture_callback.captured_teacher_input_texts if "Alpha prompt" in text]
        beta_teachers = [text for text in capture_callback.captured_teacher_input_texts if "Beta prompt" in text]
        assert alpha_teachers
        assert beta_teachers
        assert any("Correct solution:" in text for text in alpha_teachers)
        assert all("Correct solution:" not in text for text in beta_teachers)

    def test_teacher_reprompt_preserves_curly_braces_in_solution_and_feedback(self):
        dataset = Dataset.from_dict(
            {
                "prompt": ["Solve f(x) = {x^2}."],
                "privileged_context": ['Feedback: use {"x": 2} as a check.'],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            include_environment_feedback=True,
            success_reward_threshold=0.5,
            dont_reprompt_on_self_success=False,
            max_steps=1,
            report_to="none",
        )

        def reward_with_one_success(**kwargs):
            prompts = kwargs["prompts"]
            return [1.0, 0.0][: len(prompts)]

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_with_one_success,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_input_text is not None
        assert "Solve f(x) = {x^2}." in capture_callback.captured_teacher_input_text
        assert 'Feedback: use {"x": 2} as a check.' in capture_callback.captured_teacher_input_text
        assert "{{" not in capture_callback.captured_teacher_input_text
        assert "}}" not in capture_callback.captured_teacher_input_text

    def test_train_with_conversational_prompts_preserves_context(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Solve 2+2."},
                    ]
                ]
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            distillation_is_clip=None,
            success_reward_threshold=0.5,
            max_steps=1,
            report_to="none",
        )

        def first_only_reward(**kwargs):
            """Only the first sample in each group succeeds — exercises dont_reprompt_on_self_success default."""
            return [1.0, 0.0][: len(kwargs["prompts"])]

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=first_only_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        # With dont_reprompt_on_self_success=True (default), sample 0 skips itself,
        # but sample 1 finds sample 0's success and gets a teacher reprompt.
        assert capture_callback.captured_teacher_input_text is not None
        assert "careful assistant" in capture_callback.captured_teacher_input_text
        assert "Solve 2+2" in capture_callback.captured_teacher_input_text
        assert capture_callback.captured_self_distillation_mask is not None

    def test_train_with_feedback_only_reprompts_teacher(self):
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    [
                        {"role": "system", "content": "You are a careful assistant."},
                        {"role": "user", "content": "Try the puzzle again."},
                    ]
                ],
                "privileged_context": ["Your earlier answer violated the format requirements."],
            }
        )

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            distillation_is_clip=None,
            include_environment_feedback=True,
            max_steps=1,
            report_to="none",
        )

        def zero_reward(**kwargs):
            prompts = kwargs["prompts"]
            return [0.0] * len(prompts)

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_input_text is not None
        assert "format requirements" in capture_callback.captured_teacher_input_text
        assert capture_callback.captured_self_distillation_mask is not None
        assert capture_callback.captured_self_distillation_mask[0].item() == 1.0

    def test_train_warns_when_sdpo_rewards_are_flat(self, caplog):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            diagnostics_warning_interval=2,
            max_steps=2,
            report_to="none",
        )

        def zero_reward(**kwargs):
            return [0.0] * len(kwargs["prompts"])

        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=zero_reward,
            args=training_args,
            train_dataset=dataset,
        )

        with caplog.at_level(logging.WARNING):
            trainer.train()

        assert "Observed flat SDPO rewards across all sampled generations" in caplog.text
        assert "SDPO self-distillation is inactive because no reprompted samples were constructed" in caplog.text

    def test_train_preserves_teacher_completion_attention_mask(self):
        dataset = Dataset.from_dict({"prompt": ["Solve 2+2."]})

        training_args = SDPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=1,
            generation_batch_size=2,
            num_generations=2,
            max_completion_length=8,
            success_reward_threshold=0.5,
            max_steps=1,
            report_to="none",
        )

        def first_only_reward(**kwargs):
            return [1.0, 0.0][: len(kwargs["prompts"])]

        capture_callback = SelfDistillationCaptureCallback()
        trainer = SDPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=first_only_reward,
            args=training_args,
            train_dataset=dataset,
            callbacks=[capture_callback],
        )

        trainer.train()

        assert capture_callback.captured_teacher_attention_mask is not None
        assert capture_callback.captured_completion_mask is not None

        completion_length = capture_callback.captured_completion_mask.shape[1]
        teacher_completion_attention = capture_callback.captured_teacher_attention_mask[0, -completion_length:]
        assert torch.equal(teacher_completion_attention, capture_callback.captured_completion_mask[0])
