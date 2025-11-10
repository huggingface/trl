# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import (
    CPOConfig,
    CPOTrainer,
    DPOConfig,
    DPOTrainer,
    FDivergenceType,
    KTOConfig,
    KTOTrainer,
    NashMDConfig,
    NashMDTrainer,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    ORPOConfig,
    ORPOTrainer,
    RewardConfig,
    RewardTrainer,
    SFTConfig,
    SFTTrainer,
)

from .testing_utils import TrlTestCase


class TestTrainerArg(TrlTestCase):
    def test_cpo(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = CPOConfig(
            self.tmp_dir,
            max_length=256,
            max_prompt_length=64,
            max_completion_length=64,
            beta=0.5,
            label_smoothing=0.5,
            loss_type="hinge",
            disable_dropout=False,
            cpo_alpha=0.5,
            simpo_gamma=0.2,
            label_pad_token_id=-99,
            padding_value=-99,
            truncation_mode="keep_start",
            # generate_during_eval=True, # ignore this one, it requires wandb
            is_encoder_decoder=True,
            model_init_kwargs={"trust_remote_code": True},
            dataset_num_proc=4,
        )
        trainer = CPOTrainer(model=model_id, args=training_args, train_dataset=dataset, processing_class=tokenizer)
        assert trainer.args.max_length == 256
        assert trainer.args.max_prompt_length == 64
        assert trainer.args.max_completion_length == 64
        assert trainer.args.beta == 0.5
        assert trainer.args.label_smoothing == 0.5
        assert trainer.args.loss_type == "hinge"
        assert not trainer.args.disable_dropout
        assert trainer.args.cpo_alpha == 0.5
        assert trainer.args.simpo_gamma == 0.2
        assert trainer.args.label_pad_token_id == -99
        assert trainer.args.padding_value == -99
        assert trainer.args.truncation_mode == "keep_start"
        # self.assertEqual(trainer.args.generate_during_eval, True)
        assert trainer.args.is_encoder_decoder
        assert trainer.args.model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.dataset_num_proc == 4

    def test_dpo(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = DPOConfig(
            self.tmp_dir,
            beta=0.5,
            label_smoothing=0.5,
            loss_type="hinge",
            label_pad_token_id=-99,
            pad_token=".",
            truncation_mode="keep_start",
            max_length=256,
            max_prompt_length=64,
            max_completion_length=64,
            disable_dropout=False,
            # generate_during_eval=True, # ignore this one, it requires wandb
            precompute_ref_log_probs=True,
            dataset_num_proc=4,
            model_init_kwargs={"trust_remote_code": True},
            ref_model_init_kwargs={"trust_remote_code": True},
            model_adapter_name="dummy_adapter",
            ref_adapter_name="dummy_adapter",
            reference_free=True,
            force_use_ref_model=True,
            f_divergence_type="js_divergence",
            f_alpha_divergence_coef=0.5,
            # sync_ref_model=True, # cannot be True when precompute_ref_log_probs=True. Don't test this.
            ref_model_mixup_alpha=0.5,
            ref_model_sync_steps=32,
            rpo_alpha=0.5,
            discopop_tau=0.1,
        )
        trainer = DPOTrainer(
            model=model_id,
            ref_model=model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        assert trainer.args.beta == 0.5
        assert trainer.args.label_smoothing == 0.5
        assert trainer.args.loss_type == "hinge"
        assert trainer.args.label_pad_token_id == -99
        assert trainer.args.pad_token == "."
        assert trainer.args.truncation_mode == "keep_start"
        assert trainer.args.max_length == 256
        assert trainer.args.max_prompt_length == 64
        assert trainer.args.max_completion_length == 64
        assert not trainer.args.disable_dropout
        # self.assertEqual(trainer.args.generate_during_eval, True)
        assert trainer.args.precompute_ref_log_probs
        assert trainer.args.dataset_num_proc == 4
        assert trainer.args.model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.ref_model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.model_adapter_name == "dummy_adapter"
        assert trainer.args.ref_adapter_name == "dummy_adapter"
        assert trainer.args.reference_free
        assert trainer.args.force_use_ref_model
        assert trainer.args.f_divergence_type == FDivergenceType.JS_DIVERGENCE
        assert trainer.args.f_alpha_divergence_coef == 0.5
        # self.assertEqual(trainer.args.sync_ref_model, True)
        assert trainer.args.ref_model_mixup_alpha == 0.5
        assert trainer.args.ref_model_sync_steps == 32
        assert trainer.args.rpo_alpha == 0.5
        assert trainer.args.discopop_tau == 0.1

    def test_kto(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        training_args = KTOConfig(
            self.tmp_dir,
            max_length=256,
            max_prompt_length=64,
            max_completion_length=64,
            beta=0.5,
            desirable_weight=0.5,
            undesirable_weight=0.5,
            label_pad_token_id=-99,
            padding_value=-99,
            truncation_mode="keep_start",
            # generate_during_eval=True, # ignore this one, it requires wandb
            is_encoder_decoder=True,
            precompute_ref_log_probs=True,
            model_init_kwargs={"trust_remote_code": True},
            ref_model_init_kwargs={"trust_remote_code": True},
            dataset_num_proc=4,
        )
        trainer = KTOTrainer(
            model=model_id,
            ref_model=model_id,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        assert trainer.args.max_length == 256
        assert trainer.args.max_prompt_length == 64
        assert trainer.args.max_completion_length == 64
        assert trainer.args.beta == 0.5
        assert trainer.args.desirable_weight == 0.5
        assert trainer.args.undesirable_weight == 0.5
        assert trainer.args.label_pad_token_id == -99
        assert trainer.args.padding_value == -99
        assert trainer.args.truncation_mode == "keep_start"
        # self.assertEqual(trainer.args.generate_during_eval, True)
        assert trainer.args.is_encoder_decoder
        assert trainer.args.precompute_ref_log_probs
        assert trainer.args.model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.ref_model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.dataset_num_proc == 4

    @pytest.mark.parametrize("mixtures_coef_list", [False, True])
    def test_nash_md(self, mixtures_coef_list):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        reward_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = NashMDConfig(
            self.tmp_dir,
            mixture_coef=0.5 if not mixtures_coef_list else [0.5, 0.6],
        )
        trainer = NashMDTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,
            reward_funcs=reward_model,
            train_dataset=dataset,
        )
        assert trainer.args.mixture_coef == (0.5 if not mixtures_coef_list else [0.5, 0.6])

    @pytest.mark.parametrize("beta_list", [False, True])
    def test_online_dpo(self, beta_list):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        reward_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = OnlineDPOConfig(
            self.tmp_dir,
            max_new_tokens=42,
            temperature=0.5,
            missing_eos_penalty=0.33,
            beta=0.6 if not beta_list else [0.6, 0.7],
            loss_type="hinge",
        )
        trainer = OnlineDPOTrainer(
            model=model,
            ref_model=ref_model,
            reward_funcs=reward_model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_processing_classes=tokenizer,
        )
        assert trainer.args.max_new_tokens == 42
        assert trainer.args.temperature == 0.5
        assert trainer.args.missing_eos_penalty == 0.33
        assert trainer.args.beta == (0.6 if not beta_list else [0.6, 0.7])
        assert trainer.args.loss_type == "hinge"

    def test_orpo(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = ORPOConfig(
            self.tmp_dir,
            max_length=256,
            max_prompt_length=64,
            max_completion_length=64,
            beta=0.5,
            disable_dropout=False,
            label_pad_token_id=-99,
            padding_value=-99,
            truncation_mode="keep_start",
            # generate_during_eval=True, # ignore this one, it requires wandb
            is_encoder_decoder=True,
            model_init_kwargs={"trust_remote_code": True},
            dataset_num_proc=4,
        )
        trainer = ORPOTrainer(model=model_id, args=training_args, train_dataset=dataset, processing_class=tokenizer)
        assert trainer.args.max_length == 256
        assert trainer.args.max_prompt_length == 64
        assert trainer.args.max_completion_length == 64
        assert trainer.args.beta == 0.5
        assert not trainer.args.disable_dropout
        assert trainer.args.label_pad_token_id == -99

    def test_reward(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")
        training_args = RewardConfig(
            self.tmp_dir,
            max_length=256,
            dataset_num_proc=4,
            center_rewards_coefficient=0.1,
        )
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        assert trainer.args.max_length == 256
        assert trainer.args.dataset_num_proc == 4
        assert trainer.args.center_rewards_coefficient == 0.1

    def test_sft(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling", split="train")
        training_args = SFTConfig(
            self.tmp_dir,
            dataset_text_field="dummy_text_field",
            packing=True,
            max_length=256,
            dataset_num_proc=4,
            neftune_noise_alpha=0.1,
            model_init_kwargs={"trust_remote_code": True},
            dataset_kwargs={"append_concat_token": True, "skip_prepare_dataset": True},
            eval_packing=True,
        )
        trainer = SFTTrainer(model_id, args=training_args, train_dataset=dataset)
        assert trainer.args.dataset_text_field == "dummy_text_field"
        assert trainer.args.packing
        assert trainer.args.max_length == 256
        assert trainer.args.dataset_num_proc == 4
        assert trainer.args.neftune_noise_alpha == 0.1
        assert trainer.args.model_init_kwargs == {"trust_remote_code": True}
        assert "append_concat_token" in trainer.args.dataset_kwargs
        assert trainer.args.dataset_kwargs["append_concat_token"]
        assert trainer.args.eval_packing
