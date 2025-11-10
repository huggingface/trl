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

from trl.experimental.bco import BCOConfig, BCOTrainer
from trl.experimental.xpo import XPOConfig, XPOTrainer

from ..testing_utils import TrlTestCase, require_sklearn


class TestTrainerArg(TrlTestCase):
    @require_sklearn
    def test_bco(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = load_dataset("trl-internal-testing/zen", "standard_unpaired_preference", split="train")
        training_args = BCOConfig(
            self.tmp_dir,
            max_length=256,
            max_prompt_length=64,
            max_completion_length=64,
            beta=0.5,
            label_pad_token_id=-99,
            padding_value=-99,
            truncation_mode="keep_start",
            # generate_during_eval=True, # ignore this one, it requires wandb
            is_encoder_decoder=True,
            precompute_ref_log_probs=True,
            model_init_kwargs={"trust_remote_code": True},
            ref_model_init_kwargs={"trust_remote_code": True},
            dataset_num_proc=4,
            prompt_sample_size=512,
            min_density_ratio=0.2,
            max_density_ratio=20.0,
        )
        trainer = BCOTrainer(
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
        assert trainer.args.label_pad_token_id == -99
        assert trainer.args.padding_value == -99
        assert trainer.args.truncation_mode == "keep_start"
        # self.assertEqual(trainer.args.generate_during_eval, True)
        assert trainer.args.is_encoder_decoder
        assert trainer.args.precompute_ref_log_probs
        assert trainer.args.model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.ref_model_init_kwargs == {"trust_remote_code": True}
        assert trainer.args.dataset_num_proc == 4
        assert trainer.args.prompt_sample_size == 512
        assert trainer.args.min_density_ratio == 0.2
        assert trainer.args.max_density_ratio == 20.0

    @pytest.mark.parametrize("alpha_list", [False, True])
    def test_xpo(self, alpha_list):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        ref_model = AutoModelForCausalLM.from_pretrained(model_id)
        reward_model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = XPOConfig(
            self.tmp_dir,
            alpha=0.5 if not alpha_list else [0.5, 0.6],
        )
        trainer = XPOTrainer(
            args=training_args,
            processing_class=tokenizer,
            model=model,
            ref_model=ref_model,
            reward_funcs=reward_model,
            train_dataset=dataset,
        )
        assert trainer.args.alpha == (0.5 if not alpha_list else [0.5, 0.6])
