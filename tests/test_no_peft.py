# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .testing_utils import is_peft_available, require_peft


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, response_data):
        self.query_data = query_data
        self.response_data = response_data

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        return self.query_data[idx], self.response_data[idx]


EXPECTED_STATS = [
    "objective/kl",
    "objective/kl_dist",
    "objective/logprobs",
    "objective/ref_logprobs",
    "objective/kl_coef",
    "objective/entropy",
    "ppo/mean_non_score_reward",
    "ppo/loss/policy",
    "ppo/loss/value",
    "ppo/loss/total",
    "ppo/policy/entropy",
    "ppo/policy/approxkl",
    "ppo/policy/policykl",
    "ppo/policy/clipfrac",
    "ppo/policy/advantages",
    "ppo/policy/advantages_mean",
    "ppo/policy/ratio",
    "ppo/returns/mean",
    "ppo/returns/var",
    "ppo/val/vpred",
    "ppo/val/error",
    "ppo/val/clipfrac",
    "ppo/val/mean",
    "ppo/val/var",
    "ppo/val/var_explained",
    "time/ppo/forward_pass",
    "time/ppo/compute_rewards",
    "time/ppo/optimize_step",
    "time/ppo/calc_stats",
    "time/ppo/total",
    "ppo/learning_rate",
]


@require_peft
class TestPeftDependancy(unittest.TestCase):
    def setUp(self):
        self.causal_lm_model_id = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
        self.seq_to_seq_model_id = "trl-internal-testing/tiny-random-T5ForConditionalGeneration"

        if is_peft_available():
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            causal_lm_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
            self.peft_model = get_peft_model(causal_lm_model, lora_config)

    def test_no_peft(self):
        with patch.dict(sys.modules, {"peft": None}):
            from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead

            # Check that loading a model with `peft` will raise an error
            with self.assertRaises(ModuleNotFoundError):
                import peft  # noqa

            trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.causal_lm_model_id)  # noqa
            trl_seq2seq_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(self.seq_to_seq_model_id)  # noqa

    def test_imports_no_peft(self):
        with patch.dict(sys.modules, {"peft": None}):
            from trl import (  # noqa
                AutoModelForCausalLMWithValueHead,
                AutoModelForSeq2SeqLMWithValueHead,
                PPOConfig,
                PPOTrainer,
                PreTrainedModelWrapper,
            )

    def test_ppo_trainer_no_peft(self):
        with patch.dict(sys.modules, {"peft": None}):
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

            ppo_model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"

            trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_model_id)
            tokenizer = AutoTokenizer.from_pretrained(ppo_model_id)
            tokenizer.pad_token_id = tokenizer.eos_token_id

            ppo_config = PPOConfig(batch_size=2, mini_batch_size=1, log_with=None)

            dummy_dataset = DummyDataset(
                [torch.LongTensor([0, 1, 0, 1, 0, 1]), torch.LongTensor([0, 1, 0, 1, 0, 1])],
                [torch.LongTensor([1, 0, 1, 0, 1, 0]), torch.LongTensor([0, 1, 0, 1, 0, 1])],
            )

            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=trl_model,
                ref_model=None,
                tokenizer=tokenizer,
                dataset=dummy_dataset,
            )
            dummy_dataloader = ppo_trainer.dataloader

            for query_tensor, response_tensor in dummy_dataloader:
                # define a reward for response
                # (this could be any reward such as human feedback or output from another model)
                reward = [torch.tensor(1.0), torch.tensor(0.0)]
                # train model
                train_stats = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
                break

            # check gradients are not None
            for _, param in trl_model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)

            # check expected stats
            for stat in EXPECTED_STATS:
                self.assertIn(stat, train_stats)
