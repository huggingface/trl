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

from unittest.mock import patch

import pytest
import torch
from datasets import load_dataset

from trl.experimental.klpo import KLPOConfig, KLPOTrainer
from trl.trainer.grpo_trainer import GRPOTrainer

from ..testing_utils import TrlTestCase


class TestKLPOConfig:
    def test_single_rollout_defaults(self):
        # KLPO is a single-rollout method: num_generations=1 must be accepted (GRPOConfig rejects < 2).
        args = KLPOConfig("dummy")
        assert args.num_generations == 1
        assert args.klpo_beta == 0.1
        assert args.mc_samples == 128
        # The inherited reference-model KL stays disabled; KLPO regularizes toward the sampler instead.
        assert args.beta == 0.0
        # Rewards are consumed raw, so group scaling is disabled for single rollouts.
        assert args.scale_rewards == "none"

    def test_invalid_mc_samples(self):
        with pytest.raises(ValueError, match="mc_samples"):
            KLPOConfig("dummy", mc_samples=0)

    def test_invalid_route_and_estimator(self):
        with pytest.raises(ValueError, match="klpo_route"):
            KLPOConfig("dummy", klpo_route="tokens")
        with pytest.raises(ValueError, match="kl_estimator"):
            KLPOConfig("dummy", kl_estimator="montecarlo")

    def test_sequence_mc_needs_two_samples(self):
        # Sequence MC-KL uses leave-one-out residuals, which require M >= 2.
        with pytest.raises(ValueError, match="leave-one-out"):
            KLPOConfig("dummy", klpo_route="sequence", kl_estimator="mc", mc_samples=1)


class TestKLPOTrainer(TrlTestCase):
    def test_train(self):
        # Single rollout per prompt (num_generations=1), the KLPO default.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,  # single rollout per prompt, the KLPO default
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "kl" in trainer.state.log_history[-1]  # the MC-KL estimate is logged

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_multiple_iterations(self):
        # With num_iterations > 1, the batch is reused: the sampler records stay fixed while the policy drifts,
        # exercising the off-policy path (ell != 0).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            num_iterations=2,
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    @pytest.mark.parametrize("config_name", ["standard_prompt_only", "conversational_prompt_only"])
    def test_train_conversational(self, config_name):
        dataset = load_dataset("trl-internal-testing/zen", config_name, split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.parametrize(
        "route, estimator",
        [
            # token + mc is the default and is covered by test_train
            ("token", "topk"),
            ("token", "binary"),
            ("token", "full"),
            ("sequence", "mc"),
            ("sequence", "topk"),
            ("sequence", "binary"),
            ("sequence", "full"),
        ],
    )
    def test_train_routes_and_estimators(self, route, estimator):
        # All route/estimator combinations must train and move the parameters.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=1,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            klpo_route=route,
            kl_estimator=estimator,
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            kl_top_k=2,  # small head so the tail bucket is exercised
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert "kl" in trainer.state.log_history[-1]

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."

    def test_train_num_generations_gt1(self):
        # KLPO does not need groups, but grouped generation must still work (rewards stay raw).
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            mc_samples=4,  # reduce the number of MC draws to speed up the test
            mask_truncated_completions=False,  # the tiny model never emits EOS, so all completions are truncated
            report_to="none",
        )
        trainer = KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."


class TestKLPOLossContract(TrlTestCase):
    """Pins the "average over complete responses" contract of the KLPO loss through the real `_compute_loss`."""

    def _trainer(self, **config_kwargs):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = KLPOConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=4,
            num_generations=1,
            max_completion_length=8,
            mc_samples=4,
            kl_top_k=2,
            report_to="none",
            **config_kwargs,
        )
        return KLPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )

    def _inputs(self, trainer, rewards, completion_mask):
        # Hand-built batch: distinct completions, one terminal reward per row, and a caller-controlled completion
        # mask so specific rows can be fully masked (as a truncated or unscorable completion would be). Token IDs are
        # drawn once for a fixed maximum batch and sliced, so row i is identical across batches of different sizes.
        generator = torch.Generator().manual_seed(0)
        batch_size, prompt_len, completion_len = len(rewards), 3, 5
        vocab = trainer.model.config.vocab_size
        prompt_ids = torch.randint(1, vocab, (4, prompt_len), generator=generator)[:batch_size]
        completion_ids = torch.randint(1, vocab, (4, completion_len), generator=generator)[:batch_size]
        completion_mask = torch.tensor(completion_mask, dtype=torch.long)
        inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": torch.ones_like(prompt_ids),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "raw_rewards": torch.tensor(rewards, dtype=torch.float32),
        }
        if trainer.kl_estimator != "binary":
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
            inputs.update(trainer._record_sampler_data(input_ids, attention_mask, completion_len))
        # Off-policy sampler logps, so the KL term (-beta * ell) is non-zero and a fully masked row would otherwise
        # still move the policy through it.
        inputs["old_per_token_logps"] = torch.full((batch_size, completion_len), -2.0)
        return inputs

    @pytest.mark.parametrize(
        "route, estimator",
        [("token", "mc"), ("token", "binary"), ("token", "topk"), ("sequence", "mc"), ("sequence", "binary")],
    )
    def test_fully_masked_rows_leave_the_denominator(self, route, estimator):
        # Bugbot: a truncated (or unscorable) completion is fully masked, but it must also leave the average over
        # complete responses. Appending two fully masked rows to a batch must not change the loss at all.
        trainer = self._trainer(klpo_route=route, kl_estimator=estimator)
        trainer.model.eval()  # deterministic forward (no dropout) so the two losses are comparable
        full = [1.0] * 5
        empty = [0] * 5

        torch.manual_seed(0)
        loss_complete = trainer._compute_loss(trainer.model, self._inputs(trainer, [1.0, 0.0], [full, full]))
        torch.manual_seed(0)
        loss_padded = trainer._compute_loss(
            trainer.model, self._inputs(trainer, [1.0, 0.0, 1.0, 0.0], [full, full, empty, empty])
        )

        # Rows 0-1 are identical across the two batches; rows 2-3 carry no active token. Pre-fix, the padded loss
        # was exactly half the complete one. (MC draws are resampled per call, so the mc estimator is compared at a
        # looser tolerance that still rules out the factor-of-two denominator bug.)
        rtol = 0.05 if estimator == "mc" else 1e-5
        torch.testing.assert_close(loss_padded, loss_complete, rtol=rtol, atol=1e-5)

    def test_fully_masked_row_has_no_gradient(self):
        # The off-policy KL term -beta * ell must not leak through a fully masked row: the whole row, not only its
        # reward, is out of the loss.
        trainer = self._trainer()
        trainer.model.eval()
        torch.manual_seed(0)
        loss = trainer._compute_loss(trainer.model, self._inputs(trainer, [0.0], [[0] * 5]))
        assert loss.item() == 0.0
        loss.backward()
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                assert torch.count_nonzero(param.grad) == 0, f"{name} received gradient from a fully masked row"

    def test_unscorable_completion_is_fully_masked(self):
        # Bugbot: an unscorable completion (every reward func returned None) must be masked out at scoring time, the
        # same way a truncated one is, so neither its reward nor its KL term reaches the loss.
        trainer = self._trainer(mask_truncated_completions=False)
        prompt_ids = torch.randint(1, trainer.model.config.vocab_size, (2, 3))
        completion_ids = torch.randint(1, trainer.model.config.vocab_size, (2, 5))
        completion_mask = torch.ones(2, 5, dtype=torch.long)

        # Replay the scoring post-processing on a GRPO-shaped output: row 1 is unscorable (NaN for every reward
        # function), row 0 carries reward 0.75.
        trainer._rewards_per_func = torch.tensor([[0.75], [float("nan")]])
        parent_output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": torch.ones_like(prompt_ids),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask.clone(),
        }
        with patch.object(GRPOTrainer, "_generate_and_score_completions", return_value=parent_output):
            output = trainer._generate_and_score_completions([{}, {}])

        torch.testing.assert_close(output["raw_rewards"], torch.tensor([0.75, 0.0]))
        assert output["completion_mask"][0].tolist() == [1] * 5
        assert output["completion_mask"][1].tolist() == [0] * 5, "unscorable row must be fully masked"
