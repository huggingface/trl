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

import pytest
import torch
from datasets import load_dataset

from trl.experimental.up import UPConfig, UPTrainer
from trl.trainer.grpo_trainer import GRPOTrainer

from ..testing_utils import TrlTestCase


class TestUPTrainer(TrlTestCase):
    def _make_trainer_and_inputs(self, advantages, importance_sampling_level="token", **config_kwargs):
        """Build a tiny UP trainer plus a hand-crafted `_compute_loss` inputs dict."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        def dummy_reward_func(completions, **kwargs):
            return [0.0] * len(completions)

        config_kwargs.setdefault("beta", 0.0)  # no KL term by default, so no reference model is needed
        training_args = UPConfig(
            output_dir=self.tmp_dir,
            bf16=False,  # gradients are compared in full fp32 precision
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            importance_sampling_level=importance_sampling_level,
            report_to="none",
            **config_kwargs,
        )
        trainer = UPTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=dummy_reward_func,
            args=training_args,
            train_dataset=dataset,
        )
        device = trainer.accelerator.device
        generator = torch.Generator().manual_seed(42)
        vocab_size = trainer.model.config.vocab_size
        prompt_ids = torch.randint(0, vocab_size, (3, 4), generator=generator).to(device)
        completion_ids = torch.randint(0, vocab_size, (3, 6), generator=generator).to(device)
        completion_mask = torch.ones_like(completion_ids)
        completion_mask[-1, -2:] = 0  # ragged completion lengths, to exercise masking
        inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": torch.ones_like(prompt_ids),
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages.to(device),
            # Deliberately not equal to `mask.sum()`, so that the global (DAPO) normalization UP uses is
            # distinguishable from a local (BNPO) one.
            "num_items_in_batch": completion_mask.sum() * 2,
        }
        return trainer, inputs

    def _grads(self, trainer, loss):
        params = [p for p in trainer.model.parameters() if p.requires_grad]
        return torch.autograd.grad(loss, params)

    @pytest.mark.parametrize("importance_sampling_level", ["token", "sequence"])
    def test_positive_advantages_match_reinforce_and_ignore_old_logps(self, importance_sampling_level):
        # For positive advantages, the gradient of the UP loss must equal the plain REINFORCE gradient
        # Â · ∇ log π_θ (with the same global-token normalization), and must not depend on old_per_token_logps.
        # This also holds at the sequence level: the global active-token normalization weights each sequence by its
        # length, which exactly cancels the length normalization of the sequence-level anchored ratio.
        advantages = torch.tensor([0.3, 1.5, 0.8])  # all positive: every token goes through the UP branch
        trainer, inputs = self._make_trainer_and_inputs(advantages, importance_sampling_level)

        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logits_to_keep = inputs["completion_ids"].size(1)
        per_token_logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, logits_to_keep
        )

        # Simulate a stale (off-policy) old policy
        inputs["old_per_token_logps"] = per_token_logps.detach() - 0.7
        up_loss = trainer._compute_loss(trainer.model, inputs)
        up_grads = self._grads(trainer, up_loss)

        # The self-anchored ratio has forward value exactly 1, so the loss reduces to -Â summed over active tokens
        # and divided by the global token count. Asserting the value (not just the gradient) pins that: dropping the
        # exp() would leave every gradient unchanged while silently changing the reported loss.
        expected = (
            -(inputs["advantages"].unsqueeze(1) * inputs["completion_mask"]).sum() / inputs["num_items_in_batch"]
        )
        torch.testing.assert_close(up_loss, expected)

        # REINFORCE reference: -Â · log π_θ, aggregated with the same global-token normalization
        mask = inputs["completion_mask"]
        reinforce_loss = (
            -(inputs["advantages"].unsqueeze(1) * per_token_logps * mask).sum() / inputs["num_items_in_batch"]
        )
        reinforce_grads = self._grads(trainer, reinforce_loss)

        for up_grad, reinforce_grad in zip(up_grads, reinforce_grads, strict=True):
            torch.testing.assert_close(up_grad, reinforce_grad, rtol=1e-6, atol=1e-8)

        # A different old policy must yield the exact same loss and gradients: the UP branch is anchored to
        # sg(π_θ), not π_old
        inputs["old_per_token_logps"] = per_token_logps.detach() + 0.9
        other_loss = trainer._compute_loss(trainer.model, inputs)
        other_grads = self._grads(trainer, other_loss)
        assert torch.equal(up_loss.detach(), other_loss.detach())
        for up_grad, other_grad in zip(up_grads, other_grads, strict=True):
            assert torch.equal(up_grad, other_grad)

    def test_non_positive_advantages_match_dapo(self):
        # For non-positive advantages, the UP loss must fall back to the clipped surrogate: same loss and same
        # gradients as GRPOTrainer's `dapo`, which shares the global-token normalization. `UPConfig.loss_type`
        # inherits GRPOConfig's `"dapo"` default, so calling the parent implementation on the same trainer runs
        # exactly that path over the same parameters.
        advantages = torch.tensor([-0.3, 0.0, -1.2])  # all non-positive: every token takes the clipped branch
        trainer, inputs = self._make_trainer_and_inputs(advantages)
        assert trainer.loss_type == "dapo"

        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logits_to_keep = inputs["completion_ids"].size(1)
        per_token_logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, logits_to_keep
        )
        # Stale old policy, shifted enough for some ratios to be clipped and others not
        generator = torch.Generator().manual_seed(0)
        shift = torch.empty(per_token_logps.shape).uniform_(-0.5, 0.5, generator=generator)
        inputs["old_per_token_logps"] = per_token_logps.detach() + shift.to(per_token_logps.device)

        up_loss = trainer._compute_loss(trainer.model, inputs)
        up_grads = self._grads(trainer, up_loss)

        dapo_loss = GRPOTrainer._compute_loss(trainer, trainer.model, inputs)
        dapo_grads = self._grads(trainer, dapo_loss)

        assert torch.equal(up_loss.detach(), dapo_loss.detach())
        for up_grad, dapo_grad in zip(up_grads, dapo_grads, strict=True):
            assert torch.equal(up_grad, dapo_grad)

    def _loss_with(self, advantages, shift, ragged_shift=False, **config_kwargs):
        """Loss on a fixed off-policy batch, for a given config.

        A uniform `shift` makes the sequence-level mean log-ratio equal every token's, so token and sequence levels
        coincide by construction. Pass `ragged_shift=True` to vary the shift within each sequence and tell them apart.
        """
        trainer, inputs = self._make_trainer_and_inputs(advantages, **config_kwargs)
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        per_token_logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, inputs["completion_ids"].size(1)
        )
        offset = torch.full_like(per_token_logps, shift)
        if ragged_shift:
            offset = offset + torch.linspace(-0.4, 0.4, per_token_logps.size(1), device=per_token_logps.device)
        inputs["old_per_token_logps"] = per_token_logps.detach() + offset
        return trainer._compute_loss(trainer.model, inputs).item()

    def test_sequence_level_leaves_the_positive_branch_unchanged(self):
        # The positive branch is anchored to sg(π_θ), and the global active-token normalization exactly cancels the
        # sequence-level length normalization, so `importance_sampling_level` cannot change it. Only the
        # non-positive branch, which uses the real importance ratio, responds to the level.
        positive = torch.tensor([0.3, 1.5, 0.8])
        negative = torch.tensor([-0.3, -0.9, -1.2])
        # `ragged_shift` varies the log-ratio within each sequence; without it the sequence mean equals every token
        # and the two levels would agree trivially.
        assert self._loss_with(positive, -0.7, ragged_shift=True, importance_sampling_level="token") == pytest.approx(
            self._loss_with(positive, -0.7, ragged_shift=True, importance_sampling_level="sequence")
        )
        assert self._loss_with(negative, -0.7, ragged_shift=True, importance_sampling_level="token") != pytest.approx(
            self._loss_with(negative, -0.7, ragged_shift=True, importance_sampling_level="sequence")
        )

    def test_delta_caps_the_non_positive_branch_and_gates_epsilon_high(self):
        # `delta` clamps the ratio before the min, so it bounds the non-positive branch. `epsilon_high` only ever
        # binds through that same min, i.e. when `delta < 1 + epsilon_high`; otherwise it is dominated and inert.
        negative = torch.tensor([-0.3, -0.9, -1.2])
        shift = -2.0  # push ratios well above 1, so the caps are reachable

        uncapped = self._loss_with(negative, shift)
        assert self._loss_with(negative, shift, delta=1.05) < uncapped  # delta bites

        # epsilon_high is inert without delta, and inert while delta stays above 1 + epsilon_high
        assert self._loss_with(negative, shift, epsilon_high=5.0) == pytest.approx(uncapped)
        assert self._loss_with(negative, shift, delta=4.0, epsilon_high=0.2) == pytest.approx(
            self._loss_with(negative, shift, delta=4.0, epsilon_high=1.0)  # 1 + 1.0 < 4.0, still dominated
        )
        # but it binds once delta drops below 1 + epsilon_high
        assert self._loss_with(negative, shift, delta=1.05, epsilon_high=0.2) != pytest.approx(
            self._loss_with(negative, shift, delta=1.05, epsilon_high=5.0)
        )

    def test_beta_adds_the_kl_term(self):
        # With beta > 0 the KL penalty is added to the per-token loss and logged, on both branches.
        advantages = torch.tensor([0.7, -0.4, 0.0])
        trainer, inputs = self._make_trainer_and_inputs(advantages, beta=0.04)
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        per_token_logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model, input_ids, attention_mask, inputs["completion_ids"].size(1)
        )
        inputs["old_per_token_logps"] = per_token_logps.detach() - 0.7
        # A reference policy that differs from the current one, so the KL term is non-zero
        inputs["ref_per_token_logps"] = per_token_logps.detach() - 0.5

        loss_with_kl = trainer._compute_loss(trainer.model, inputs).item()
        trainer.beta = 0.0
        loss_without_kl = trainer._compute_loss(trainer.model, inputs).item()

        assert loss_with_kl != pytest.approx(loss_without_kl)
        mode = "train" if trainer.model.training else "eval"
        assert trainer._metrics[mode]["kl"], "the KL metric must be logged when beta > 0"
        assert trainer._metrics[mode]["kl"][0] > 0.0

    def test_liger_kernel_not_supported(self):
        # `GRPOTrainer.compute_loss` routes to the fused Liger loss before `_compute_loss`, which would silently
        # optimize the GRPO objective instead of UP, so the trainer refuses the combination up front.
        training_args = UPConfig(output_dir=self.tmp_dir, bf16=False, use_liger_kernel=True, report_to="none")
        with pytest.raises(NotImplementedError, match="not supported by `UPTrainer`"):
            UPTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
                args=training_args,
            )

    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = UPConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 1 in this case
            report_to="none",
        )
        trainer = UPTrainer(
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
