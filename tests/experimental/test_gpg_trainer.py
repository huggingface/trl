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
from datasets import Dataset, load_dataset

from trl import GRPOConfig
from trl.experimental.gpg import GPGConfig, GPGTrainer

from ..testing_utils import TrlTestCase


class TestGPGConfig:
    def test_defaults_match_the_method(self):
        # GPG drops the KL constraint and the reference model, and uses the mean-centered advantage without
        # standard-deviation scaling. Those are config defaults rather than code, so pin them.
        args = GPGConfig("dummy")
        assert args.beta == 0.0
        assert args.scale_rewards == "none"
        assert args.bias_correction
        # The correction cancels a per-completion denominator, and rescales any regularizer folded into the loss
        # before it. Both of those are defaults that diverge from `GRPOConfig`, so pin them too.
        assert args.loss_type == "grpo"
        assert args.router_aux_loss_coef == 0.0


class TestGPGTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            report_to="none",
        )
        trainer = GPGTrainer(
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

    def test_bias_correction_rescales_the_loss(self):
        # The correction divides the loss by the fraction of completion slots that carry signal. Drive `_compute_loss`
        # directly with a stubbed parent so the assertion pins the factor and nothing else.
        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.bias_correction = True
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer.model = type("M", (), {"training": True})()

        base_loss = torch.tensor(2.0)
        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_compute_loss = GRPOTrainerParent._compute_loss
        GRPOTrainerParent._compute_loss = lambda self, model, inputs: base_loss
        try:
            for frac_zero_std, expected in [(0.0, 2.0), (0.5, 4.0), (0.75, 8.0)]:
                trainer._valid_sample_fraction["train"] = 1.0 - frac_zero_std
                assert trainer._compute_loss(None, {}).item() == expected

            # Every group degenerate: the advantages are all zero, so there is no gradient to restore and the factor
            # would divide by zero. The loss must pass through untouched.
            trainer._valid_sample_fraction["train"] = 0.0
            assert trainer._compute_loss(None, {}).item() == 2.0

            # Opting out recovers the uncorrected GRPO magnitude.
            trainer.bias_correction = False
            trainer._valid_sample_fraction["train"] = 0.5
            assert trainer._compute_loss(None, {}).item() == 2.0
        finally:
            GRPOTrainerParent._compute_loss = original_compute_loss

    def test_eval_does_not_clobber_the_train_correction_factor(self):
        # `GRPOTrainer` buffers one train generation across `steps_per_generation` optimizer steps and only
        # regenerates every `steps_per_generation * num_iterations` steps (grpo_trainer.py:1584-1592), while eval
        # generates per batch (:1596). An eval landing inside that window must not leave the buffered train steps
        # rescaling with the eval fraction, so the cache is keyed by mode.
        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.bias_correction = True
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}

        base_loss = torch.tensor(1.0)
        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_compute_loss = GRPOTrainerParent._compute_loss
        GRPOTrainerParent._compute_loss = lambda self, model, inputs: base_loss
        try:
            # Train generation scores a batch where half the groups are degenerate.
            trainer._valid_sample_fraction["train"] = 0.5
            # An eval pass then scores a batch where none are, which used to overwrite the shared scalar.
            trainer._valid_sample_fraction["eval"] = 1.0

            trainer.model = type("M", (), {"training": True})()
            assert trainer._compute_loss(None, {}).item() == 2.0, (
                "the buffered train step must still use the train fraction after an eval pass"
            )

            trainer.model = type("M", (), {"training": False})()
            assert trainer._compute_loss(None, {}).item() == 1.0
        finally:
            GRPOTrainerParent._compute_loss = original_compute_loss

    def test_valid_sample_fraction_counts_informative_groups(self):
        # Paper eq. 7 defines `M` as the samples belonging to groups whose responses are "all right or wrong", so
        # invalidity is a property of the group. Reshaping the gathered advantages to (-1, num_generations) and
        # asking whether any member is non-zero reproduces that. Scoring each completion on its own does not: a
        # continuous reward can land one completion on its group mean inside an otherwise informative group.
        trainer = GPGTrainer.__new__(GPGTrainer)
        # Deliberately different, so a divisor that ignored the mode would land on the wrong group size. The eval
        # cases below are written for groups of 2 and give a different answer when read three at a time.
        trainer.num_generations = 3
        trainer.num_generations_eval = 2
        # The parent slices `advantages` to the local process before returning, and the override gathers it back.
        # Single-process here, so the gather is the identity; what it must do across ranks is restore the parent's
        # pre-slice ordering, which is the tensor these cases stand in for.
        trainer.accelerator = type("A", (), {"process_index": 0, "gather": staticmethod(lambda tensor: tensor)})()
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer._scorable_mask = None

        cases = {
            "train": [
                # One informative group of 3 and one degenerate one: half the groups carry a gradient.
                ([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.5),
                # Rewards [0, 1, 2] center to [-1, 0, 1]. The group is informative, so the factor is 1.0; a
                # per-completion rule would read 2 valid out of 3 and over-correct by 3/2.
                ([-1.0, 0.0, 1.0], 1.0),
                # Every group degenerate: nothing carries a gradient.
                ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
            ],
            "eval": [
                # Groups of 2. Read three at a time this would be one informative group of 3, so 1.0 instead.
                ([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1 / 3),
                ([0.0, 0.0, -1.0, 1.0], 0.5),
                ([0.0, 0.0, 0.0, 0.0], 0.0),
            ],
        }

        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_generate = GRPOTrainerParent._generate_and_score_completions
        try:
            for training in [True, False]:
                mode = "train" if training else "eval"
                trainer.model = type("M", (), {"training": training})()
                for advantages, expected in cases[mode]:
                    trainer._scorable_mask = torch.ones(len(advantages), dtype=torch.bool)
                    GRPOTrainerParent._generate_and_score_completions = lambda self, inputs, values=advantages: {
                        "advantages": torch.tensor(values)
                    }
                    trainer._generate_and_score_completions({})
                    assert trainer._valid_sample_fraction[mode] == pytest.approx(expected), (
                        f"{mode}: advantages {advantages} should give a factor of {expected}"
                    )
        finally:
            GRPOTrainerParent._generate_and_score_completions = original_generate

    def test_generation_and_optimizer_steps_must_be_aligned(self):
        args = GPGConfig(
            output_dir=self.tmp_dir,
            steps_per_generation=2,
            gradient_accumulation_steps=1,
            report_to="none",
        )
        with pytest.raises(ValueError, match="to divide `gradient_accumulation_steps`"):
            GPGTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=args,
            )

    def test_bias_correction_rejects_settings_that_break_the_identity(self):
        # Each of these makes the factor stop canceling the loss denominator, or moves a regularizer's effective
        # coefficient with the reward spread. `__init__` rejects them before the parent loads anything, so no model
        # is built here.
        rejected = [
            ({"loss_type": "luspo"}, "sums each completion's token losses"),
            ({"mask_truncated_completions": True}, "mask_truncated_completions=True"),
            ({"top_entropy_quantile": 0.8}, "top_entropy_quantile<1.0"),
            ({"off_policy_mask_threshold": 2.0}, "off_policy_mask_threshold"),
            ({"beta": 0.1}, "KL term"),
            ({"multi_objective_aggregation": "normalize_then_sum"}, "floating-point error"),
            ({"use_liger_kernel": True}, "compute_liger_loss"),
        ]
        for overrides, message in rejected:
            args = GPGConfig(output_dir=self.tmp_dir, report_to="none", **overrides)
            with pytest.raises(ValueError, match=message):
                GPGTrainer(
                    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                    reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                    args=args,
                )

    def test_token_total_loss_types_are_accepted(self):
        # Eq. 5 normalizes by the total completion-token count, which is what `bnpo` and `dapo` do. Rejecting them
        # would reject the paper's own objective, so the config validation must let them through. These reach the
        # parent, so a bad value would raise from `GRPOTrainer` rather than pass silently.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        for loss_type in ["bnpo", "dapo", "sapo"]:
            training_args = GPGConfig(
                output_dir=self.tmp_dir,
                loss_type=loss_type,
                per_device_train_batch_size=3,
                num_generations=3,
                max_completion_length=8,
                max_steps=1,
                report_to="none",
            )
            trainer = GPGTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )
            assert trainer.bias_correction, f"{loss_type} should keep the correction enabled"

    def test_plain_grpo_config_is_rejected(self):
        # A `GRPOConfig` carries none of GPG's defaults and no `bias_correction` switch, so accepting one would
        # silently train uncorrected GRPO under the GPG name.
        args = GRPOConfig(output_dir=self.tmp_dir, report_to="none")
        with pytest.raises(ValueError, match="requires a `GPGConfig`"):
            GPGTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=args,
            )

    def test_disabling_the_correction_lifts_the_restrictions(self):
        # The negative control for the rejections above: with `bias_correction=False` there is no factor to apply,
        # so settings the correction cannot tolerate are accepted and the trainer builds.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            bias_correction=False,
            loss_type="luspo",
            importance_sampling_level="sequence",
            mask_truncated_completions=True,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            max_steps=1,
            report_to="none",
        )
        trainer = GPGTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        assert not trainer.bias_correction

    def test_valid_sample_fraction_is_computed_on_the_gathered_advantages(self):
        # The parent slices `advantages` to the local process before returning, so reading the local slice would give
        # each rank its own factor and make the result depend on the world size. Simulate two ranks in one process:
        # the local slice holds a partial group, and the stub gather returns what the parent had before slicing. A
        # fraction read from the local slice cannot even reshape by `num_generations`, let alone give 0.5.
        local = torch.tensor([-1.0, 0.0])
        gathered = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.num_generations = 3
        trainer.num_generations_eval = 3
        trainer.model = type("M", (), {"training": True})()
        trainer.accelerator = type("A", (), {"process_index": 0, "gather": staticmethod(lambda tensor: gathered)})()
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer._scorable_mask = torch.ones(6, dtype=torch.bool)

        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_generate = GRPOTrainerParent._generate_and_score_completions
        GRPOTrainerParent._generate_and_score_completions = lambda self, inputs: {"advantages": local}
        try:
            trainer._generate_and_score_completions({})
        finally:
            GRPOTrainerParent._generate_and_score_completions = original_generate

        assert trainer._valid_sample_fraction["train"] == 0.5, (
            "the fraction must come from the gathered advantages, not the local slice"
        )

    def test_degenerate_groups_survive_the_float_residual(self):
        # A degenerate group's advantages are all equal but not all zero: every member subtracts the same group mean
        # from the same reward, so they land together on a shared rounding residual. Build that residual with the
        # parent's own arithmetic rather than hand-picking a constant, so the test tracks whatever the parent does.
        # Testing against zero calls such a group informative; testing the members against each other does not.
        rewards = torch.tensor([2.9, 2.9, 2.9, 3.1, 3.1, 3.1], dtype=torch.float32)
        mean = torch.nanmean(rewards.view(-1, 3), dim=1).repeat_interleave(3, dim=0)
        gathered = torch.nan_to_num(rewards - mean, nan=0.0)
        assert (gathered != 0).any(), "this test is vacuous unless the parent's subtraction leaves a residual"

        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.num_generations = 3
        trainer.num_generations_eval = 3
        trainer.model = type("M", (), {"training": True})()
        trainer.accelerator = type("A", (), {"process_index": 0, "gather": staticmethod(lambda tensor: gathered)})()
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer._scorable_mask = torch.ones(6, dtype=torch.bool)

        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_generate = GRPOTrainerParent._generate_and_score_completions
        GRPOTrainerParent._generate_and_score_completions = lambda self, inputs: {"advantages": gathered}
        try:
            output = trainer._generate_and_score_completions({})
        finally:
            GRPOTrainerParent._generate_and_score_completions = original_generate

        assert trainer._valid_sample_fraction["train"] == 0.0, (
            "both groups are degenerate, so no group carries a gradient and the correction must not fire"
        )
        assert output["advantages"].equal(torch.zeros_like(gathered)), (
            "a degenerate group's shared rounding residual must not reach the policy-gradient loss"
        )

    def test_a_partly_unscorable_degenerate_group_stays_degenerate(self):
        # Three identical rewards plus one completion no reward function could score. `nan_to_num` forces that
        # completion's advantage to exactly zero while the other three share a rounding residual, so the row holds two
        # distinct values without holding any signal. Judging the spread of the non-zero entries keeps it degenerate;
        # comparing every entry against the first would read the lone exact zero as evidence of a gradient.
        rewards = torch.tensor([2.9, 2.9, 2.9, float("nan")], dtype=torch.float32)
        mean = torch.nanmean(rewards.view(-1, 4), dim=1).repeat_interleave(4, dim=0)
        gathered = torch.nan_to_num(rewards - mean, nan=0.0)
        assert gathered[0] != 0 and gathered[3] == 0, (
            "this test is vacuous unless the scorable members carry a residual and the unscorable one is exactly zero"
        )

        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.num_generations = 4
        trainer.num_generations_eval = 4
        trainer.model = type("M", (), {"training": True})()
        trainer.accelerator = type("A", (), {"process_index": 0, "gather": staticmethod(lambda tensor: gathered)})()
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer._scorable_mask = torch.tensor([True, True, True, False])

        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_generate = GRPOTrainerParent._generate_and_score_completions
        GRPOTrainerParent._generate_and_score_completions = lambda self, inputs: {"advantages": gathered}
        try:
            trainer._generate_and_score_completions({})
        finally:
            GRPOTrainerParent._generate_and_score_completions = original_generate

        assert trainer._valid_sample_fraction["train"] == 0.0, (
            "every scorable member of the group has the same reward, so the group carries no gradient"
        )

    def test_zero_advantages_in_an_informative_group_remain_valid(self):
        rewards = torch.tensor([1.0, 1.0, torch.nextafter(torch.tensor(1.0), torch.tensor(torch.inf))])
        mean = torch.nanmean(rewards.view(-1, 3), dim=1).repeat_interleave(3)
        advantages = rewards - mean
        assert (advantages != 0).sum() == 1

        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.num_generations = trainer.num_generations_eval = 3
        trainer.model = type("M", (), {"training": True})()
        trainer.accelerator = type("A", (), {"process_index": 0, "gather": staticmethod(lambda tensor: tensor)})()
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer._scorable_mask = torch.ones(3, dtype=torch.bool)

        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_generate = GRPOTrainerParent._generate_and_score_completions
        GRPOTrainerParent._generate_and_score_completions = lambda self, inputs: {"advantages": advantages}
        try:
            trainer._generate_and_score_completions({})
        finally:
            GRPOTrainerParent._generate_and_score_completions = original_generate

        assert trainer._valid_sample_fraction["train"] == 1.0

    def test_partly_unscorable_informative_group_counts_only_scorable_samples(self):
        advantages = torch.tensor([-0.5, 0.5, 0.0, 0.0])
        scorable = torch.tensor([True, True, False, False])

        trainer = GPGTrainer.__new__(GPGTrainer)
        trainer.num_generations = trainer.num_generations_eval = 4
        trainer.model = type("M", (), {"training": True})()
        trainer.accelerator = type("A", (), {"process_index": 0, "gather": staticmethod(lambda tensor: tensor)})()
        trainer._valid_sample_fraction = {"train": 1.0, "eval": 1.0}
        trainer._scorable_mask = scorable

        GRPOTrainerParent = GPGTrainer.__mro__[1]
        original_generate = GRPOTrainerParent._generate_and_score_completions
        GRPOTrainerParent._generate_and_score_completions = lambda self, inputs: {"advantages": advantages}
        try:
            trainer._generate_and_score_completions({})
        finally:
            GRPOTrainerParent._generate_and_score_completions = original_generate

        assert trainer._valid_sample_fraction["train"] == 0.5

    def test_vllm_importance_sampling_mask_is_rejected(self):
        # `vllm_importance_sampling_correction` defaults to True and its mode defaults to "sequence_mask", so
        # `use_vllm=True` alone silences a completion's tokens while leaving its advantage intact. That is the same
        # dilution the three explicit masking settings are rejected for, so it has to be rejected too.
        with pytest.raises(ValueError, match="vllm_importance_sampling_mode"):
            GPGTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=GPGConfig(output_dir=self.tmp_dir, use_vllm=True, report_to="none"),
                train_dataset=load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train"),
            )

    def test_entropy_bonus_is_rejected(self):
        # `_entropy_bonus_enabled` is only known after the parent runs, so this rejection lives after `super()`.
        # The entropy term is added to the loss before the correction divides it, which would move its effective
        # coefficient with the reward spread.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            entropy_coef=0.1,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        with pytest.raises(ValueError, match="entropy bonus"):
            GPGTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_moe_router_aux_loss_is_rejected(self):
        # `aux_loss_enabled` needs the loaded model to be a Mixture-of-Experts, so this rejection is also post-parent
        # and needs a real MoE model to reach. Passing a non-zero coefficient opts back into the term that
        # `GPGConfig` defaults to `0.0` precisely to avoid.
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            router_aux_loss_coef=0.001,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            report_to="none",
        )
        with pytest.raises(ValueError, match="MoE router auxiliary loss"):
            GPGTrainer(
                model="trl-internal-testing/tiny-Qwen3MoeForCausalLM",
                reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
                args=training_args,
                train_dataset=dataset,
            )

    def test_the_override_runs_on_the_real_generation_path(self):
        # Initialize the factor to an impossible sentinel so the final assertion proves that the override ran.
        dataset = Dataset.from_dict({"prompt": ["The weather is", "The sky is", "The sun is"]})

        training_args = GPGConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,
            num_generations=3,
            max_completion_length=8,
            max_steps=1,
            report_to="none",
        )
        trainer = GPGTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        trainer._valid_sample_fraction["train"] = -1.0
        trainer.train()

        assert 0.0 <= trainer._valid_sample_fraction["train"] <= 1.0
