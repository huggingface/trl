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
from datasets import DatasetDict, load_dataset

from trl import GRPOConfig
from trl.experimental.gspo_token import GRPOTrainer as GSPOTokenTrainer

from ..testing_utils import TrlTestCase


class TestGSPOTokenTrainer(TrlTestCase):
    def test_train(self):
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            num_iterations=2,  # the importance sampling weights won't be 0 in this case
            importance_sampling_level="sequence_token",
            report_to="none",
        )
        trainer = GSPOTokenTrainer(
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

    def test_sequence_token_routes_per_token_advantages_into_the_gradient(self):
        """`sequence_token` differs from `sequence` only in the gradient. Its importance weight is the detached sequence
        weight times `exp(logp - logp.detach())`, which is 1 in value, so both levels give the same loss for the same
        inputs. The gradient at token t is the sequence weight times the advantage at t, where `sequence` spreads the
        mean advantage of the sequence over every token. `test_train` runs with one advantage per sequence, for which
        the two levels agree in value and in gradient, so only token-varying advantages tell them apart."""
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
        training_args = GRPOConfig(
            output_dir=self.tmp_dir,
            beta=0.0,  # no KL term, so the importance-sampling level is the only thing that can differ
            loss_type="grpo",  # per-sequence normalization; the default dapo needs the trainer-supplied num_items_in_batch
            importance_sampling_level="sequence_token",
            report_to="none",
        )
        trainer = GSPOTokenTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset,
        )
        trainer.model.eval()

        # Trainer.__init__ moves the model to args.device, so on a GPU runner the inputs have to be built there too.
        device = next(trainer.model.parameters()).device
        torch.manual_seed(0)
        batch_size, prompt_len, completion_len = 2, 3, 6
        prompt_ids = torch.randint(1, 1000, (batch_size, prompt_len), device=device)
        prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
        completion_ids = torch.randint(1, 1000, (batch_size, completion_len), device=device)
        completion_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], device=device)
        # old_per_token_logps must be given explicitly: left unset, it falls back to per_token_logps.detach() and the
        # sequence weight is exactly 1. A small offset from the model's real log-probs keeps the ratio inside the clip
        # range so the two levels are compared on their unclipped objective.
        with torch.no_grad():
            baseline_logps, _, _ = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                torch.cat([prompt_ids, completion_ids], dim=1),
                torch.cat([prompt_mask, completion_mask], dim=1),
                completion_len,
            )
        inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": baseline_logps + 0.05,
        }
        per_sequence_advantages = torch.tensor([1.0, -1.0], device=device)
        per_token_advantages = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, -1.0, 0.0, 0.0, 0.0]], device=device
        )

        def loss_and_grad(level, advantages):
            trainer.importance_sampling_level = level
            trainer.model.zero_grad()
            loss = trainer._compute_loss(trainer.model, {**inputs, "advantages": advantages})
            loss.backward()
            grad = torch.cat([p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None])
            return loss.detach(), grad

        # Control: one advantage per sequence, the two levels agree in value and in gradient.
        seq_loss, seq_grad = loss_and_grad("sequence", per_sequence_advantages)
        tok_loss, tok_grad = loss_and_grad("sequence_token", per_sequence_advantages)
        torch.testing.assert_close(tok_loss, seq_loss)
        torch.testing.assert_close(tok_grad, seq_grad)

        # Token-varying advantages: same loss value, different gradient. Replacing the sequence_token branch by the
        # sequence branch makes the second assertion fail.
        seq_loss, seq_grad = loss_and_grad("sequence", per_token_advantages)
        tok_loss, tok_grad = loss_and_grad("sequence_token", per_token_advantages)
        torch.testing.assert_close(tok_loss, seq_loss)
        assert not torch.allclose(tok_grad, seq_grad), "sequence_token gradient collapsed to the sequence-level one"

    @pytest.mark.parametrize("eval_dataset_type", ["dataset", "dataset_dict", "dict_of_dataset", "none"])
    def test_init_with_eval_dataset(self, eval_dataset_type):
        # Streaming datasets are not yet supported in GSPO-token
        dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

        if eval_dataset_type == "none":
            eval_dataset = None
        elif eval_dataset_type == "dataset":
            eval_dataset = dataset["test"]
        elif eval_dataset_type == "dataset_dict":
            eval_dataset = DatasetDict({"data1": dataset["test"], "data2": dataset["test"]})
        else:  # "dict_of_dataset"
            eval_dataset = {"data1": dataset["test"], "data2": dataset["test"]}

        training_args = GRPOConfig(output_dir=self.tmp_dir, report_to="none")
        trainer = GSPOTokenTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs="trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
        )

        if eval_dataset_type == "none":
            assert trainer.eval_dataset is None
        elif isinstance(trainer.eval_dataset, dict):
            assert set(trainer.eval_dataset.keys()) == {"data1", "data2"}
        else:
            assert trainer.eval_dataset is eval_dataset
