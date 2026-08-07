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

import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

from trl.experimental.api import LocalTrainingClient
from trl.trainer.utils import patch_chunked_lm_head

from ..testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def grpo_loss(log_probs, old_log_probs, advantages, completion_mask, epsilon_low=0.2, epsilon_high=0.2):
    """`AsyncGRPOTrainer.compute_loss` from `log_probs` onward, kept in sync with the trainer."""
    log_ratio = log_probs - old_log_probs
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
    return (per_token_loss * completion_mask).sum()


class TestLocalTrainingClient(TrlTestCase):
    def setup_method(self):
        torch.manual_seed(0)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="sdpa")
        patch_chunked_lm_head(self.model, chunk_size=8192, temperature=1.0)
        self.model.train()

        torch.manual_seed(1)
        self.input_ids = torch.randint(0, self.model.config.vocab_size, (2, 12))
        self.position_ids = torch.arange(12).expand(2, 12)
        self.completion_mask = torch.zeros(2, 12, dtype=torch.long)
        self.completion_mask[:, 6:] = 1

        shifted = self.completion_mask[:, 1:].float()
        torch.manual_seed(2)
        self.mask = shifted
        self.old_log_probs = torch.randn_like(shifted) * 0.1 - 1.0
        self.advantages = torch.randn_like(shifted)

    def _model_forward(self):
        outputs = self.model(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            labels=self.input_ids,
            completion_mask=self.completion_mask,
            use_cache=False,
        )
        return outputs["log_probs"]

    def _loss(self, log_probs):
        return grpo_loss(log_probs, self.old_log_probs, self.advantages, self.mask)

    def _grads(self):
        return {n: p.grad.detach().clone() for n, p in self.model.named_parameters() if p.grad is not None}

    def _inline_grads(self):
        """Gradients from computing the loss on the live graph, i.e. the behavior before the client existed."""
        self.model.zero_grad(set_to_none=True)
        self._loss(self._model_forward()).backward()
        return self._grads()

    def test_gradients_match_inline_loss(self):
        expected = self._inline_grads()

        self.model.zero_grad(set_to_none=True)
        client = LocalTrainingClient()
        output = client.forward(self.model, self.input_ids, self.position_ids, self.completion_mask)
        self._loss(output.log_probs).backward()
        client.backward(output.log_probs.grad)

        actual = self._grads()
        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_gradients_match_when_log_probs_round_trip(self):
        """A remote backend cannot share an autograd graph, so it replays the forward and applies the surrogate.

        Same gradients, at the cost of a second forward pass.
        """
        expected = self._inline_grads()

        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            shipped = self._model_forward().clone()
        shipped.requires_grad_(True)
        self._loss(shipped).backward()
        # Second forward, standing in for the backend's replay; the surrogate is what carries the gradient.
        (self._model_forward() * shipped.grad).sum().backward()

        actual = self._grads()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_forward_hands_out_a_leaf(self):
        client = LocalTrainingClient()
        output = client.forward(self.model, self.input_ids, self.position_ids, self.completion_mask)
        assert output.log_probs.is_leaf
        assert output.log_probs.requires_grad
        assert not output.entropy.requires_grad
        assert output.aux_loss is None

    def test_forward_matches_model_log_probs(self):
        client = LocalTrainingClient()
        output = client.forward(self.model, self.input_ids, self.position_ids, self.completion_mask)
        torch.testing.assert_close(output.log_probs, self._model_forward().detach(), rtol=0, atol=0)

    def test_clip_grad_norm_delegates_to_accelerator(self):
        client = LocalTrainingClient()
        output = client.forward(self.model, self.input_ids, self.position_ids, self.completion_mask)
        self._loss(output.log_probs).backward()
        client.backward(output.log_probs.grad)

        accelerator = Accelerator()
        expected = torch.nn.utils.get_total_norm([p.grad for p in self.model.parameters() if p.grad is not None])
        torch.testing.assert_close(
            client.clip_grad_norm(self.model, accelerator, 1.0).float(), expected.float(), rtol=1e-5, atol=1e-8
        )

    def test_hook_wiring_matches_manual_backward(self):
        """The trainer registers `client.backward` as a hook on `log_probs` rather than calling it directly."""
        expected = self._inline_grads()

        self.model.zero_grad(set_to_none=True)
        client = LocalTrainingClient()
        output = client.forward(self.model, self.input_ids, self.position_ids, self.completion_mask)
        output.log_probs.register_hook(client.backward)
        self._loss(output.log_probs).backward()

        actual = self._grads()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
