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
from transformers import AutoModelForCausalLM

from trl.experimental.api import ForwardBackwardOutput, LocalTrainingClient
from trl.trainer.utils import patch_chunked_lm_head

from ..testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


class RemoteStyleTrainingClient:
    """The off-process implementation the protocol documents, with the wire replaced by a second forward.

    A real backend scores the batch remotely and cannot return a loss attached to the model. So it evaluates `loss_fn`
    locally on a leaf, takes `d(loss)/d(log_probs)`, and sends that back to be applied against a surrogate. Standing
    the two "remote" calls up as local forwards keeps the test dependency-free while exercising the same arithmetic.
    """

    def __init__(self, model):
        self.model = model

    def _score(self, input_ids, position_ids, completion_mask):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=input_ids,
            completion_mask=completion_mask,
            use_cache=False,
        )

    def forward_backward(self, model, input_ids, position_ids, completion_mask, loss_fn, aux_loss_coef=0.0):
        with torch.no_grad():
            outputs = self._score(input_ids, position_ids, completion_mask)

        leaf = outputs["log_probs"].detach().requires_grad_(True)
        loss = loss_fn(leaf)
        (grad_log_probs,) = torch.autograd.grad(loss, leaf)

        def apply_surrogate(grad_loss):
            # `grad_loss` is whatever scaling the trainer's backward carries into this scalar. Grad mode is off
            # inside a backward hook, so the replay has to re-enable it or it builds nothing to back-propagate.
            with torch.enable_grad():
                replayed = self._score(input_ids, position_ids, completion_mask)
                surrogate = (replayed["log_probs"] * grad_log_probs * grad_loss).sum()
            surrogate.backward()

        out = loss.detach().requires_grad_(True)
        out.register_hook(apply_surrogate)
        return ForwardBackwardOutput(
            loss=out,
            log_probs=outputs["log_probs"].detach(),
            entropy=outputs["entropy"].detach(),
        )


class TestTrainingClient(TrlTestCase):
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

    def loss_fn(self, log_probs, epsilon_low=0.2, epsilon_high=0.2):
        """`AsyncGRPOTrainer.compute_loss`'s closure, kept in sync with the trainer."""
        coef_1 = torch.exp(log_probs - self.old_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        per_token_loss = -torch.min(coef_1 * self.advantages, coef_2 * self.advantages)
        return (per_token_loss * self.mask).sum()

    def _model_outputs(self):
        return self.model(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            labels=self.input_ids,
            completion_mask=self.completion_mask,
            use_cache=False,
        )

    def _grads(self):
        return {n: p.grad.detach().clone() for n, p in self.model.named_parameters() if p.grad is not None}

    def _client(self, client_type):
        return LocalTrainingClient() if client_type == "local" else RemoteStyleTrainingClient(self.model)

    def _inline_grads(self, loss_scale):
        """Gradients from computing the loss on the live graph, i.e. the behavior before the client existed."""
        self.model.zero_grad(set_to_none=True)
        (self.loss_fn(self._model_outputs()["log_probs"]) * loss_scale).backward()
        return self._grads()

    def _client_grads(self, client, loss_scale):
        self.model.zero_grad(set_to_none=True)
        outputs = client.forward_backward(
            self.model,
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            completion_mask=self.completion_mask,
            loss_fn=self.loss_fn,
        )
        (outputs.loss * loss_scale).backward()
        return self._grads()

    @pytest.mark.parametrize("client_type", ["local", "remote"])
    @pytest.mark.parametrize("loss_scale", [1.0, pytest.param(0.25, id="accumulation-scaled")])
    def test_gradients_match_inline_loss(self, client_type, loss_scale):
        expected = self._inline_grads(loss_scale)
        actual = self._client_grads(self._client(client_type), loss_scale)

        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_loss_stays_attached_in_process(self):
        outputs = LocalTrainingClient().forward_backward(
            self.model,
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            completion_mask=self.completion_mask,
            loss_fn=self.loss_fn,
        )
        # Attached rather than a leaf: the trainer's backward reaches the model without a second forward.
        assert outputs.loss.requires_grad
        assert not outputs.loss.is_leaf
        assert not outputs.log_probs.requires_grad
        assert not outputs.entropy.requires_grad
        assert outputs.aux_loss is None

    @pytest.mark.parametrize("client_type", ["local", "remote"])
    def test_reported_outputs_match_the_model(self, client_type):
        outputs = self._client(client_type).forward_backward(
            self.model,
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            completion_mask=self.completion_mask,
            loss_fn=self.loss_fn,
        )
        expected = self._model_outputs()

        torch.testing.assert_close(outputs.log_probs, expected["log_probs"].detach(), rtol=0, atol=0)
        torch.testing.assert_close(outputs.entropy, expected["entropy"].detach(), rtol=0, atol=0)
        torch.testing.assert_close(outputs.loss.detach(), self.loss_fn(expected["log_probs"]).detach(), rtol=0, atol=0)
