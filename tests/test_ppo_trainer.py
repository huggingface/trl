# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import copy
import fnmatch
import gc
import re
import tempfile
import unittest
from functools import partial

import pytest
import torch
from huggingface_hub import HfApi, HfFolder, delete_repo
from parameterized import parameterized
from pytest import mark
from requests.exceptions import HTTPError
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import respond_to_batch

from .testing_constants import CI_HUB_ENDPOINT, CI_HUB_USER, CI_HUB_USER_TOKEN
from .testing_utils import require_peft, require_torch_multi_gpu


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


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, response_data):
        self.query_data = query_data
        self.response_data = response_data

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        return self.query_data[idx], self.response_data[idx]


def apply_mask(values, mask):
    unmasked_values = []
    for v, m in zip(values, mask):
        if m == 1:
            unmasked_values.append(v)
    return torch.Tensor(unmasked_values)


def abs_diff_masked_tensors(tensor_1, tensor_2, mask_1, mask_2):
    diffs = []
    for l1, l2, m1, m2 in zip(tensor_1, tensor_2, mask_1, mask_2):
        diff = apply_mask(l1, m1) - apply_mask(l2, m2)
        diffs.append(diff.sum())
    return abs(sum(diffs))


class PPOTrainerTester(unittest.TestCase):
    """
    A wrapper class for testing PPOTrainer
    """

    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls._token = CI_HUB_USER_TOKEN
        cls._api = HfApi(endpoint=CI_HUB_ENDPOINT)
        HfFolder.save_token(CI_HUB_USER_TOKEN)

        # model_id
        cls.model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"

        # get models and tokenizer
        cls.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(cls.model_id)
        cls.gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(cls.model_id)
        cls.gpt2_tokenizer = AutoTokenizer.from_pretrained(cls.model_id)

        cls.gpt2_tokenizer.pad_token = cls.gpt2_tokenizer.eos_token

        # get bloom as right padding examples:
        model_id = "trl-internal-testing/tiny-BloomForCausalLM-correct-vocab"
        cls.bloom_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        cls.bloom_tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_id = "trl-internal-testing/tiny-T5ForConditionalGeneration-correct-vocab"
        cls.t5_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_id)
        cls.t5_tokenizer = AutoTokenizer.from_pretrained(model_id)

        # initialize trainer
        cls.ppo_config = PPOConfig(batch_size=2, mini_batch_size=1, log_with=None)

    @classmethod
    def tearDownClass(cls):
        for model in [f"{CI_HUB_USER}/test-ppo-trainer"]:
            try:
                delete_repo(token=cls._token, repo_id=model)
            except HTTPError:
                pass

    def setUp(self):
        # initialize trainer
        self.ppo_config = PPOConfig(batch_size=2, mini_batch_size=1, log_with=None)
        self.gpt2_model.train()
        return super().setUp()

    def tearDown(self):
        # free memory
        gc.collect()

    def _init_dummy_dataset(self):
        # encode a query
        query_txt = "This morning I went to the "
        query_tensor = self.gpt2_tokenizer.encode(query_txt, return_tensors="pt")
        assert query_tensor.shape == (1, 7)
        # get model response
        response_tensor = respond_to_batch(self.gpt2_model, query_tensor)
        assert response_tensor.shape == (1, 20)

        # create a dummy dataset
        min_length = min(len(query_tensor[0]), len(response_tensor[0]))
        dummy_dataset = DummyDataset(
            [query_tensor[:, :min_length].squeeze(0) for _ in range(2)],
            [response_tensor[:, :min_length].squeeze(0) for _ in range(2)],
        )

        return dummy_dataset

    def test_drop_last_dataloader(self):
        self.ppo_config = PPOConfig(batch_size=3, mini_batch_size=1, log_with=None)

        dummy_dataset = self._init_dummy_dataset()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=self.gpt2_model_ref,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        dummy_dataloader = ppo_trainer.dataloader

        assert len(dummy_dataloader) == 0

    def test_ppo_step(self):
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=self.gpt2_model_ref,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        for param in ppo_trainer.model.parameters():
            assert param.grad is not None

        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

    def test_ppo_step_with_masks(self):
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=self.gpt2_model_ref,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]

            response_mask = [torch.ones_like(r) for r in response_tensor]

            # train model
            train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward, response_mask)
            break

        for param in ppo_trainer.model.parameters():
            assert param.grad is not None

        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

    def test_ppo_step_with_no_ref_sgd(self):
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()
        optimizer = torch.optim.SGD(self.gpt2_model.parameters(), lr=0.01)

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            optimizer=optimizer,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader

        assert isinstance(ppo_trainer.optimizer.optimizer, torch.optim.SGD)

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        for name, param in ppo_trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            assert param.grad is None, f"Parameter {name} has a gradient"

        # Finally check stats
        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

    def test_ppo_step_with_no_ref_sgd_lr_scheduler(self):
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()
        optimizer = torch.optim.SGD(self.gpt2_model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            optimizer=optimizer,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
            lr_scheduler=lr_scheduler,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader

        assert isinstance(ppo_trainer.optimizer.optimizer, torch.optim.SGD)
        assert isinstance(ppo_trainer.lr_scheduler.scheduler, torch.optim.lr_scheduler.ExponentialLR)

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        for name, param in ppo_trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            assert param.grad is None, f"Parameter {name} has a gradient"

        # Finally check stats
        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

        # assert that the LR has increased for exponential decay
        assert train_stats["ppo/learning_rate"] > self.ppo_config.learning_rate

    def test_ppo_step_with_no_ref(self):
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()
        self.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id)

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        for name, param in ppo_trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            assert param.grad is None, f"Parameter {name} has a gradient"

        # initialize a new gpt2 model:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id)
        for name, param in ppo_trainer.ref_model.named_parameters():
            if "v_head" not in name:
                name = name.replace("pretrained_model.", "")

                assert torch.allclose(
                    param.cpu(), model.state_dict()[name].cpu()
                ), f"Parameter {name} has changed from the original model"

        # Finally check stats
        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

    def test_ppo_step_with_no_ref_custom_layers(self):
        """
        Test PPO step with no reference model and custom layers
        For shared layers configuration, all the layers after the `num_shared_layers` are considered as custom layers
        therefore the gradients should be computed for these layers only.
        """
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()
        self.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id)
        num_shared_layers = 1

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
            num_shared_layers=num_shared_layers,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        pattern = r".*transformer\.h\.(\d+)\..*"
        final_layers = ["ln_f", "v_head", "lm_head"]

        for name, param in ppo_trainer.model.named_parameters():
            if re.match(pattern, name):
                layer_number = int(re.match(pattern, name).groups(0)[0])
                if layer_number < num_shared_layers:
                    assert param.grad is None, f"Parameter {name} has a gradient"
                else:
                    assert param.grad is not None, f"Parameter {name} has no gradient"
            elif any(layer in name for layer in final_layers):
                assert param.grad is not None, f"Parameter {name} has no gradient"

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            assert param.grad is None, f"Parameter {name} has a gradient"

        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

    def test_ppo_step_with_ref_and_custom_layers_warning(self):
        """
        Test PPO step with a reference model and custom layers
        The trainer should raise a warning if the argument `num_shared_layers` is set
        together with a reference model.
        """
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        num_shared_layers = 6

        with self.assertWarns(UserWarning):
            _ = PPOTrainer(
                config=self.ppo_config,
                model=self.gpt2_model,
                ref_model=self.gpt2_model_ref,
                tokenizer=self.gpt2_tokenizer,
                dataset=dummy_dataset,
                num_shared_layers=num_shared_layers,
            )

    def test_ppo_step_rewards_shape(self):
        """
        Test if the rewards shape is correct by asserting that if a wrong reward shape is passed, we get
        a value error.
        """

        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor([[1.0]]), torch.tensor([[0.0]])]
            # train model - this should raise an error
            with pytest.raises(ValueError):
                _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)

            reward = [torch.tensor([1.0]), torch.tensor([0.0])]
            # train model - this should work
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        # check if the gradients are computed for the model
        for name, param in ppo_trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            assert param.grad is None, f"Parameter {name} has a gradient"

    def test_ppo_step_input_shape(self):
        """
        Test if the shape of the expected inputs are correct
        """
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor([1.0]), torch.tensor([0.0])]
            # train model - this should raise an error
            bs = ppo_trainer.config.batch_size

            queries, responses, _, _ = ppo_trainer._step_safety_checker(
                bs, list(query_tensor), list(response_tensor), reward
            )

            assert isinstance(queries, list), f"queries should be a list, got {type(queries)}"
            assert isinstance(responses, list), f"responses should be a list, got {type(responses)}"

            # check the shapes
            for i in range(bs):
                assert queries[i].shape == torch.Size([7])
                assert responses[i].size() == torch.Size([7])
            break

    def test_ppo_step_no_dataset(self):
        """
        Test if the training loop works fine without passing a dataset
        """
        query_txt = "This morning I went to the "
        query_tensor = self.gpt2_tokenizer.encode(query_txt, return_tensors="pt")
        self.ppo_config.batch_size = 1

        response_tensor = respond_to_batch(self.gpt2_model, query_tensor)

        # Check that this warns the user about batch size
        with self.assertWarns(UserWarning):
            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                model=self.gpt2_model,
                ref_model=self.gpt2_model_ref,
                tokenizer=self.gpt2_tokenizer,
            )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        # train model with ppo
        reward = [torch.tensor([1.0])]
        # train model - this should work fine
        train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

        # check gradients
        for name, param in ppo_trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            assert param.grad is None, f"Parameter {name} has a gradient"

        # check train stats
        for stat in EXPECTED_STATS:
            assert stat in train_stats, f"Train stats should contain {stat}"

    def test_loss_trainer(self):
        """
        Test if the loss trainer works fine
        """
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        self.gpt2_model.eval()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        dummy_queries = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6, 7])]
        dummy_responses = [torch.tensor([5, 6, 7, 8, 9]), torch.tensor([8, 9, 10, 11, 12, 13])]
        dummy_scores = torch.Tensor([1, 2])

        ppo_trainer.config.mini_batch_size = 1
        ppo_trainer.config.batch_size = 1
        model_inputs = ppo_trainer.prepare_model_inputs(dummy_queries, dummy_responses)
        all_logprobs, _, values, mask = ppo_trainer.batched_forward_pass(
            self.gpt2_model, dummy_queries, dummy_responses, model_inputs
        )

        # dummy values
        ref_logprobs = all_logprobs + 1
        logits = torch.exp(all_logprobs)
        vpreds = values + 0.1

        score, non_score, kls = ppo_trainer.compute_rewards(dummy_scores, all_logprobs, ref_logprobs, mask)
        values, advantages, returns = ppo_trainer.compute_advantages(values, score, mask)

        # just make sure a dummy loss is computed
        idx = 0
        pg_loss, v_loss, _ = ppo_trainer.loss(
            all_logprobs[idx].unsqueeze(0),
            values[idx].unsqueeze(0),
            logits[idx].unsqueeze(0),
            vpreds[idx].unsqueeze(0),
            ref_logprobs[idx].unsqueeze(0),
            mask[idx].unsqueeze(0),
            advantages[idx].unsqueeze(0),
            returns[idx].unsqueeze(0),
        )

        assert abs(pg_loss.item() - 2.0494) < 0.0001
        assert abs(v_loss.item() - 0.0711) < 0.0001

        # check if we get same results with masked parts removed
        pg_loss_unmasked, v_loss_unmasked, _ = ppo_trainer.loss(
            apply_mask(all_logprobs[idx], mask[idx]).unsqueeze(0),
            apply_mask(values[idx], mask[idx]).unsqueeze(0),
            apply_mask(logits[idx], mask[idx]).unsqueeze(0),
            apply_mask(vpreds[idx], mask[idx]).unsqueeze(0),
            apply_mask(ref_logprobs[idx], mask[idx]).unsqueeze(0),
            apply_mask(mask[idx], mask[idx]).unsqueeze(0),
            apply_mask(advantages[idx], mask[idx]).unsqueeze(0),
            apply_mask(returns[idx], mask[idx]).unsqueeze(0),
        )
        assert abs(pg_loss_unmasked.item() - 2.0494) < 0.0001
        assert abs(v_loss_unmasked.item() - 0.0711) < 0.0001

    @parameterized.expand(
        [
            ["gpt2"],
            ["bloom"],
            ["t5"],
        ]
    )
    def test_batched_forward_pass(self, name):
        """
        Test if the loss trainer works fine
        """
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        dummy_queries = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6, 7])]
        dummy_responses = [torch.tensor([5, 6, 7, 8, 9]), torch.tensor([8, 9, 10, 11, 12, 13])]

        if name == "gpt2":
            model = self.gpt2_model
            tokenizer = self.gpt2_tokenizer
        elif name == "bloom":
            model = self.bloom_model
            tokenizer = self.bloom_tokenizer
        elif name == "t5":
            model = self.t5_model
            tokenizer = self.t5_tokenizer

        model.eval()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dummy_dataset,
        )

        # we test all combinations of fwd_bs and bs:
        # if fwd_bs=bs=1: no padding is applied and only one forward pass
        # if fwd_bs=1/bs=2: padding is applied and results computed in two fwd passes
        # if fwd_bs=bs=2: padding is applied and results computed in one fwd pass

        ppo_trainer.config.mini_batch_size = 1
        ppo_trainer.config.batch_size = 1

        model_inputs = ppo_trainer.prepare_model_inputs([dummy_queries[0]], [dummy_responses[0]])
        logprobs_0, logits_0, values_0, mask_0 = ppo_trainer.batched_forward_pass(
            model, [dummy_queries[0]], [dummy_responses[0]], model_inputs
        )

        ppo_trainer.config.batch_size = 2
        model_inputs = ppo_trainer.prepare_model_inputs(dummy_queries, dummy_responses)
        logprobs_1, logits_1, values_1, mask_1 = ppo_trainer.batched_forward_pass(
            model, dummy_queries, dummy_responses, model_inputs
        )

        ppo_trainer.config.mini_batch_size = 2
        model_inputs = ppo_trainer.prepare_model_inputs(dummy_queries, dummy_responses)
        logprobs_2, logits_2, values_2, mask_2 = ppo_trainer.batched_forward_pass(
            model, dummy_queries, dummy_responses, model_inputs
        )

        assert abs_diff_masked_tensors(logprobs_1, logprobs_2, mask_1, mask_2) <= 0.0001
        assert abs_diff_masked_tensors(values_1, values_2, mask_1, mask_2) <= 0.0001

        assert abs_diff_masked_tensors(logprobs_0, logprobs_2[:1], mask_0, mask_2[:1]) <= 0.0001
        assert abs_diff_masked_tensors(values_0, values_2[:1], mask_0, mask_2[:1]) <= 0.0001

    def test_ppo_trainer_max_grad_norm(self):
        """
        Test if the `max_grad_norm` feature works as expected
        """
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        self.ppo_config.max_grad_norm = 0.00001
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        # check gradients
        for name, param in ppo_trainer.model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.all(
                param.grad.abs() <= self.ppo_config.max_grad_norm
            ), f"Parameter {name} has a gradient larger than max_grad_norm"

    def test_ppo_trainer_kl_penalty(self):
        dummy_dataset = self._init_dummy_dataset()

        log_probs = torch.Tensor([[0.5, 0.2, 0.1], [0.6, 0.2, 0.1]])
        ref_log_probs = torch.Tensor([[0.4, 0.3, 0.0], [0.7, 0.1, 0.3]])

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        expected_output = torch.Tensor([[0.1000, -0.1000, 0.1000], [-0.1000, 0.1000, -0.2000]])
        assert torch.allclose(ppo_trainer._kl_penalty(log_probs, ref_log_probs), expected_output)

        self.ppo_config.kl_penalty = "abs"
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        expected_output = torch.Tensor([[0.1000, 0.1000, 0.1000], [0.1000, 0.1000, 0.2000]])
        assert torch.allclose(ppo_trainer._kl_penalty(log_probs, ref_log_probs), expected_output)

        self.ppo_config.kl_penalty = "mse"
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        expected_output = torch.Tensor([[0.0050, 0.0050, 0.0050], [0.0050, 0.0050, 0.0200]])
        assert torch.allclose(ppo_trainer._kl_penalty(log_probs, ref_log_probs), expected_output)

    def test_ppo_trainer_full_kl_penalty(self):
        # a few more extensive tests for the full kl option as it is more involved
        dummy_dataset = self._init_dummy_dataset()

        self.ppo_config.kl_penalty = "full"
        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        # Test on tensors for size B,S,T = (1,2,3)
        # test for when the two dists are the same
        log_probs = torch.Tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.3],
                ]
            ]
        ).exp()

        ref_log_probs = torch.Tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.3],
                ]
            ]
        ).exp()

        expected_output = torch.Tensor(
            [[0.0, 0.0]],
        )
        output = ppo_trainer._kl_penalty(log_probs, ref_log_probs)
        assert output.shape == (1, 2)
        assert torch.allclose(output, expected_output)

        # test for when the two dists are almost not overlapping
        log_probs = torch.Tensor(
            [
                [
                    [0.98, 0.01, 0.01],
                    [0.01, 0.98, 0.01],
                ]
            ]
        ).log()

        ref_log_probs = torch.Tensor(
            [
                [
                    [0.01, 0.01, 0.98],
                    [0.01, 0.01, 0.98],
                ]
            ]
        ).log()

        expected_output = torch.Tensor(
            [[4.4474, 4.4474]],
        )
        output = ppo_trainer._kl_penalty(log_probs, ref_log_probs)
        assert output.shape == (1, 2)
        assert torch.allclose(output, expected_output)

        # test for when the two dists are almost not overlapping
        log_probs = torch.Tensor(
            [
                [
                    [0.49, 0.02, 0.49],
                    [0.49, 0.02, 0.49],
                ]
            ]
        ).log()

        ref_log_probs = torch.Tensor(
            [
                [
                    [0.01, 0.98, 0.01],
                    [0.49, 0.02, 0.49],
                ]
            ]
        ).log()

        expected_output = torch.Tensor(
            [[3.7361, 0.0]],
        )
        output = ppo_trainer._kl_penalty(log_probs, ref_log_probs)
        assert output.shape == (1, 2)
        assert torch.allclose(output, expected_output, atol=0.0001)

    @require_peft
    @mark.peft_test
    def test_peft_model_ppo_trainer(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        gpt2_model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # this line is very important
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        gpt2_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        peft_model = get_peft_model(gpt2_model, lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

        dummy_dataset = self._init_dummy_dataset()
        self.ppo_config.batch_size = 2
        self.ppo_config.mini_batch_size = 1

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        assert ppo_trainer.ref_model is None

        dummy_dataloader = ppo_trainer.dataloader

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model by running a step twice
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)

            ppo_trainer.model.train()
            ppo_trainer.model.gradient_checkpointing_enable()
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        # check gradients
        for name, param in model.named_parameters():
            if "lora" in name or "v_head" in name:
                assert param.grad is not None, f"Parameter {name} has a no gradient"
            else:
                assert param.grad is None, f"Parameter {name} has a gradient"

    @require_peft
    @mark.peft_test
    def test_peft_model_ppo_adapter_rm_trainer(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

        dummy_inputs = torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        rm_lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        )

        reward_model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        reward_model = get_peft_model(reward_model, rm_lora_config)
        dummy_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, reward_model.parameters()), lr=1e-3)

        previous_rm_logits = reward_model(dummy_inputs).logits
        loss = previous_rm_logits.mean()
        loss.backward()

        dummy_optim.step()
        reward_model.eval()

        original_rm_logits = reward_model(dummy_inputs).logits

        with tempfile.TemporaryDirectory() as tmpdirname:
            reward_model.save_pretrained(tmpdirname)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            gpt2_model = AutoModelForCausalLM.from_pretrained(self.model_id)

            # this line is very important
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            gpt2_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            peft_model = get_peft_model(gpt2_model, lora_config)
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                peft_model,
                reward_adapter=tmpdirname,
            )

            dummy_dataset = self._init_dummy_dataset()
            self.ppo_config.batch_size = 2
            self.ppo_config.mini_batch_size = 1

            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                model=model,
                ref_model=None,
                tokenizer=self.gpt2_tokenizer,
                dataset=dummy_dataset,
            )
            ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
            assert ppo_trainer.ref_model is None

            dummy_dataloader = ppo_trainer.dataloader

            # train model with ppo
            for query_tensor, response_tensor in dummy_dataloader:
                # define a reward for response
                # (this could be any reward such as human feedback or output from another model)
                reward = [torch.tensor(1.0), torch.tensor(0.0)]
                # train model by running a step twice
                _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)

                ppo_trainer.model.train()
                ppo_trainer.model.gradient_checkpointing_enable()
                _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
                break

            new_logits = ppo_trainer.model.compute_reward_score(dummy_inputs)
            assert not torch.allclose(previous_rm_logits, new_logits[:, -1, :])
            assert torch.allclose(original_rm_logits, new_logits[:, -1, :])

            # check gradients
            for name, param in model.named_parameters():
                if ("lora" in name or "v_head" in name) and ("reward" not in name):
                    assert param.grad is not None, f"Parameter {name} has a no gradient"
                else:
                    assert param.grad is None, f"Parameter {name} has a gradient"

    @unittest.skip("Fix by either patching `whomai()` to work in the staging endpoint or use a dummy prod user.")
    def test_push_to_hub(self):
        REPO_NAME = "test-ppo-trainer"
        repo_id = f"{CI_HUB_USER}/{REPO_NAME}"

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=self._init_dummy_dataset(),
        )
        with tempfile.TemporaryDirectory():
            url = ppo_trainer.push_to_hub(repo_id=repo_id, token=self._token, api_endpoint=CI_HUB_ENDPOINT)
            # Extract repo_name from the url
            re_search = re.search(CI_HUB_ENDPOINT + r"/([^/]+/[^/]+)/", url)
            assert re_search is not None
            hub_repo_id = re_search.groups()[0]
            # Check we created a Hub repo
            assert hub_repo_id == repo_id
            # Ensure all files are present
            files = sorted(self._api.list_repo_files(hub_repo_id))
            assert all(
                fnmatch.fnmatch(file, expected_file)
                for file, expected_file in zip(
                    files,
                    [
                        ".gitattributes",
                        "README.md",
                        "config.json",
                        "merges.txt",
                        "pytorch_model.bin",
                        "special_tokens_map.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    ],
                )
            )

    @require_peft
    @require_torch_multi_gpu
    @mark.peft_test
    def test_peft_model_ppo_trainer_multi_gpu(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        gpt2_model = AutoModelForCausalLM.from_pretrained(
            "gpt2", device_map="balanced", max_memory={0: "500MB", 1: "500MB"}
        )

        assert set(gpt2_model.hf_device_map.values()) == {0, 1}

        # this line is very important
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        gpt2_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        peft_model = get_peft_model(gpt2_model, lora_config)
        model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)

        assert model.is_sequential_parallel

        dummy_dataset = self._init_dummy_dataset()
        self.ppo_config.batch_size = 2
        self.ppo_config.mini_batch_size = 1

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        assert ppo_trainer.ref_model is None

        dummy_dataloader = ppo_trainer.dataloader

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model by running a step twice
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)

            ppo_trainer.model.train()
            ppo_trainer.model.gradient_checkpointing_enable()
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        # check gradients
        for name, param in model.named_parameters():
            if "lora" in name or "v_head" in name:
                assert param.grad is not None, f"Parameter {name} has a no gradient"
            else:
                assert param.grad is None, f"Parameter {name} has a gradient"

    def test_generation(self):
        dummy_dataset = self._init_dummy_dataset()

        model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            dataset=dummy_dataset,
        )

        input_texts = ["this is a test", "this is another, longer test"]

        generation_kwargs = {"do_sample": False, "max_new_tokens": 4, "pad_token_id": tokenizer.eos_token_id}

        tokenizer.pad_token = tokenizer.eos_token

        model_inputs = [tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts]

        generations_batched = ppo_trainer.generate(model_inputs, batch_size=2, **generation_kwargs)
        generations_batched = tokenizer.batch_decode(generations_batched)

        generations_single = [ppo_trainer.generate(inputs, **generation_kwargs).squeeze() for inputs in model_inputs]
        generations_single = tokenizer.batch_decode(generations_single)

        assert generations_single == generations_batched

    def test_generation_with_ref_model(self):
        dummy_dataset = self._init_dummy_dataset()
        model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Negate the weights in the last layer of the ref model so it never
        # outputs the same things as the primary model
        ref_model = copy.deepcopy(model)
        lm_head_weight = ref_model.pretrained_model.lm_head.weight
        lm_head_weight.data = -lm_head_weight.data

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dummy_dataset,
        )

        input_texts = ["this is a test", "this is another, longer test"]

        generation_kwargs = {"do_sample": False, "max_new_tokens": 4, "pad_token_id": tokenizer.eos_token_id}

        tokenizer.pad_token = tokenizer.eos_token

        model_inputs = [tokenizer(txt, return_tensors="pt").input_ids.squeeze() for txt in input_texts]

        generations_batched, ref_generations_batched = ppo_trainer.generate(
            model_inputs, batch_size=2, generate_ref_response=True, **generation_kwargs
        )
        generations_batched = tokenizer.batch_decode(generations_batched)
        ref_generations_batched = tokenizer.batch_decode(ref_generations_batched)

        generations_single = []
        ref_generations_single = []
        for inputs in model_inputs:
            generation, ref_generation = ppo_trainer.generate(inputs, generate_ref_response=True, **generation_kwargs)
            generations_single.append(generation.squeeze())
            ref_generations_single.append(ref_generation.squeeze())

        generations_single = tokenizer.batch_decode(generations_single)
        ref_generations_single = tokenizer.batch_decode(ref_generations_single)

        assert generations_single == generations_batched
        assert ref_generations_single == ref_generations_batched

        assert generations_batched != ref_generations_batched
        assert generations_single != ref_generations_single

    def test_grad_accumulation(self):
        dummy_dataset = self._init_dummy_dataset()

        torch.manual_seed(0)
        gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_id, summary_dropout_prob=0.0)
        gpt2_model_clone = copy.deepcopy(gpt2_model)

        self.ppo_config.mini_batch_size = 2
        self.ppo_config.ppo_epochs = 1

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        dummy_dataloader = ppo_trainer.dataloader

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(1.0)]
            # train model by running a step twice
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        model_grad = gpt2_model.v_head.summary.weight

        self.ppo_config.mini_batch_size = 1
        self.ppo_config.gradient_accumulation_steps = 2

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=gpt2_model_clone,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        dummy_dataloader = ppo_trainer.dataloader

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(1.0)]
            # train model by running a step twice
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

        model_grad_acc = gpt2_model_clone.v_head.summary.weight
        assert torch.allclose(model_grad_acc, model_grad, rtol=0.001, atol=0.001)

    @unittest.skip("Fix by either patching `whomai()` to work in the staging endpoint or use a dummy prod user.")
    def test_push_to_hub_if_best_reward(self):
        REPO_NAME = "test-ppo-trainer"
        repo_id = f"{CI_HUB_USER}/{REPO_NAME}"

        dummy_dataset = self._init_dummy_dataset()

        push_to_hub_if_best_kwargs = {"repo_id": repo_id}

        ppo_config = PPOConfig(
            batch_size=2,
            mini_batch_size=1,
            log_with=None,
            push_to_hub_if_best_kwargs=push_to_hub_if_best_kwargs,
            compare_steps=1,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.gpt2_model,
            ref_model=self.gpt2_model_ref,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )

        ppo_trainer.optimizer.zero_grad = partial(ppo_trainer.optimizer.zero_grad, set_to_none=False)
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            _ = ppo_trainer.step(list(query_tensor), list(response_tensor), reward)
            break

    def test_batch_size_check(self):
        with pytest.raises(ValueError):
            PPOConfig(batch_size=2, mini_batch_size=2, gradient_accumulation_steps=2)
