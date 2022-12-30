# imports
import re
import unittest

import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import respond_to_batch


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
]


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, response_data):
        self.query_data = query_data
        self.response_data = response_data

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        return self.query_data[idx], self.response_data[idx]


class PPOTrainerTester(unittest.TestCase):
    """
    A wrapper class for testing PPOTrainer
    """

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

    def setUp(self):
        # model_id
        model_id = "gpt2"

        # get models and tokenizer
        self.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        self.gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_id)

        # initialize trainer
        ppo_config = {"batch_size": 2, "forward_batch_size": 1, "log_with_wandb": False}
        self.ppo_config = PPOConfig(**ppo_config)

        return super().setUp()

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
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
            break

        for param in ppo_trainer.model.parameters():
            assert param.grad is not None

        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

    def test_ppo_step_with_no_ref(self):
        # initialize dataset
        dummy_dataset = self._init_dummy_dataset()

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
        )
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
            break

        for name, param in ppo_trainer.model.named_parameters():
            self.assertTrue(param.grad is not None, f"Parameter {name} has no gradient")

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            self.assertTrue(param.grad is None, f"Parameter {name} has a gradient")

        # initialize a new gpt2 model:
        model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        for name, param in ppo_trainer.ref_model.named_parameters():
            if "v_head" not in name:
                self.assertTrue(
                    torch.allclose(param.cpu(), model.state_dict()[name].cpu()),
                    f"Parameter {name} has changed from the original model",
                )

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

        num_shared_layers = 6

        ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.gpt2_model,
            ref_model=None,
            tokenizer=self.gpt2_tokenizer,
            dataset=dummy_dataset,
            num_shared_layers=num_shared_layers,
        )
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            train_stats = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
            break

        pattern = r".*transformer\.h\.(\d+)\..*"
        final_layers = ["ln_f", "v_head", "lm_head"]

        for name, param in ppo_trainer.model.named_parameters():
            if re.match(pattern, name):
                layer_number = int(re.match(pattern, name).groups(0)[0])
                if layer_number < num_shared_layers:
                    self.assertTrue(param.grad is None, f"Parameter {name} has a gradient")
                else:
                    self.assertTrue(param.grad is not None, f"Parameter {name} has no gradient")
            elif any([layer in name for layer in final_layers]):
                self.assertTrue(param.grad is not None, f"Parameter {name} has no gradient")

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            self.assertTrue(param.grad is None, f"Parameter {name} has a gradient")

        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()
