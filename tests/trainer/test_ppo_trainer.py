import fnmatch
import gc
import re
import tempfile
import unittest

import torch
from huggingface_hub import HfApi, HfFolder, delete_repo
from requests.exceptions import HTTPError
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import respond_to_batch

from ..testing_constants import CI_HUB_ENDPOINT, CI_HUB_USER, CI_HUB_USER_TOKEN


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


class PPOTrainerTester(unittest.TestCase):
    """
    A wrapper class for testing PPOTrainer
    """

    @classmethod
    def setUpClass(cls):
        cls._token = CI_HUB_USER_TOKEN
        cls._api = HfApi(endpoint=CI_HUB_ENDPOINT)
        cls._api.set_access_token(CI_HUB_USER_TOKEN)
        HfFolder.save_token(CI_HUB_USER_TOKEN)

    @classmethod
    def tearDownClass(cls):
        for model in [f"{CI_HUB_USER}/test-ppo-trainer"]:
            try:
                delete_repo(token=cls._token, repo_id=model)
            except HTTPError:
                pass

    def setUp(self):

        # model_id
        model_id = "gpt2"

        # get models and tokenizer
        self.gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        self.gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_id)

        # initialize trainer
        self.ppo_config = PPOConfig(batch_size=2, forward_batch_size=1, log_with=None)

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
        dummy_dataloader = ppo_trainer.dataloader

        self.assertTrue(isinstance(ppo_trainer.optimizer.optimizer, torch.optim.SGD))

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
        dummy_dataloader = ppo_trainer.dataloader

        self.assertTrue(isinstance(ppo_trainer.optimizer.optimizer, torch.optim.SGD))
        self.assertTrue(isinstance(ppo_trainer.lr_scheduler.scheduler, torch.optim.lr_scheduler.ExponentialLR))

        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor(1.0), torch.tensor(0.0)]
            # train model
            _ = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
            train_stats = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
            break

        for name, param in ppo_trainer.model.named_parameters():
            self.assertTrue(param.grad is not None, f"Parameter {name} has no gradient")

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            self.assertTrue(param.grad is None, f"Parameter {name} has a gradient")

        # Finally check stats
        for stat in EXPECTED_STATS:
            assert stat in train_stats.keys()

        # assert that the LR has increased for exponential decay
        self.assertTrue(train_stats["ppo/learning_rate"] > self.ppo_config.learning_rate)

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
                name = name.replace("pretrained_model.", "")

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
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor([[1.0]]), torch.tensor([[0.0]])]
            # train model - this should raise an error
            with self.assertRaises(ValueError):
                _ = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)

            reward = [torch.tensor([1.0]), torch.tensor([0.0])]
            # train model - this should work
            _ = ppo_trainer.step([q for q in query_tensor], [r for r in response_tensor], reward)
            break

        # check if the gradients are computed for the model
        for name, param in ppo_trainer.model.named_parameters():
            self.assertTrue(param.grad is not None, f"Parameter {name} has no gradient")

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            self.assertTrue(param.grad is None, f"Parameter {name} has a gradient")

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
        dummy_dataloader = ppo_trainer.dataloader
        # train model with ppo
        for query_tensor, response_tensor in dummy_dataloader:
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            reward = [torch.tensor([1.0]), torch.tensor([0.0])]
            # train model - this should raise an error
            bs = ppo_trainer.config.batch_size

            queries, responses, _ = ppo_trainer._step_safety_checker(
                bs, [q for q in query_tensor], [r for r in response_tensor], reward
            )

            self.assertTrue(isinstance(queries, list), f"queries should be a list, got {type(queries)}")
            self.assertTrue(isinstance(responses, list), f"responses should be a list, got {type(responses)}")

            # check the shapes
            for i in range(bs):
                self.assertEqual(queries[i].shape, torch.Size([7]))
                self.assertEqual(responses[i].size(), torch.Size([7]))
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
        # train model with ppo
        reward = [torch.tensor([1.0])]
        # train model - this should work fine
        train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

        # check gradients
        for name, param in ppo_trainer.model.named_parameters():
            self.assertTrue(param.grad is not None, f"Parameter {name} has no gradient")

        # ref model should not be trained
        for name, param in ppo_trainer.ref_model.named_parameters():
            self.assertTrue(param.grad is None, f"Parameter {name} has a gradient")

        # check train stats
        for stat in EXPECTED_STATS:
            self.assertTrue(stat in train_stats, f"Train stats should contain {stat}")

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
            self.assertTrue(re_search is not None)
            hub_repo_id = re_search.groups()[0]
            # Check we created a Hub repo
            self.assertEqual(hub_repo_id, repo_id)
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
