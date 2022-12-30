# imports
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import respond_to_batch


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, query_data, response_data):
        self.query_data = query_data
        self.response_data = response_data

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        return self.query_data[idx], self.response_data[idx]


def test_gpt2_model():
    # get models
    gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # initialize trainer
    ppo_config = {"batch_size": 2, "forward_batch_size": 1, "log_with_wandb": False}
    ppo_config = PPOConfig(**ppo_config)

    # encode a query
    query_txt = "This morning I went to the "
    query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
    assert query_tensor.shape == (1, 7)
    # get model response
    response_tensor = respond_to_batch(gpt2_model, query_tensor)
    assert response_tensor.shape == (1, 20)

    # create a dummy dataset
    min_length = min(len(query_tensor[0]), len(response_tensor[0]))
    dummy_dataset = DummyDataset(
        [query_tensor[:, :min_length].squeeze(0) for _ in range(2)],
        [response_tensor[:, :min_length].squeeze(0) for _ in range(2)],
    )
    dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2, shuffle=True)

    ppo_trainer = PPOTrainer(
        config=ppo_config, model=gpt2_model, ref_model=gpt2_model_ref, tokenizer=gpt2_tokenizer, dataset=dummy_dataset
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

    for stat in EXPECTED_STATS:
        assert stat in train_stats.keys()
