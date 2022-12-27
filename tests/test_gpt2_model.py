# imports
import pytest
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead
from trl.gpt2 import respond_to_batch

from trl.ppo import PPOTrainer


def test_gpt2_model():
    # get models
    gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # initialize trainer
    ppo_config = {"batch_size": 1, "forward_batch_size": 1}
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **ppo_config)

    # encode a query
    query_txt = "This morning I went to the "
    query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")
    assert query_tensor.shape == (1, 7)
    # get model response
    response_tensor = respond_to_batch(gpt2_model, query_tensor)
    assert response_tensor.shape == (1, 20)
    response_txt = gpt2_tokenizer.decode(response_tensor[0, :])

    # define a reward for response
    # (this could be any reward such as human feedback or output from another model)
    reward = [torch.tensor(1.0)]

    # train model with ppo
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

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
