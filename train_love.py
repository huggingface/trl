import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification 

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch

config = {
    "model_name": "gpt2",
    "cls_model_name": "bhadresh-savani/bert-base-uncased-emotion",
    "cls_tokenizer_name": "bhadresh-savani/bert-base-uncased-emotion",
    "auth_token": "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",
    "wandb_key": "f3c2ba6991e7af7c6225908adad8f098296d7433",
    "steps": 20000,
    "batch_size": 64,
    "forward_batch_size": 16,
    "ppo_epochs": 4,
    "input_size": 960,
    "output_size": 32,
    "lr": 1e-5,
    "init_kl_coef": 0.2,
    "target": 6,
    "horizon": 10000,
    "gamma": 1,
    "lam": 0.95,
    "cliprange": 0.2,
    "cliprange_value": 0.2,
    "vf_coef": 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key=config["wandb_key"])
wandb.init(name="run-love", project="gpt2-ppo", config=config)

ds = load_dataset(
    "ChaiML/user_model_inputs",
    split="train",
    use_auth_token="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",
)

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config["model_name"])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config["model_name"])

gpt2_tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

wandb.watch(gpt2_model, log="all")

gpt2_model.to(device)
gpt2_model_ref.to(device)


reward_model = AutoModelForSequenceClassification.from_pretrained(
    config["cls_model_name"],
    use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(config["cls_tokenizer_name"])


def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["text"])[-config["input_size"] :]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample

# ds = ds.filter(lambda x: np.random.uniform() < 0.1)
ds = ds.map(tokenize, batched=False)

gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
}


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataloader = torch.utils.data.DataLoader(
    ds, batch_size=config["batch_size"], collate_fn=collater
)


def calculate_reward(response):
    encoded_input = reward_tokenizer(response, return_tensors='pt').to(device)
    if encoded_input["input_ids"].shape[-1] <= 1:
        return torch.tensor(-4.0).to(device)
    output = reward_model(**encoded_input)
    return output.logits[0, 2]


ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)

total_ppo_epochs = int(np.ceil(config["steps"] / config["batch_size"]))

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(len(query_tensors)):
        query_len = len(query_tensors[i])
        response = gpt2_model.generate(
            query_tensors[i].unsqueeze(dim=0), max_length=query_len + config["output_size"], **gen_kwargs
        ).squeeze()[query_len:]

        stop_idx = (response == torch.tensor(198)).nonzero().flatten()
        if len(stop_idx) > 0:
            response = response[:stop_idx[0] + 1]
        response_tensors.append(response)

    batch["response"] = [gpt2_tokenizer.decode(r) for r in response_tensors]
    timing["time/get_response"] = time.time() - t

    #### Compute reward score
    t = time.time()
    rewards = torch.tensor([
        calculate_reward(r) for r in batch["response"]
    ]).to(device)
    timing["time/get_reward_preds"] = time.time() - t

    #### Run PPO step
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing["time/optimization"] = time.time() - t

    #### Log everything
    timing["time/epoch"] = time.time() - t0
    table_rows = [
        list(r) for r in zip(batch["query"], batch["response"], rewards.cpu().tolist())
    ]
    logs.update(
        {
            "game_log": wandb.Table(
                columns=["query", "response", "reward"], rows=table_rows
            )
        }
    )
    logs.update(timing)
    logs.update(stats)
    logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy()
    logs["env/reward_std"] = torch.std(rewards).cpu().numpy()
    logs["env/reward_dist"] = rewards.cpu().numpy()

    for key in logs:
        if isinstance(logs[key], list):
            if isinstance(logs[key][0], torch.Tensor):
                logs[key] = [array.cpu().numpy() for array in logs[key]]
    wandb.log(logs)
