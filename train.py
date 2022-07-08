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
    "cls_model_name": "ChaiML/rewardModel90kEpoch2K1M3",
    "cls_tokenizer_name": "roberta-large-mnli",
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
wandb.init(name="run-debug", project="gpt2-ppo", config=config)

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
    config["cls_model_name"], use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(config["cls_tokenizer_name"])


def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["text"])[-config["input_size"] :]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample


# ds = ds.filter(lambda x: np.random.uniform() < 0.1)
ds = ds.map(tokenize, batched=False).shuffle(seed=42)

gen_kwargs = {
    "min_length": -1,
    "temperature": 0.1,
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


def calculate_reward(query, response):
    # if response == "<|endoftext|>" or not response.strip():
    #     return torch.tensor(2.0).to(device)
    encoded_input = reward_tokenizer(query + response, return_tensors="pt").to(device)
    output = reward_model(**encoded_input)
    return output.logits[0, 1] - 4


def score_responses(input_texts):
    """Use reward model to score responses. """
    reward_tokens = reward_tokenizer(
        input_texts,
        return_tensors='pt',
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=512
    ).to(device)

    logits = reward_model(**reward_tokens).logits
    rewards = torch.Tensor([i for i in logits[:, 1]])
    preds = torch.Tensor([float(p[1]) for p in torch.softmax(logits, dim=1)])

    # add bias so has a roughly mean closer to 0 (suggested in paper and by library author)
    rewards = rewards - 4.0
    return rewards, preds


def evaluate(eval_batch):
    game_data = dict()
    game_data["query"] = eval_batch["query"]
    query_tensors = [torch.tensor(t).long().to(device) for t in eval_batch["tokens"]]

    #### get response from gpt2 and gpt2_ref
    response_tensors_ref, response_tensors = [], []
    for i in range(len(query_tensors)):
        query_len = len(query_tensors[i])

        output_ref = gpt2_model_ref.generate(
            query_tensors[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs
        ).squeeze()
        response_tensors_ref.append(clip_response(output_ref, query_len))

        output = gpt2_model.generate(
            query_tensors[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs
        ).squeeze()
        response_tensors.append(clip_response(output, query_len))

    #### decode responses
    game_data["original_model_response"] = [gpt2_tokenizer.decode(r) for r in response_tensors_ref]
    game_data["rl_model_response"] = [gpt2_tokenizer.decode(r) for r in response_tensors]

    # responses using original model
    texts = [q + r for q, r in zip(eval_batch['reward_input'], game_data['original_model_response'])]
    rewards, preds = score_responses(texts)

    game_data['original_model_rewards'] = rewards.cpu()
    game_data['original_model_preds'] = preds.cpu()

    # responses using new RL model

    texts = [q + r for q, r in zip(eval_batch['reward_input'], game_data['rl_model_response'])]
    rewards, preds = score_responses(texts)

    game_data['rl_model_rewards'] = rewards.cpu()
    game_data['rl_model_preds'] = preds.cpu()

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)

    logs = dict()
    logs.update({'evaluation/comparison_table': wandb.Table(dataframe=df_results)})

    # update rewards and preds how they change over time

    mean_reward_before = torch.mean(torch.tensor(game_data['original_model_rewards']))
    mean_preds_before = torch.mean(torch.tensor(game_data['original_model_preds']))

    mean_reward_after = torch.mean(torch.tensor(game_data['rl_model_rewards']))
    mean_preds_after = torch.mean(torch.tensor(game_data['rl_model_preds']))

    logs.update({
        'evaluation/original_model_mean_reward': mean_reward_before.cpu().numpy(),
        'evaluation/original_model_mean_preds': mean_preds_before.cpu().numpy(),
        'evaluation/rl_model_mean_reward': mean_reward_after.cpu().numpy(),
        'evaluation/rl_model_mean_preds': mean_preds_after.cpu().numpy(),
    })

    return logs


def clip_response(response, query_len):
    response = response[query_len:]
    stop_idx = (response == torch.tensor(198)).nonzero().flatten()
    if len(stop_idx) > 0:
        response = response[: stop_idx[0] + 1]
    return response


ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **config)

total_ppo_epochs = int(np.ceil(config["steps"] / config["batch_size"]))

dataloader_iter = iter(dataloader)
eval_batch = dataloader_iter.next()

for epoch, batch in tqdm(zip(range(total_ppo_epochs), dataloader_iter)):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(len(query_tensors)):
        query_len = len(query_tensors[i])
        response = gpt2_model.generate(
            query_tensors[i].unsqueeze(dim=0),
            max_length=query_len + config["output_size"],
            **gen_kwargs
        ).squeeze()
        response_tensors.append(clip_response(response, query_len))

    batch["response"] = [gpt2_tokenizer.decode(r) for r in response_tensors]
    timing["time/get_response"] = time.time() - t

    #### Compute reward score
    t = time.time()
    rewards = torch.tensor(
        [
            calculate_reward(q, r)
            for q, r in zip(batch["reward_input"], batch["response"])
        ]
    ).to(device)
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

    logs.update(evaluate(eval_batch))
    wandb.log(logs)
