import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt, listify_batch


def reduce_randomness(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


reduce_randomness(42)

config = {
    "run_name": str(os.environ.get("RUN_NAME", "run-test")),
    "project_name": str(os.environ.get("PROJECT_NAME", "gpt2-ppo")),
    "auth_token": "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",
    "wandb_key": "f3c2ba6991e7af7c6225908adad8f098296d7433",
    "model_name": str(os.environ.get("MODEL_NAME", "gpt2")),
    "ref_model_name": str(os.environ.get("REF_MODEL_NAME", "gpt2")),
    "cls_model_name": str(
        os.environ.get("CLS_MODEL_NAME", "ChaiML/rewardModel90kEpoch2K1M3")
    ),
    "cls_tokenizer_name": str(
        os.environ.get("CLS_TOKENIZER_NAME", "roberta-large-mnli")
    ),
    "cls_shift": float(os.environ.get("CLS_SHIFT", -4.0)),
    "cls_penal_coef": float(os.environ.get("CLS_PENAL_COEF", 1.2)),
    "steps": int(os.environ.get("STEPS", 50000)),
    "epochs": int(os.environ.get("EPOCHS", 5)),
    "eval_interval": int(os.environ.get("EVAL_INTERVAL", 10)),
    "batch_size": int(os.environ.get("BATCH_SIZE", 64)),
    "forward_batch_size": int(os.environ.get("FORWARD_BATCH_SIZE", 16)),
    "ppo_epochs": int(os.environ.get("PPO_EPOCHS", 4)),
    "input_size": int(os.environ.get("INPUT_SIZE", 960)),
    "output_size": int(os.environ.get("OUTPUT_SIZE", 32)),
    "lr": float(os.environ.get("LR", 1e-5)),
    "adap_kl_ctrl": (os.environ.get("ADAP_KL_CTRL", "False") == "True"),
    "init_kl_coef": float(os.environ.get("INIT_KL_COEF", 0.05)),
    "target": int(os.environ.get("TARGET", 6)),
    "horizon": int(os.environ.get("HORIZON", 10000)),
    "gamma": float(os.environ.get("GAMMA", 1.0)),
    "lam": float(os.environ.get("LAM", 0.95)),
    "cliprange": float(os.environ.get("CLIPRANGE", 0.2)),
    "cliprange_value": float(os.environ.get("CLIPRANGE_VALUE", 0.2)),
    "vf_coef": float(os.environ.get("VF_COEF", 0.1)),
    "temperature": float(os.environ.get("TEMPERATURE", 1.0)),
    "top_k": int(os.environ.get("TOP_K", 0)),
    "top_p": float(os.environ.get("TOP_P", 1.0)),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key=config["wandb_key"])
wandb.init(name=config["run_name"], project=config["project_name"], config=config)

ds = load_dataset(
    "ChaiML/user_model_inputs",
    split="train",
    use_auth_token="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",
)

model = AutoModelForCausalLM.from_pretrained(config["model_name"])
model_ref = AutoModelForCausalLM.from_pretrained(config["ref_model_name"])

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

wandb.watch(model, log="all")

model.to(device)
model_ref.to(device)

gen_kwargs = {
    "min_length": -1,
    "temperature": config["temperature"],
    "top_k": config["top_k"],
    "top_p": config["top_p"],
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

value_model = GPT2HeadWithValueModel.from_pretrained(config["model_name"]).to(device)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    config["cls_model_name"], use_auth_token=config["auth_token"]
).to(device)

reward_tokenizer = AutoTokenizer.from_pretrained(
    config["cls_tokenizer_name"], truncation_side="left", padding_side="left"
)


def tokenize(sample):
    sample["tokens"] = tokenizer.encode(sample["text"])[-config["input_size"] :]
    sample["query"] = tokenizer.decode(sample["tokens"])
    return sample


ds = ds.filter(lambda x: np.random.uniform() < 0.01)
ds = ds.map(tokenize, batched=False).shuffle(seed=42)


def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


dataloader = torch.utils.data.DataLoader(
    ds, batch_size=config["batch_size"], collate_fn=collater
)


def calculate_reward(query, response, response_len, return_preds=False):
    encoded_input = reward_tokenizer(
        query + response, max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    logits = reward_model(**encoded_input).logits
    preds = torch.softmax(logits, dim=1)
    rewards = shifted_logits_with_penalty(inverse_sigmoid(preds), response_len)

    if return_preds:
        return rewards[0, 1], preds[0, 1]
    else:
        return rewards[0, 1]


def inverse_sigmoid(preds):
    return torch.log(preds) - torch.log(1 - preds)


def shifted_logits_with_penalty(logits, response_len):
    return (
        logits
        + config["cls_shift"]
        - config["cls_penal_coef"] * np.exp(1 - response_len)
    )


def evaluate(eval_batch):
    game_data = dict()
    game_data["query"] = eval_batch["query"]
    query_tensors = [torch.tensor(t).long().to(device) for t in eval_batch["tokens"]]

    model.eval()

    #### get response from gpt2 and gpt2_ref
    response_tensors_ref, response_tensors = [], []
    for i in range(len(query_tensors)):
        query_len = len(query_tensors[i])

        output_ref = model_ref.generate(
            query_tensors[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs
        ).squeeze()
        response_tensors_ref.append(clip_response(output_ref, query_len))

        output = model.generate(
            query_tensors[i].unsqueeze(dim=0).to(device),
            max_length=query_len + config["output_size"],
            **gen_kwargs
        ).squeeze()
        response_tensors.append(clip_response(output, query_len))

    #### decode responses
    game_data["original_model_response"] = [
        tokenizer.decode(r) for r in response_tensors_ref
    ]
    game_data["rl_model_response"] = [
        tokenizer.decode(r) for r in response_tensors
    ]

    # responses using original model
    rewards = torch.tensor(
        [
            calculate_reward(q, r, len(rt), return_preds=True)
            for q, r, rt in zip(
                eval_batch["reward_input"],
                game_data["original_model_response"],
                response_tensors_ref,
            )
        ]
    )
    game_data["original_model_rewards"] = rewards[:, 0]
    game_data["original_model_preds"] = rewards[:, 1]

    # responses using new RL model
    rewards = torch.tensor(
        [
            calculate_reward(q, r, len(rt), return_preds=True)
            for q, r, rt in zip(
                eval_batch["reward_input"],
                game_data["rl_model_response"],
                response_tensors,
            )
        ]
    )
    game_data["rl_model_rewards"] = rewards[:, 0]
    game_data["rl_model_preds"] = rewards[:, 1]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)

    logs = dict()
    logs.update({"evaluation/comparison_table": wandb.Table(dataframe=df_results)})

    # update rewards and preds how they change over time
    mean_reward_before = torch.mean(game_data["original_model_rewards"])
    mean_preds_before = torch.mean(game_data["original_model_preds"])

    mean_reward_after = torch.mean(game_data["rl_model_rewards"])
    mean_preds_after = torch.mean(game_data["rl_model_preds"])

    logs.update(
        {
            "evaluation/original_model_mean_reward": mean_reward_before.cpu().numpy(),
            "evaluation/original_model_mean_preds": mean_preds_before.cpu().numpy(),
            "evaluation/rl_model_mean_reward": mean_reward_after.cpu().numpy(),
            "evaluation/rl_model_mean_preds": mean_preds_after.cpu().numpy(),
        }
    )

    return logs


def clip_response(response, query_len):
    response = response[query_len:]
    stop_idx = (response == torch.tensor(198)).nonzero().flatten()
    if len(stop_idx) > 0:
        response = response[: stop_idx[0] + 1]
    return response


ppo_trainer = PPOTrainer(model, model_ref, value_model, tokenizer, **config)

total_ppo_steps = int(np.ceil(config["steps"] / config["batch_size"]))
total_epochs = config["epochs"]

dataloader_iter = iter(dataloader)
eval_batch = dataloader_iter.next()

for epoch in range(total_epochs):
    print(f"Epoch {epoch + 1}/{total_epochs}")

    for step, batch in tqdm(zip(range(total_ppo_steps), iter(dataloader))):
        logs, timing = dict(), dict()
        t0 = time.time()
        query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]

        model.train()

        #### Get response from gpt2
        t = time.time()
        response_tensors = []
        for i in range(len(query_tensors)):
            query_len = len(query_tensors[i])
            response = model.generate(
                query_tensors[i].unsqueeze(dim=0),
                max_length=query_len + config["output_size"],
                **gen_kwargs
            ).squeeze()
            response_tensors.append(clip_response(response, query_len))

        batch["response"] = [tokenizer.decode(r) for r in response_tensors]
        timing["time/get_response"] = time.time() - t

        #### Compute reward score
        t = time.time()
        rewards = torch.tensor(
            [
                calculate_reward(q, r, len(rt))
                for q, r, rt in zip(
                    batch["reward_input"], batch["response"], response_tensors
                )
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

        if not step % config["eval_interval"]:
            logs.update(evaluate(eval_batch))

        wandb.log(logs)
