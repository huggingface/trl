import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
tqdm.pandas()

from transformers import pipeline

from trl.trainer.accelerate_ppo import AcceleratePPOTrainer

config = {
    "model_name": "lvwerra/gpt2-imdb",
    # "model_name": "facebook/opt-350m",
    "cls_model_name": "lvwerra/distilbert-imdb",
    "steps": 20000,
    "batch_size": 256,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    "txt_in_min_len": 2,
    "txt_in_max_len": 8,
    "txt_out_min_len": 4,
    "txt_out_max_len": 16,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

wandb.init(name='run-42', project='gpt2-test', config=config)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}

sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)

ppo_trainer = AcceleratePPOTrainer(**config)
tokenizer = ppo_trainer.tokenizer

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

total_ppo_epochs = int(np.ceil(config["steps"]/config['batch_size']))

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(ppo_trainer.dataloader))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
    #### Get response from gpt2
    t = time.time()
    response_tensors = ppo_trainer.get_response(query_tensors, **gen_kwargs)
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time()-t

    #### Compute sentiment score
    t = time.time()
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
    timing['time/get_sentiment_preds'] = time.time()-t
    
    #### Run PPO step 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
    logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)