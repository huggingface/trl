import torch
import time
from tqdm import tqdm
import numpy as np
tqdm.pandas()

from transformers import pipeline

from trl import AcceleratePPOTrainer

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

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config["forward_batch_size"]
}

ppo_trainer = AcceleratePPOTrainer(**config)
tokenizer = ppo_trainer.tokenizer

device = ppo_trainer.accelerator.device
if device.index is None:
    # single GPU - maybe introduce this hack inside AcceleratePPOTrainer?
    device = 0
sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=device)


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
    response_tensors = ppo_trainer.generate(query_tensors, **gen_kwargs)
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/generate'] = time.time()-t

    #### Compute sentiment score
    t = time.time()
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
    timing['time/get_sentiment_preds'] = time.time()-t

    #### Run PPO step 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, timing, batch, rewards, t0, logs)
    # Log the timing of the whole optimization step.
    timing['time/optimization'] = time.time()-t
    