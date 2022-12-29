# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import torch
import time
from tqdm import tqdm
import numpy as np
tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.trainer import LengthSampler

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO 
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": config.forward_batch_size
}

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb"):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should 
    customize this function to train the model on its own dataset.
    
    Args:
        dataset_name (`str`): 
            The name of the dataset to be loaded.
    
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split='train')
    ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
    ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

    input_size = LengthSampler(config.txt_in_min_len, config.txt_in_max_len)

    def tokenize(sample):
        sample["tokens"] = tokenizer.encode(sample["review"])[:input_size()]
        sample["query"] = tokenizer.decode(sample["tokens"])
        return sample

    ds = ds.map(tokenize, batched=False)

    def collater(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    dataloader = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, collate_fn=collater)
    return dataloader

# We retrieve the dataloader by calling the `build_dataset` function.
dataloader = build_dataset(config)

# Now let's build the model, the reference model, and the tokenizer.
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer.pad_token = tokenizer.eos_token

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataloader)

# the PPOTrainer has a dataloader attribute, which we can use to get the dataloader - 
# this step is important in a distributed setting, as the dataloader needs to be
# converted to a distributed dataloader.
dataloader = ppo_trainer.dataloader

# We then build the sentiment analysis pipeline, passing the model name and the
# sentiment analysis pipeline arguments. Let's also make sure to set the device
# to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if device.index is None:
    # single GPU - maybe introduce this hack inside PPOTrainer?
    device = 0
sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=device)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around 
# the `generate` function of the trained model.
gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

total_ppo_epochs = int(np.ceil(config.steps/config.batch_size))

for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
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
    rewards = [torch.tensor(output[1]["score"]).to(device) for output in pipe_outputs]
    timing['time/get_sentiment_preds'] = time.time()-t

    #### Run PPO step 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, logs, timing, t0)
    # Log the timing of the whole optimization step.
    timing['time/optimization'] = time.time()-t
    