# lm_ppo: Language modeling with PPO
> A Pytorch implementation of Proximal Policy Optimization for transfomer language models.


## What is it?
The library `lm_ppo` one can fine-tune transformer language models with Proximal Policy Optimization (PPO). The library is built with the `transformer` library by  🤗Huggingface. Therefore, one can load pre-trained language models directly via the transformer interface.

## How it works
Fine-tuning a language model via PPO consists of roughly three steps:

1. **Rollout**: The language model generates a response or continuation based on query which could be the start of a sentence.
2. **Evaluation**: The query and response are evaluated with a function, model, human feedback or some combination of them. The important thing is that this process should yield a scalar value for each query/response pair.
3. **Optimization**: This is the most complex part. In the optimisation step the query/response pairs are used to calculate the log-probabilities of the tokens in the sequences. This is done with the model that is trained and and a reference model, which is usually the pre-trained model before fine-tuning. The KL-divergence between the two outputs is used as an additional reward signal to make sure the generated responses don't deviate to far from the reference language model. The active language model is then trained with PPO.

This process is illustrated in the sketch below:

![Overview](nbs/images/lm_ppo_overview.png)

## Install

Install the library with pip:

`pip install lm_ppo`

If you want to run the example a few additional libraries are required. Clone the repository and install it with pip:

`pip install -r requirements.txt`

## How to use

### Basic example
This is a basic example on how to use the library. Based on a query the language model creates a response which is then evaluated. The evaluation could be a human in the loop or another model's output.

```
# imports
import torch
from transformers import GPT2Tokenizer
from lm_ppo.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from lm_ppo.ppo import PPOTrainer

# get models
gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(gpt2_model, query_tensor, pad_token_id=gpt2_tokenizer.eos_token_id)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# define a reward for response
reward = torch.tensor([1.0]) # this could be any reward such as a human or another model

# train model with ppo
train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
```




    2



### Advanced example: IMDB sentiment
For a detailed example check out the notebook `nbs/04-gpt2-sentiment-training.ipynb`, where GPT2 is fine-tuned to generate positive movie reviews. An few examples from the language models before and after optimisation are given below:

![Overview](nbs/images/table_imdb_preview.png)

## Reference

### Proximal Policy Optimisation
The PPO implementation largely follows the structure introduced in the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

### Language models
The language models utilize the `transformer` library by 🤗Huggingface.
