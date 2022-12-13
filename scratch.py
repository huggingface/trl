import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
# from trl.accelerate_ppo import AcceleratePPOTrainer as PPOTrainer

torch.manual_seed(0)

# get models
gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# initialize trainer
ppo_config = {'batch_size': 1, 'forward_batch_size': 1}
ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, gpt2_tokenizer, **ppo_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
# response_tensor  = ppo_trainer.respond_to_batch(gpt2_model, query_tensor)
response_tensor  = respond_to_batch(gpt2_model, query_tensor)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0)]

# train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

assert train_stats['objective/logprobs'][0].mean().item() == -4.8132643699646

# from transformers import GPT2LMHeadModel

# model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="auto")

# print(train_stats)