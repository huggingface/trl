# 0. imports
import torch
from transformers import GPT2Tokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.trainer import PtxData, PtxDataArgs, PtxLossArgs


# 1. load a pretrained model
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"mini_batch_size": 1, "batch_size": 1}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(
    config,
    model,
    model_ref,
    tokenizer,
)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = ppo_trainer.generate(list(query_tensor), return_prompt=False, **generation_kwargs)
response_txt = tokenizer.decode(response_tensor[0])

# 5. define a reward for response
# (this could be any reward such as human feedback or output from another model)
reward = [torch.tensor(1.0, device=model.pretrained_model.device)]

# 6. train model with ppo
train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)

# 7. initialize ppo_ptx trainer
ppo_config = {"batch_size": 3}
config = PPOConfig(**ppo_config)
ptx_data_args = PtxDataArgs(max_length=5, truncation_mode="keep_end")
ptx_loss_args = PtxLossArgs(ptx_coef=0.1)
ppo_ptx_trainer = PPOTrainer(
    config, model, model_ref, tokenizer, ptx_data_args=ptx_data_args, ptx_loss_args=ptx_loss_args
)

# 8. Prepare data for ppo_ptx

# encode a query
query_txt = "This noon I plan to go to the "
query_tensor_sec = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# generate model response
response_tensor_sec = ppo_trainer.generate(
    [item for item in query_tensor_sec], return_prompt=False, **generation_kwargs
)
response_txt_sec = tokenizer.decode(response_tensor_sec[0])

# encode a query
query_txt = "This afternoon I do not have time to go to the "
query_tensor_thd = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)

# generate model response
response_tensor_thd = ppo_trainer.generate(
    [item for item in query_tensor_thd], return_prompt=False, **generation_kwargs
)
response_txt_thd = tokenizer.decode(response_tensor_thd[0])

# prepare pre-trained data
ptx_input_ids = []

pretrain_txt = "This morning I went to the zoo"
pretrain_txt_tensor = tokenizer.encode(pretrain_txt, return_tensors="pt").to(model.pretrained_model.device)
ptx_input_ids.append(pretrain_txt_tensor[0])

pretrain_txt = "This morning I went to the zoo near my home"
pretrain_txt_tensor = tokenizer.encode(pretrain_txt, return_tensors="pt").to(model.pretrained_model.device)
ptx_input_ids.append(pretrain_txt_tensor[0])

ptx_data = PtxData(input_ids=ptx_input_ids)

# 8. prepare rewards for ppo_ptx
rewards = reward + reward + reward

# 9. train ppo_ptx model
train_stats = ppo_ptx_trainer.step(
    [query_tensor[0], query_tensor_sec[0], query_tensor_thd[0]],
    [response_tensor[0], response_tensor_sec[0], response_tensor_thd[0]],
    rewards,
    ptx_data=ptx_data,
)
