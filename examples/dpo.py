# 0. imports
import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM

from trl import DPOConfig, DPOTrainer


# 1. load a pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")
model_ref = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
dpo_config = {"batch_size": 1}
config = DPOConfig(**dpo_config)
dpo_trainer = DPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
query_txt = "This morning I went to the "
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 20,
}
response_tensor = dpo_trainer.generate(
    [item for item in query_tensor], return_prompt=False, **generation_kwargs
)
response_txt = tokenizer.decode(response_tensor[0])

# 5. train model with dpo which uses the reward implicitly defined by the mode and model_ref
train_stats = dpo_trainer.step([query_tensor[0]], [response_tensor[0]])
