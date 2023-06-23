# 0. imports
import torch
from transformers import AutoTokenizer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from trl.core import LengthSampler


# 1. load a pretrained model
model_id = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
batch_size = 256
config = PPOConfig(
    batch_size=batch_size,
    learning_rate=1.41e-5,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    log_with="wandb",
)
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer)

# 3. encode a query
query_txt = """Model Input: What is 304829 * 9920330
Expected Model Output: Calculator(304829 * 9920330)

Model Input: What is 877546 * 41323213
Expected Model Output: Calculator(877546 * 41323213)

Model Input: What is 1123 * 22
Expected Model Output: Calculator(1123 * 22)

Model Input: What is 4235435 * 8796879
Expected Model Output:"""
query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.pretrained_model.device)
query_tensor = query_tensor.expand(batch_size, -1)

desired_txt = " Calculator(4235435 * 8796879)"
desired_tensor = tokenizer.encode(desired_txt, return_tensors="pt").to(model.pretrained_model.device)

# 4. generate model response
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
    "max_new_tokens": len(desired_tensor[0]),
}
output_min_length = len(desired_tensor[0])  # what is the difference between `output_min_length` and `min_length`?
output_max_length = (
    len(desired_tensor[0]) + 1
)  # + 1 because `output_max_length` is more like the `stop` in range(start, stop)
output_length_sampler = LengthSampler(output_min_length, output_max_length)
for step in range(40):
    response_tensor = ppo_trainer.generate(
        [item for item in query_tensor], return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
    )
    response_txt = tokenizer.decode(response_tensor[0])
    print(response_txt)

    # 5. define a reward for response
    # test if each token is the same as the desired token
    reward = [(item == desired_tensor[0]).sum() / len(desired_tensor[0]) for item in response_tensor]
    train_stats = ppo_trainer.step(
        [item for item in query_tensor],
        response_tensor,
        reward,
    )
    ppo_trainer.log_stats(train_stats, {}, reward)
    print(f"step {step} reward {torch.stack(reward).mean().item()}")
