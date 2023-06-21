import re
import time

import numpy as np
import rich
import torch
from transformers import AutoTokenizer, load_tool

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment


console = rich.get_console()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set up models
model_id = "gpt2"
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)
model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tool = load_tool("ybelkada/simple-calculator")
tool = lambda text: str(round(float(tool(text)), 2))  # rounding to 2 decimal places


# system prompt
prompt = """\
What is 13.1 - 3?

<request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>

Result = 10.1 <submit>

What is 4 * 3?

<request><SimpleCalculatorTool>4 * 3<call>12<response>

Result = 12 <submit>

What is 12.1 + 1?

<request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>

Result = 13.1 <submit>"""

regex = r"Result = (\d+\.\d+)"
int_regex = r"Result = (\d+)"
pattern = r"Result = (\d+(?:\.\d+)?) <submit>"  # generated by chatGPT

# reward function (dummy)
reward_fn = lambda x: 1

# trainer
batch_size = 32
ppo_config = PPOConfig(
    batch_size=batch_size,
    learning_rate=1.41e-5,
    mini_batch_size=16,
    log_with="wandb",
)
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# environment
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": -1,
    "max_new_tokens": 32,
}
env = TextEnvironment(
    model,
    tokenizer,
    [tool],
    reward_fn,
    prompt,
    generation_kwargs=generation_kwargs,
)

# # 4. generate model response
# output_min_length = len(desired_tensor[0])  # what is the difference between `output_min_length` and `min_length`?
# output_max_length = (
#     len(desired_tensor[0]) + 1
# )  # + 1 because `output_max_length` is more like the `stop` in range(start, stop)
# output_length_sampler = LengthSampler(output_min_length, output_max_length)

# run environment episodes
for step in range(400):
    tasks = []
    answers = []
    for _ in range(batch_size):
        a = np.random.randint(0, 50)
        b = np.random.randint(0, 50)
        op = np.random.choice(["-", "+", "*"])
        tasks.append(f"\n\nWhat is {a} {op} {b}?")
        if op == "-":
            answers.append(a - b)
        elif op == "+":
            answers.append(a + b)
        else:
            answers.append(a * b)
    queries, responses, masks, rewards, histories = env.run(tasks)
    response_texts = [tokenizer.decode(response) for response in responses]
    query_texts = [tokenizer.decode(query) for query in queries]

    if step % 20 == 0:
        try:
            histories[0].show_text()
        except:
            pass

    # reward shaping
    predicted_numbers = []
    rewards = []
    for response_text, answer in zip(response_texts, answers):
        reward = 0
        predicted_number = None
        match_pattern = re.findall(pattern, response_text)
        if match_pattern:
            predicted_number = float(match_pattern[0])
        predicted_numbers.append(predicted_number)
        if (
            "Result = " in response_text
        ):  # sometimes the model generates giberish so we give positive rewards for correctly using tools
            reward += 1
        if "<submit>" in response_text:
            reward += 1
        if predicted_number is not None:
            if predicted_number == answer:
                reward += 10
        reward /= 12  # normalize
        rewards.append(torch.tensor(reward))

    train_stats = ppo_trainer.step(queries, responses, rewards, masks)
    ppo_trainer.log_stats(train_stats, {}, reward)
    correct_rate = np.average(np.array(predicted_numbers) == np.array(answers))
    train_stats["correct_rate"] = correct_rate
    print(f"correct rate {correct_rate}")
    print(f"step {step} reward {torch.stack(rewards).mean().item()}")
    # raise
ppo_trainer.save_pretrained(f"models/{time.time()}")
