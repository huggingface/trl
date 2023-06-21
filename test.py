# # from trl import TextEnvironment, TextHistory, AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
# # from transformers import AutoModelForCausalLM, AutoTokenizer, load_tool

# # tool = load_tool("ybelkada/simple-calculator")

# # model_id = "gpt2"

# # model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(0)
# # tokenizer = AutoTokenizer.from_pretrained(model_id)
# # tokenizer.pad_token = tokenizer.eos_token


# # # tokenizer.add_special_tokens({"additional_special_tokens": ["<request>", "<response>", "<call>", "<submit>", "<SimpleCalculatorTool>"]})

# # prompt = """\
# # What is 12.1 + 1 - 3?
# # <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>
# # <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>
# # Result = 10.1 <submit>

# # """

# # reward = lambda x: 1

# # env = TextEnvironment(model, tokenizer, [tool], reward, prompt, tool_tokens=["<SimpleCalculatorTool>"], generation_kwargs={"max_new_tokens": 32})

# # ppo_config = PPOConfig(
# #     use_text_environment=True,
# #     batch_size=2,
# #     mini_batch_size=1,
# # )
# # ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

# # tokens, masks, rewards, history = env.run(["What is 387 * 228?", "What is 23.1 + 1 - 3?"])
# # queries, responses = tokens
# # ppo_trainer.step(queries, responses, rewards, masks)

# # # encoded_prompt = env.encode_prompt(prompt)
# # import re
# # import numpy as np
# # import torch
# # from transformers import AutoTokenizer, load_tool

# # from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment

# # # set up models
# # model_id = "gpt2"
# # model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(0)
# # tokenizer = AutoTokenizer.from_pretrained(model_id)
# # tokenizer.pad_token = tokenizer.eos_token
# # tool = load_tool("ybelkada/simple-calculator")

# # # system prompt
# # prompt = """\
# # What is 12.1 + 1 - 3?
# # <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>
# # <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>
# # Result = 10.1 <submit>

# # What is 4 * 3?
# # <request><SimpleCalculatorTool>4 * 3<call>12<response>
# # Result = 12 <submit>

# # """

# # regex = r"Result = (\d+\.\d+)"
# # int_regex = r"Result = (\d+)"

# # # reward function (dummy)
# # reward_fn = lambda x: 1

# # # trainer
# # batch_size = 64
# # ppo_config = PPOConfig(
# #     use_text_environment=True,
# #     batch_size=batch_size,
# #     mini_batch_size=1,
# #     log_with="wandb",
# # )
# # ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

# # # environment
# # env = TextEnvironment(
# #     model,
# #     tokenizer,
# #     [tool],
# #     reward_fn,
# #     prompt,
# #     tool_tokens=["<SimpleCalculatorTool>"],
# #     generation_kwargs={"max_new_tokens": 32},
# # )

# # # run environment episodes
# # for step in range(40):
# #     tasks = []
# #     answers = []
# #     for _ in range(batch_size):
# #         a = np.random.randint(0, 50)
# #         b = np.random.randint(0, 50)
# #         op = np.random.choice(["-", "+", "*"])
# #         tasks.append(f"What is {a} {op} {b}?")
# #         if op == "-":
# #             answers.append(a - b)
# #         elif op == "+":
# #             answers.append(a + b)
# #         else:
# #             answers.append(a * b)

# #     tokens, masks, rewards, history = env.run(tasks)
# #     queries, responses = tokens
# #     response_texts = [tokenizer.decode(response) for response in responses]
# #     rewards = []
# #     for response_text, answer in zip(response_texts, answers):
# #         reward = 0
# #         if "Result = " in response_text: # sometimes the model generates giberish so we give positive rewards for correctly using tools
# #             reward += 1
# #             match = re.search(regex, response_text)
# #             int_match = re.search(int_regex, response_text)
# #             if match or int_match:
# #                 reward += 3
# #                 if int_match is not None:
# #                     if float(int_match.group(1)) == answer:
# #                         reward += 10
# #                 elif match is not None:
# #                     if float(match.group(1)) == answer:
# #                         reward += 10
# #         if "<submit>" in response_text:
# #             reward += 1
# #         reward /= 2  # normalize
# #         rewards.append(torch.tensor(reward))
# #     train_stats = ppo_trainer.step(queries, responses, rewards, masks)
# #     ppo_trainer.log_stats(train_stats, {}, reward)
# #     print(f"step {step} reward {torch.stack(rewards).mean().item()}")


# # h[0].show()
# # attention_masks = (tokenizer(h[0].text, return_tensors="pt")['input_ids'] < tokenizer.vocab_size) * 1.0
# # import torch
# # from torch.utils.data import TensorDataset, DataLoader
# # from accelerate import Accelerator
# # import copy

# # # seed
# # torch.manual_seed(0)
# # accumulation_steps = 2


# # accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)

# # # define toy inputs and labels
# # x = torch.tensor([1., 2., 3., 4.])
# # y = torch.tensor([2., 4., 6., 8.])

# # # define dataset and dataloader
# # dataset = TensorDataset(x, y)
# # dataloader = DataLoader(dataset, batch_size=2)

# # # define model, optimizer and loss function
# # model = torch.nn.Linear(1, 1).to(0)
# # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# # # clone the model
# # model_clone = copy.deepcopy(model)
# # optimizer_clone = torch.optim.SGD(model_clone.parameters(), lr=0.01)

# # criterion = torch.nn.MSELoss()

# # model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# # # loop over batches
# # for i, (inputs, labels) in enumerate(dataloader):
# #     with accelerator.accumulate(model):
# #         # reshape inputs and labels
# #         inputs = inputs.view(-1, 1)
# #         labels = labels.view(-1, 1)
# #         # forward pass
# #         outputs = model(inputs)
# #         loss = criterion(outputs, labels) 
# #         # backward pass
# #         accelerator.backward(loss)
# #         optimizer.step()
# #         # optimizer.zero_grad()
# #         # check if accumulation is done
# #         if (i + 1) % accumulation_steps == 0:
# #             print("w/ accumulation, the final model grad is", model.weight.grad)
# #             optimizer.zero_grad()
# #             break

# # loss = criterion(model_clone(x.view(-1, 1).to(0)), y.view(-1, 1).to(0))
# # loss.backward()
# # print("w/o accumulation, the final model grad is", model_clone.weight.grad)
# # from trl import TextEnvironment, AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
# # from transformers import AutoTokenizer, load_tool

# # tool = load_tool("ybelkada/simple-calculator")

# # model_id = "gpt2"

# # model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(0)
# # tokenizer = AutoTokenizer.from_pretrained(model_id)
# # tokenizer.pad_token = tokenizer.eos_token


# # prompt = """\
# # What is 12.1 + 1 - 3?
# # <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>
# # <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>
# # Result = 10.1 <submit>

# # """

# # reward = lambda x: 1

# # env = TextEnvironment(model, tokenizer, [tool], reward, prompt, tool_tokens=["<SimpleCalculatorTool>"], generation_kwargs={"max_new_tokens": 32})

# # ppo_config = PPOConfig(
# #     batch_size=2,
# #     mini_batch_size=1,
# # )
# # ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

# # tokens, _, rewards, history = env.run(["What is 387 * 228?", "What is 23.1 + 1 - 3?"])
# # queries, responses = tokens
# # ppo_trainer.step(queries, responses, rewards)
# import re
# import numpy as np
# import torch
# from transformers import AutoTokenizer, load_tool

# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment

# # set up models
# model_id = "gpt2"
# model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(0)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# tool = load_tool("ybelkada/simple-calculator")

# # system prompt
# prompt = """\
# What is 12.1 + 1 - 3?
# <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>
# <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>
# Result = 10.1 <submit>

# What is 4 * 3?
# <request><SimpleCalculatorTool>4 * 3<call>12<response>
# Result = 12 <submit>

# """

# regex = r"Result = (\d+\.\d+)"
# int_regex = r"Result = (\d+)"

# # reward function (dummy)
# reward_fn = lambda x: 1

# # trainer
# batch_size = 64
# ppo_config = PPOConfig(
#     batch_size=batch_size,
#     mini_batch_size=1,
#     log_with="wandb",
# )
# ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

# # environment
# env = TextEnvironment(
#     model,
#     tokenizer,
#     [tool],
#     reward_fn,
#     prompt,
#     tool_tokens=["<SimpleCalculatorTool>"],
#     generation_kwargs={"max_new_tokens": 32},
# )

# # run environment episodes
# for step in range(40):
#     tasks = []
#     answers = []
#     for _ in range(batch_size):
#         a = np.random.randint(0, 50)
#         b = np.random.randint(0, 50)
#         op = np.random.choice(["-", "+", "*"])
#         tasks.append(f"What is {a} {op} {b}?")
#         if op == "-":
#             answers.append(a - b)
#         elif op == "+":
#             answers.append(a + b)
#         else:
#             answers.append(a * b)

#     tokens, _, rewards, history = env.run(tasks)
#     queries, responses = tokens
#     response_texts = [tokenizer.decode(response) for response in responses]
#     rewards = []
#     for response_text, answer in zip(response_texts, answers):
#         reward = 0
#         if "Result = " in response_text: # sometimes the model generates giberish so we give positive rewards for correctly using tools
#             reward += 1
#             match = re.search(regex, response_text)
#             int_match = re.search(int_regex, response_text)
#             if match or int_match:
#                 reward += 3
#                 if int_match is not None:
#                     if float(int_match.group(1)) == answer:
#                         reward += 10
#                 elif match is not None:
#                     if float(match.group(1)) == answer:
#                         reward += 10
#         if "<submit>" in response_text:
#             reward += 1
#         reward /= 2  # normalize
#         rewards.append(torch.tensor(reward))
#     train_stats = ppo_trainer.step(queries, responses, rewards)
#     ppo_trainer.log_stats(train_stats, {}, reward)
# #     print(f"step {step} reward {torch.stack(rewards).mean().item()}")
# from datasets import load_dataset
# from trl import SFTTrainer
# from trl.trainer import DataCollatorForCompletionOnlyLM
# import transformers

# dataset = load_dataset("tatsu-lab/alpaca", split="train")

# model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
# tokenizer.pad_token = tokenizer.eos_token

# # print(tokenizer(" ### Response:"))
# # print(tokenizer("some random text\n ### Response:"))

# def formatting_prompts_func(examples):
#     output_text = []
#     for i in range(len(examples["instruction"])):
#         instruction = examples["instruction"][i]
#         input_text = examples["input"][i]
#         response = examples["output"][i]

#         if len(input_text) >= 2:
#             text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
#             ### Instruction:
#             {instruction}
            
#             ### Input:
#             {input_text}
            
#             ### Response:
#             {response}
#             '''
#         else:
#             text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
#             ### Instruction:
#             {instruction}
            
#             ### Response:
#             {response}
#             '''
#         output_text.append(text)

#     return output_text

# response_template = " ### Response:\n"
# data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

# trainer = SFTTrainer(
#     model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     formatting_func=formatting_prompts_func,
#     data_collator=data_collator,
#     max_seq_length=1024,
# )

# trainer.train()
# from trl import TextEnvironment, AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
# from transformers import AutoTokenizer, load_tool

# tool = load_tool("ybelkada/simple-calculator")

# model_id = "gpt2"

# model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(0)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token


# prompt = """\
# What is 12.1 + 1 - 3?
# <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>
# <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>
# Result = 10.1 <submit>

# """

# reward = lambda x: 1

# env = TextEnvironment(model, tokenizer, [tool], reward, prompt, tool_tokens=["<SimpleCalculatorTool>"], generation_kwargs={"max_new_tokens": 32})

# ppo_config = PPOConfig(
#     use_text_environment=True,
#     batch_size=2,
#     mini_batch_size=1,
# )
# ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

# tokens, masks, rewards, history = env.run(["What is 387 * 228?", "What is 23.1 + 1 - 3?"])
# queries, responses = tokens
# ppo_trainer.step(queries, responses, rewards, masks)

# # 0. imports
# import re
# import numpy as np
# import torch
# from transformers import AutoTokenizer, load_tool

# from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, TextEnvironment
# from trl.core import LengthSampler
# import rich
# console = rich.get_console()

# # set up models
# model_id = "EleutherAI/gpt-neo-125m"
# model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     model_id, torch_dtype=torch.bfloat16
# )
# model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# tool = load_tool("ybelkada/simple-calculator")


# # system prompt
# prompt = """\
# What is 13.1 - 3?

# <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>

# Result = 10.1 <submit>

# What is 4 * 3?

# <request><SimpleCalculatorTool>4 * 3<call>12<response>

# Result = 12 <submit>

# What is 12.1 + 1?

# <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>

# Result = 13.1 <submit>"""
# # tokenizer.decode(tokenizer.encode(prompt)[:45])
# # tokenizer.decode(tokenizer.encode(prompt)[43:45])
# # raise
# regex = r"Result = (\d+\.\d+)"
# int_regex = r"Result = (\d+)"

# # reward function (dummy)
# reward_fn = lambda x: 1

# # trainer
# batch_size = 32
# ppo_config = PPOConfig(
#     batch_size=batch_size,
#     learning_rate=1.41e-5,
#     mini_batch_size=16,
#     log_with="wandb",
# )
# ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

# # environment
# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.eos_token_id,
#     "eos_token_id": -1,
#     "max_new_tokens": 32,
# }
# env = TextEnvironment(
#     model,
#     tokenizer,
#     [tool],
#     reward_fn,
#     prompt,
#     generation_kwargs=generation_kwargs,
# )

# # # 4. generate model response
# # output_min_length = len(desired_tensor[0])  # what is the difference between `output_min_length` and `min_length`?
# # output_max_length = (
# #     len(desired_tensor[0]) + 1
# # )  # + 1 because `output_max_length` is more like the `stop` in range(start, stop)
# # output_length_sampler = LengthSampler(output_min_length, output_max_length)

# # run environment episodes
# for step in range(400):
#     tasks = []
#     answers = []
#     for _ in range(batch_size):
#         a = np.random.randint(0, 50)
#         b = np.random.randint(0, 50)
#         op = np.random.choice(["-", "+", "*"])
#         tasks.append(f"\n\nWhat is {a} {op} {b}?")
#         if op == "-":
#             answers.append(a - b)
#         elif op == "+":
#             answers.append(a + b)
#         else:
#             answers.append(a * b)
#     queries, responses, masks, rewards, history = env.run(tasks)
    
#     console.print(f"[bold red]{tokenizer.decode(queries[0])}[/]" + f"[bold green]{tokenizer.decode(responses[0])}[/]")
#     # print(tokenizer.decode(queries[1]))
#     # print(tokenizer.decode(responses[1]))
#     # g = model.generate(queries[1].unsqueeze(0), **generation_kwargs)[0, len(queries[1]):]
#     # print(tokenizer.decode(g))
#     response_texts = [tokenizer.decode(response) for response in responses]
#     rewards = []
#     correct_count = 0
#     for response_text, answer in zip(response_texts, answers):
#         reward = 0
#         if "Result = " in response_text: # sometimes the model generates giberish so we give positive rewards for correctly using tools
#             reward += 1
#             match = re.search(regex, response_text)
#             int_match = re.search(int_regex, response_text)
#             if match or int_match:
#                 reward += 3
#                 if int_match is not None:
#                     if float(int_match.group(1)) == answer:
#                         reward += 10
#                         correct_count += 1
#                 elif match is not None:
#                     if float(match.group(1)) == answer:
#                         reward += 10
#                         correct_count += 1
#         if "<submit>" in response_text:
#             reward += 1
#         reward /= 15  # normalize
#         rewards.append(torch.tensor(reward))

#     # debugging output
#     for item in response_texts[:1]:
#         print("--------")
#         print(item)
#         print("--------")

#     train_stats = ppo_trainer.step(queries, responses, rewards, masks)
#     ppo_trainer.log_stats(train_stats, {}, reward)
#     print(f"correct rate {correct_count / len(response_texts)}")
#     print(f"step {step} reward {torch.stack(rewards).mean().item()}")

# from trl import TextEnvironment, AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
# from transformers import AutoTokenizer, load_tool

# tool = load_tool("ybelkada/simple-calculator")

# model_id = "gpt2"

# model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id).to(0)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token


# prompt = """\
# What is 12.1 + 1 - 3?
# <request><SimpleCalculatorTool>12.1 + 1<call>13.1<response>
# <request><SimpleCalculatorTool>13.1 - 3<call>10.1<response>
# Result = 10.1 <submit>

# """

# reward = lambda x: 1

# env = TextEnvironment(model, tokenizer, [tool], reward, prompt, generation_kwargs={"max_new_tokens": 32})

# ppo_config = PPOConfig(
#     batch_size=2,
#     mini_batch_size=1,
# )
# ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer)

# queries, responses, masks, rewards, history = env.run(["What is 387 * 228?", "What is 23.1 + 1 - 3?"])
# ppo_trainer.step(queries, responses, rewards, masks)
# https://gist.github.com/younesbelkada/8bb36332cd2147c070b52ab25878c78f
from datasets import load_dataset
from trl import SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer
import transformers

dataset = load_dataset("tatsu-lab/alpaca", split="train")

model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 2:
            text = f'''\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}
            '''
        else:
            text = f'''\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
### Instruction:
{instruction}

### Response:
{response}
            '''
        output_text.append(text)

    return output_text

response_template = "### Response:\n"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
    max_seq_length=1024,
)

trainer.train()

# import numpy as np

# model_id = "trl-internal-testing/dummy-GPT2-correct-vocab"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token

# response_template = "### Response:\n"
# data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

# text = f'''\ 
# ### Instructions: 
# Hello all this should be masked

# ### Response:
# I have not been masked correctly.
# '''
# encoded_text = tokenizer(text, return_tensors="pt")
# encoded_text["input_ids"] = encoded_text["input_ids"].squeeze()

# examples = [encoded_text]

# batch = data_collator(examples)
# labels = batch["labels"]
# last_pad_idx = np.where(labels == -100)[1][-1]
# result_text = tokenizer.decode(batch["input_ids"][0, last_pad_idx+1:])