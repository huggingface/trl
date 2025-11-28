# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)

# with open("trl/chat_templates/qwen2.5_sft_assistant_only.jinja", encoding="utf-8") as chat_template_file:
#     tokenizer.chat_template = chat_template_file.read()

# print(tokenizer.chat_template)  # Print the chat template


# messages = [{'content': '## Task B-1.3.\n\nA ship traveling along a river has covered $24 \\mathrm{~km}$ upstream and $28 \\mathrm{~km}$ downstream. For this journey, it took half an hour less than for traveling $30 \\mathrm{~km}$ upstream and $21 \\mathrm{~km}$ downstream, or half an hour more than for traveling $15 \\mathrm{~km}$ upstream and $42 \\mathrm{~km}$ downstream, assuming that both the ship and the river move uniformly.\n\nDetermine the speed of the ship in still water and the speed of the river.', 'role': 'user'}, {'content': "I don't know how to do.", 'role': 'assistant'}]

# processed = tokenizer.apply_chat_template(
#     messages,
#     return_dict=True,
#     tokenize=True,
#     return_assistant_tokens_mask=True,
# )


# print(tokenizer.decode(processed["input_ids"]))
# print(processed)

with open("results/eval/qwen2.5-1.5B-instruct/gsm8k.jsonl") as f:
    lines = f.readlines()

import sys
text = lines[int(sys.argv[1])]
import json
text = json.loads(text)

print(text["prompt"])
print(text["completion"])
print("##############")
print(text["answer"])
