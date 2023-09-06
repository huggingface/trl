import transformers

model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token

print(tokenizer("### Response:\n"))
print(tokenizer("some random text\n\n### Response:\n"))