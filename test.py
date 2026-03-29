from datasets import load_dataset
import os
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
ds = load_dataset("Anthropic/hh-rlhf",split = "train",)
print(ds[1])
