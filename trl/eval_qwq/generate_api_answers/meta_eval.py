import os
import json
import torch
import argparse
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import (
    is_master,
    dist_init,
    dist_print,
    dist_close,
    build_config,
    get_device,
)
from modeling import HFModel

def main():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="jet-ai/Jet-Nemotron-2B", help="Path to the model configuration or checkpoint.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=2048)


    args, opt = parser.parse_known_args()

    dist_init(gpu=None, cudnn_benchmark=False)

    dist_print("Additional Arguments: ", json.dumps(opt, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 trust_remote_code=True,
                                                 attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = model.eval()
    model = HFModel(model)


