"""Load an exported checkpoint in vLLM (1 GPU) and generate one W1-style answer greedily.
Usage: python vllm_load_check.py <hf_dir>
"""

import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


d = sys.argv[1]
llm = LLM(model=d, max_model_len=4096, gpu_memory_utilization=0.5, seed=42)
print("vLLM architecture:", llm.llm_engine.model_config.architectures)
tok = AutoTokenizer.from_pretrained(d)
msgs = [
    {
        "role": "user",
        "content": "What is $7 \\times 8$?\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
    }
]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=256))[0].outputs[0].text
print("greedy answer tail:", repr(out[-200:]))
print("contains \\boxed{56}:", "\\boxed{56}" in out)
