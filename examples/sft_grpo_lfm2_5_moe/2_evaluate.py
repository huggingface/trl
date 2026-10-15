# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# docstyle-ignore
"""
Exact-match accuracy of a model playing the HotpotQA library game, on held-out validation questions, with greedy
decoding and the same tool loop as training. Tool calls are parsed with TRL's response schema for the model's chat
template, so a malformed call counts against the model, as it does in training.

    python examples/sft_grpo_lfm2_5_moe/2_evaluate.py --model LiquidAI/LFM2.5-8B-A1B --output base.jsonl
"""

import argparse
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from wiki_library import WikiLibrary, load_questions

from trl.chat_template_utils import add_response_schema, parse_response


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B")
parser.add_argument("--num_questions", type=int, default=500)
parser.add_argument("--max_turns", type=int, default=10)
parser.add_argument("--output", default="eval.jsonl")
args = parser.parse_args()

tokenizer = add_response_schema(AutoTokenizer.from_pretrained(args.model))
questions = load_questions("validation").select(range(args.num_questions))
environments = [WikiLibrary() for _ in questions]
for environment, question in zip(environments, questions, strict=True):
    environment.reset(**question)
tools = [environments[0].search, environments[0].read, environments[0].submit_answer]
conversations = [list(question["prompt"]) for question in questions]

llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=0.85, enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

num_turns = [0] * len(questions)
emitted_calls = parsed_calls = 0
active = list(range(len(questions)))
for _ in range(args.max_turns):
    prompts = [
        tokenizer.apply_chat_template(
            conversations[i], tools=tools, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        for i in active
    ]
    outputs = llm.generate([{"prompt_token_ids": prompt} for prompt in prompts], sampling_params)
    still_active = []
    for i, prompt, output in zip(active, prompts, outputs, strict=True):
        completion_ids = list(output.outputs[0].token_ids)
        message = parse_response(tokenizer, completion_ids, prefix=prompt, tools=tools)
        conversations[i].append(message)
        num_turns[i] += 1
        # Every `<|tool_call_start|>` block the model emits must come back as at least one parsed call
        emitted_calls += tokenizer.decode(completion_ids).count("<|tool_call_start|>")
        parsed_calls += bool(message.get("tool_calls"))
        if not message.get("tool_calls"):
            continue
        for call in message["tool_calls"]:
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            try:
                result = getattr(environments[i], name)(**arguments)
            except Exception as error:
                result = {"error": str(error)}
            conversations[i].append({"role": "tool", "name": name, "content": str(result)})
        still_active.append(i)
    active = still_active

rewards = [environment.get_reward() for environment in environments]
with open(args.output, "w") as f:
    for question, conversation, environment, reward, turns in zip(
        questions, conversations, environments, rewards, num_turns, strict=True
    ):
        record = {
            "question": question["question"],
            "answer": question["answer"],
            "submitted": environment.submitted,
            "reward": reward,
            "turns": turns,
            "conversation": conversation,
        }
        f.write(json.dumps(record) + "\n")
print(
    json.dumps(
        {
            "model": args.model,
            "num_questions": len(questions),
            "exact_match": sum(rewards) / len(rewards),
            "submitted": sum(environment.submitted is not None for environment in environments) / len(environments),
            "mean_turns": sum(num_turns) / len(num_turns),
            "turns_with_tool_call_block": emitted_calls,
            "turns_with_parsed_tool_calls": parsed_calls,
        }
    )
)
