# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams

from trl import HfPairwiseJudge, OpenAIPairwiseJudge


"""
Examples:

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/rloo_tldr --num_examples 1000
Model win rate: 31.40%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/rloo_tldr --judge_model gpt-3.5-turbo-0125 --num_examples 1000
Model win rate: 51.60%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/rloo_tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 51.20%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/ppo_tldr --num_examples 1000
Model win rate: 46.30%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/ppo_tldr --judge_model gpt-3.5-turbo-0125 --num_examples 1000
Model win rate: 52.50%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/ppo_tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 63.00%
"""


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "The model name or path to the model to evaluate."})
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    num_examples: Optional[int] = field(default=None, metadata={"help": "The number of examples to evaluate."})


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

# Load the dataset
dataset = load_dataset("trl-lib/tldr", split="validation")
if args.num_examples is not None:
    dataset = dataset.select(range(args.num_examples))

# Extract the prompts and reference completions
prompts = dataset["prompt"]
reference_completions = dataset["completion"]

# Generate the model completions
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=200)  # very generous max token length
llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
model_completions = [output.outputs[0].text.strip() for output in outputs]

# Judge the outputs
if "gpt" in args.judge_model:
    judge = OpenAIPairwiseJudge(args.judge_model)
else:
    judge = HfPairwiseJudge(args.judge_model)

completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]
best_idxs = judge.judge(prompts, completions)
model_win_rate = best_idxs.count(1) / len(best_idxs)
print(f"Model win rate: {model_win_rate*100:.2f}%")
