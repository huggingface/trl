import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from gpt_tldr_judge import LLMJudgeConfig, llm_judge
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams


"""
python -i examples/scripts/evals/generate_tldr.py \
    --model_name_or_path vwxyzjn/rloo_tldr \
    --output_path examples/scripts/minimal/evals/rloo_tldr.csv \
    --n 1000
python -i examples/scripts/evals/generate_tldr.py \
    --model_name_or_path vwxyzjn/ppo_tldr \
    --output_path examples/scripts/minimal/evals/ppo_tldr.csv \
    --n 1000
"""


@dataclass
class Args:
    output_path: str
    model_name_or_path: str
    model_revision: str = "main"
    judge_model: str = "gpt-3.5-turbo-0125"
    n: int = 1000


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    subprocess.run(command_list, stderr=sys.stderr, stdout=sys.stdout)


MAX_TOKENS = 200  # a very generous max token length
parser = HfArgumentParser(Args)
args = parser.parse_args_into_dataclasses()[0]
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    revision=args.model_revision,
)
raw_datasets = load_dataset("trl-internal-testing/tldr-preference-sft-trl-style")
prompts = raw_datasets["test"]["prompt"]
if args.n is not None:
    prompts = prompts[: args.n]
reference_summaries = [message[-1]["content"] for message in raw_datasets["test"]["messages"]]
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=MAX_TOKENS)
llm = LLM(
    model=args.model_name_or_path,
    revision=args.model_revision,
    tokenizer_revision=args.model_revision,
    tensor_parallel_size=1,
)
outputs = llm.generate(prompts, sampling_params)
table = defaultdict(list)

# Print the outputs.
for output, reference_response in zip(outputs, reference_summaries):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    table["prompt"].append(prompt)
    table["model_response"].append(generated_text.strip())  # need `strip()` because of the leading space
    table["model_response_len"].append(len(output.outputs[0].token_ids))
    table["reference_response"].append(reference_response)
    table["reference_response_len"].append(
        len(tokenizer(f" {reference_response}")["input_ids"])
    )  # prepend leading space

df = pd.DataFrame(table)
df.to_csv(args.output_path)

#####
# GPT as a judge
####
df["response0"] = df["model_response"]
df["response1"] = df["reference_response"]
judged_df = llm_judge(
    LLMJudgeConfig(
        n=args.n,
        model=args.judge_model,
    ),
    df,
)
judged_df.rename(columns={"response0": "model_response", "response1": "reference_response"}, inplace=True)
print(judged_df["preferred"].value_counts())
# print percentage
print(judged_df["preferred"].value_counts(normalize=True))

judged_df.to_csv(args.output_path.replace(".csv", "_judged.csv"))
