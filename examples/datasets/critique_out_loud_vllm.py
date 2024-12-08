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

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

from datasets import Dataset, DatasetDict, load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


"""
Critique Out Loud Dataset Generation

This script generates a dataset of self-critiques for a given dataset of "prompt" and preferred responses in "chosen" and rejected responses in "rejected" columns.
It uses the VLLM server to generate the self-critiques to train a reward model.

First start the VLLM server:

python -m vllm.entrypoints.openai.api_server \
    --model your_model_name \
    --host localhost \
    --port 8000

Then run the script:

python critique_out_loud_vllm.py \
    --model-name your_model_name \
    --base-dataset your_dataset \
    --upload-name your_output_dataset \
    --batch-size 32 \
    --num-workers 4
"""


COT_PROMPT = (
    "The following is a break down on the correctness and usefulness of the assistant's response to my question: "
)


def build_chat_messages(prompt, response):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def build_feedback_prompts(tokenizer, example):
    messages = build_chat_messages(example["prompt"], example["chosen"])
    chosen_prefix = tokenizer.apply_chat_template(messages, tokenize=False)

    messages = build_chat_messages(example["prompt"], example["rejected"])
    rejected_prefix = tokenizer.apply_chat_template(messages, tokenize=False)

    # Add COT prompt as a separate turn
    cot_messages = [{"role": "user", "content": COT_PROMPT}]
    cot_fmt = tokenizer.apply_chat_template(cot_messages, tokenize=False)

    example["chosen_feedback_prompt"] = chosen_prefix + cot_fmt
    example["rejected_feedback_prompt"] = rejected_prefix + cot_fmt
    return example


def main(args):
    # Initialize OpenAI client with VLLM endpoint
    client = OpenAI(
        api_key="EMPTY",  # VLLM doesn't need real key
        base_url=f"http://{args.host}:{args.port}/v1",
    )

    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.base_dataset)

    # Prepare prompts
    ds = ds.map(lambda x: build_feedback_prompts(tokenizer, x), num_proc=args.preprocessing_workers)

    def fetch_responses(examples):
        # Generate feedback for chosen responses
        chosen_prompts = [ex["chosen_feedback_prompt"] for ex in examples]
        chosen_responses = client.completions.create(
            model=args.model_name,
            prompt=chosen_prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=1,
        )
        chosen_feedback = [choice.text for choice in chosen_responses.choices]

        # Generate feedback for rejected responses
        rejected_prompts = [ex["rejected_feedback_prompt"] for ex in examples]
        rejected_responses = client.completions.create(
            model=args.model_name,
            prompt=rejected_prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=1,
        )
        rejected_feedback = [choice.text for choice in rejected_responses.choices]

        return [
            {**example, "chosen_feedback": [c_fb], "rejected_feedback": [r_fb]}
            for example, c_fb, r_fb in zip(examples, chosen_feedback, rejected_feedback)
        ]

    feedback_ds = {}
    splits = args.splits if args.splits else ds.keys()

    # Process each split
    for split in splits:
        results = []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for i in range(0, len(ds[split]), args.batch_size):
                batch = [ds[split][j] for j in range(i, min(i + args.batch_size, len(ds[split])))]
                futures.append(executor.submit(fetch_responses, batch))

            for future in tqdm(futures, desc=f"Processing {split}"):
                results.extend(future.result())

        feedback_ds[split] = results

    # Convert to HuggingFace dataset
    hf_ds = DatasetDict({split: Dataset.from_list(feedback_ds[split]) for split in splits})

    # Select relevant columns
    columns = ["prompt", "chosen", "rejected", "chosen_feedback", "rejected_feedback", "id"]
    hf_ds = hf_ds.select_columns(columns)

    # Push to hub if specified
    if args.upload_name:
        print(f"Uploading dataset to {args.upload_name}")
        for split in splits:
            hf_ds[split].push_to_hub(args.upload_name, split=split)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate self-critiques using VLLM")

    # Model and data parameters
    parser.add_argument("--model-name", type=str, required=True, help="Name or path of the model to use")
    parser.add_argument("--base-dataset", type=str, required=True, help="HuggingFace dataset to use as input")
    parser.add_argument("--splits", type=str, nargs="+", help="Dataset splits to process (default: all)")
    parser.add_argument("--upload-name", type=str, help="Name for uploading the dataset to HuggingFace Hub")

    # VLLM server settings
    parser.add_argument("--host", type=str, default="localhost", help="VLLM server host")
    parser.add_argument("--port", type=int, default=8000, help="VLLM server port")

    # Processing parameters
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for generation")
    parser.add_argument(
        "--preprocessing-workers", type=int, default=4, help="Number of workers for dataset preprocessing"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    args = parser.parse_args()
    main(args)
