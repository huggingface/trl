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

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "transformers>=5.15.0",
#     "flash-linear-attention",  # kernels for Qwen3.5's Gated DeltaNet layers
#     "torchvision",  # Qwen3.5 ships a vision processor, loaded and saved with the model
#     "pillow",
#     "trackio",
# ]
# ///

"""SFT a student on agent traces, with the loss on the assistant turns only.

Each row of the dataset is a whole agent conversation (`messages` + `tools`), as `build_dataset.py` writes it. With
`assistant_only_loss=True`, TRL swaps in its training version of the Qwen3.5 chat template, which marks the
assistant turns, so the student learns the teacher's tool calls and answers and not the tool outputs. A trace longer
than `--max-length` is dropped rather than truncated, since truncation would cut its final answer.

On one H200, about 500 traces of about 10k tokens train for 3 epochs in about 16 minutes:

```sh
hf jobs uv run --flavor h200 --timeout 2h --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/main/examples/sft_agent_traces/sft.py \
    --dataset <you>/smoldataenvs-agent-traces --push-to <you>/Qwen3.5-2B-SmolDataEnvs-SFT
```

`--max-rows` trains on the first rows only, for a smaller run.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import get_training_chat_template


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--dataset", required=True, help="Hub dataset written by build_dataset.py")
    p.add_argument("--dataset-config", default=None, help="config of --dataset, e.g. 500 for the published traces")
    p.add_argument("--push-to", required=True, help="Hub model id for the trained student")
    p.add_argument("--private", action="store_true", help="push the model privately (and skip the Trackio Space)")
    p.add_argument("--max-rows", type=int, default=None, help="train on the first N rows only")
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--grad-accum", type=int, default=8)
    args = p.parse_args()

    dataset = load_dataset(args.dataset, args.dataset_config, split="train").select_columns(["messages", "tools"])
    if args.max_rows:
        dataset = dataset.select(range(min(args.max_rows, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template = get_training_chat_template(tokenizer)

    def n_tokens(row):
        ids = tokenizer.apply_chat_template(
            row["messages"], tools=row["tools"], chat_template=template, tokenize=True, return_dict=False
        )
        return {"n_tokens": len(ids)}

    before = len(dataset)
    dataset = dataset.map(n_tokens).filter(lambda row: row["n_tokens"] <= args.max_length).remove_columns("n_tokens")
    print(f"{len(dataset)}/{before} traces fit in {args.max_length} tokens")

    run_name = args.push_to.split("/")[-1]
    config = SFTConfig(
        output_dir=run_name,
        assistant_only_loss=True,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # below 1, a fraction of the total steps
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        logging_steps=1,
        save_strategy="no",
        # Trackio publishes its dashboard as a public Space when the model is pushed, so a private run logs to the job
        # output only.
        report_to="none" if args.private else "trackio",
        run_name=run_name,
        push_to_hub=True,
        hub_model_id=args.push_to,
        hub_private_repo=args.private,
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset)
    trainer.train()
    trainer.save_model()  # with push_to_hub=True, this also uploads the model


if __name__ == "__main__":
    main()
