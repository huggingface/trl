# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)


from trl import SFTTrainer, SFTConfig


"""
python examples/scripts/minimal/sft.py \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-05 \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --max_seq_length 1024 \
    --num_train_epochs 5 \
    --output_dir models/minimal/sft
"""


if __name__ == "__main__":
    parser = HfArgumentParser(SFTConfig)
    args = parser.parse_args_into_dataclasses()[0]
    base_model = "EleutherAI/pythia-160m"

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    left_tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")  # for generation
    left_tokenizer.pad_token = left_tokenizer.eos_token
    if tokenizer.chat_template is None:
        # a default chat template to simply concatenate the messages
        tokenizer.chat_template = (
            "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model)
    ref_model = AutoModelForCausalLM.from_pretrained(base_model)

    ################
    # Dataset
    ################
    raw_datasets = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False).strip()
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False).strip()
        return row

    raw_datasets = raw_datasets.map(process, load_from_cache_file=False)
    eval_samples = 20
    train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
    eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))

    ################
    # Training
    ################
    # treats the EOS token and the padding token distinctively
    default_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def data_collator(x):
        batch = default_collator(x)
        batch["input_ids"].masked_fill_(~batch["attention_mask"].bool(), 0)
        return batch

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="chosen",
        max_seq_length=1000,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()
    # trainer.generate_completions(True)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)
