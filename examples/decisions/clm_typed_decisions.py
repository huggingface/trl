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
# dependencies = [
#     "trl",
#     "transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/50649/head",
#     "accelerate @ git+https://github.com/huggingface/accelerate.git",
# ]
# ///

"""
Train CLM projection heads on typed decisions with a frozen SmolLM2 encoder.

python examples/decisions/clm_typed_decisions.py --config examples/decisions/clm_typed_decisions.yaml

torchrun --nproc_per_node=2 \
    examples/decisions/clm_typed_decisions.py --config examples/decisions/clm_typed_decisions.yaml
"""

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer, EarlyStoppingCallback, set_seed
from transformers.decision import parse_questions

from trl import ModelConfig, ScriptArguments, TrlParser
from trl.experimental.decision import DataCollatorForDecision, DecisionConfig, DecisionTrainer


@dataclass
class DataCollatorForClm(DataCollatorForDecision):
    """
    Pack CLM questions with a shared pool of distinct candidates for contrastive training.

    Accepts the shared decision request schema with one question per row. Nullable decisions are unsupported.
    Preprocess with `dataset.map(collator.tokenize)` to render and tokenize each question once. Batching packs the
    states and unique candidates into one prompt and builds soft targets over the candidate pool. `option_mask`
    restricts reported probabilities to each question's own candidates; the model's contrastive loss uses the full
    pool. Candidate identity is determined by its token IDs after truncation.


    Args:
        tokenizer ([`~transformers.PreTrainedTokenizerBase`]):
            Tokenizer used to render and encode states and candidates.
        decision_config ([`~transformers.DecisionConfig`]):
            Decision settings and template from the model.
        max_state_length (`int`, *optional*, defaults to `2047`):
            Maximum state and question token length.
        max_option_length (`int`, *optional*, defaults to `2047`):
            Maximum candidate token length.
        truncation (`str`, *optional*, defaults to `"error"`):
            Reject overflowing text (`"error"`) or truncate it (`"truncate"`) using the tokenizer's truncation side.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of tensor to return. Only `"pt"` is supported.
    """

    max_state_length: int = 2047
    max_option_length: int = 2047
    truncation: str = "error"
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.max_state_length < 1 or self.max_option_length < 1:
            raise ValueError("max_state_length and max_option_length must be positive")
        if self.truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be error or truncate")

    def tokenize(self, example: dict[str, Any]) -> dict[str, Any]:
        """Render and tokenize one decision with the native Transformers template."""
        inputs, question, targets = self.prepare_example(example)
        marks = inputs["marks"]
        if len(marks) != len(question.options) + 1:
            raise ValueError("The decision template must mark one state span and one span per option")
        segments = inputs["input_ids"][0].split(torch.diff(marks, prepend=marks.new_zeros(1)).tolist())
        sequences = []
        for index, segment in enumerate(segments):
            limit = self.max_state_length if index == 0 else self.max_option_length
            if self.truncation == "error" and len(segment) > limit:
                raise ValueError(
                    f"Input exceeds its {limit} token limit; increase the limit or set truncation='truncate'"
                )
            segment = segment[-limit:] if self.tokenizer.truncation_side == "left" else segment[:limit]
            sequences.append(segment.tolist())
        features = {
            "input_ids": sequences[0],
            "option_input_ids": sequences[1:],
            "labels": list(targets),
            "allow_null": asdict(question).get("allow_null", False),
            "is_choice": question.type == "choice",
        }
        if features["allow_null"]:
            raise ValueError("CLM does not support nullable decisions")
        features["labels"] = features["labels"][1:]
        return features

    def torch_call(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if "input_ids" not in features[0]:
            features = [self.tokenize(example) for example in features]
        pool, columns, sequences, counts = {}, [], [], []
        for example in features:
            sequences.append(example["input_ids"])
            row_columns, count = [], 0
            for ids in example["option_input_ids"]:
                key = tuple(ids)
                if key not in pool:
                    pool[key] = len(pool)
                    sequences.append(ids)
                    count += 1
                row_columns.append(pool[key])
            columns.append(row_columns)
            counts.append(count)
        labels = torch.zeros(len(features), len(pool))
        option_mask = torch.zeros_like(labels, dtype=torch.bool)
        for index, (example, row_columns) in enumerate(zip(features, columns, strict=True)):
            indices = torch.tensor(row_columns)
            labels[index].scatter_add_(0, indices, torch.tensor(example["labels"]))
            option_mask[index, indices] = True
        return {
            "input_ids": torch.tensor([[token for sequence in sequences for token in sequence]]),
            "marks": torch.tensor([[len(sequence) for sequence in sequences]]).cumsum(-1),
            "num_options": torch.tensor([counts]),
            "labels": labels,
            "option_mask": option_mask,
            "is_choice": torch.tensor([example["is_choice"] for example in features]),
        }


@dataclass
class ClmScriptArguments(ScriptArguments):
    dataset_name: str = field(default="LocalLLaMA/typed-decisions", metadata={"help": "Typed decision dataset."})
    dataset_config: str = field(default="all", metadata={"help": "Dataset workflow."})
    validation_fraction: float = field(
        default=0.1, metadata={"help": "Fraction of source rows held out for validation."}
    )
    max_length: int = field(default=2047, metadata={"help": "Maximum state or candidate token length."})
    head_width: int = field(default=1536, metadata={"help": "Projection-head hidden width."})
    projection_dim: int = field(default=512, metadata={"help": "Normalized embedding width."})
    head_depth: int = field(default=3, metadata={"help": "Projection-head depth."})
    freeze_encoder: bool = field(default=True, metadata={"help": "Train only the projection heads and temperature."})
    max_train_samples: int | None = field(default=None, metadata={"help": "Limit training source rows."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Limit validation source rows."})


@dataclass
class ClmModelArguments(ModelConfig):
    model_name_or_path: str = field(
        default="HuggingFaceTB/SmolLM2-135M", metadata={"help": "Text encoder checkpoint."}
    )


def prepare_dataset(dataset, decision_config):
    examples = []
    for row in dataset:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for question in parse_questions(questions, decision_config):
            answer = gold.get(question.name)
            if answer is None or "label" not in answer:
                continue
            label = answer["label"]
            label = str(label).lower() if isinstance(label, bool) else str(label)
            probabilities = answer.get("probabilities") or {}
            targets = [float(probabilities.get(option, 0.0)) for option in question.options]
            total = sum(targets)
            targets = (
                [value / total for value in targets]
                if total
                else [float(option == label) for option in question.options]
            )
            examples.append(
                {
                    "request": json.dumps({"state": state, "questions": {question.name: questions[question.name]}}),
                    "target_probs": [0.0, *targets],
                }
            )
    return Dataset.from_list(examples)


def compute_metrics(prediction):
    return {"accuracy": float(np.mean(prediction.predictions.argmax(-1) == prediction.label_ids.argmax(-1)))}


def main(script_args, training_args, model_args):
    from transformers import ClmConfig, ClmForDecision

    set_seed(training_args.seed)
    training_args.accelerator_config.dispatch_batches = False
    training_args.accelerator_config.split_batches = False
    model_kwargs = dict(training_args.model_init_kwargs or {})
    model_kwargs.setdefault("revision", model_args.model_revision)
    model_kwargs.setdefault("dtype", model_args.dtype)
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    encoder = AutoModel.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model = ClmForDecision(
        ClmConfig(
            text_config=encoder.config,
            hidden_size=script_args.head_width,
            projection_dim=script_args.projection_dim,
            num_hidden_layers=script_args.head_depth,
        ),
        model=encoder,
    )
    model.base_model.requires_grad_(not script_args.freeze_encoder)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    ids = sorted(dataset.unique("id"))
    random.Random(training_args.seed).shuffle(ids)
    validation_ids = set(ids[: max(1, round(script_args.validation_fraction * len(ids)))])
    train_dataset = dataset.filter(lambda row: row["id"] not in validation_ids)
    eval_dataset = dataset.filter(lambda row: row["id"] in validation_ids)
    if script_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(script_args.max_train_samples))
    if script_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(script_args.max_eval_samples))
    collator = DataCollatorForClm(
        tokenizer,
        model.decision_config,
        max_state_length=script_args.max_length,
        max_option_length=script_args.max_length,
        truncation="truncate",
    )
    with training_args.main_process_first(desc="Preparing typed decisions"):
        train_dataset = prepare_dataset(train_dataset, model.decision_config)
        eval_dataset = prepare_dataset(eval_dataset, model.decision_config)
        train_dataset = train_dataset.map(
            collator.tokenize, remove_columns=train_dataset.column_names, desc="Tokenizing train dataset"
        )
        eval_dataset = eval_dataset.map(
            collator.tokenize, remove_columns=eval_dataset.column_names, desc="Tokenizing eval dataset"
        )
    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    steps = training_args.max_steps
    if steps < 0:
        updates_per_epoch = math.ceil(len(trainer.get_train_dataloader()) / training_args.gradient_accumulation_steps)
        steps = math.ceil(training_args.num_train_epochs * updates_per_epoch)
    trainer.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.create_optimizer(),
        max_lr=training_args.learning_rate,
        total_steps=steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    if not training_args.resume_from_checkpoint:
        trainer.save_metrics("eval_before", trainer.evaluate())
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    model.decision_config.save_pretrained(training_args.output_dir)
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    trainer.end()


if __name__ == "__main__":
    parser = TrlParser((ClmScriptArguments, DecisionConfig, ClmModelArguments))
    main(*parser.parse_args_and_config())
