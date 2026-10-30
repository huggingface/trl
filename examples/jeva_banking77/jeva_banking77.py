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
# dependencies = ["trl[peft]", "transformers>=5.22.1"]
# ///

"""
Train a Qwen decision model to select banking intents, including requests with no matching option.

python examples/jeva_banking77/jeva_banking77.py --mode head --max_steps 100 --report_to none

accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/jeva_banking77/jeva_banking77.py --mode lora --use_peft --max_steps 100 --report_to none
"""

import random
from dataclasses import dataclass, field

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from trl.experimental.jeva import JevaConfig, JevaModel, JevaTrainer


@dataclass
class JevaScriptArguments(ScriptArguments):
    dataset_name: str = field(default="PolyAI/banking77", metadata={"help": "Intent classification dataset."})
    num_options: int = field(default=8, metadata={"help": "Candidate intents per decision."})
    null_fraction: float = field(
        default=0.2, metadata={"help": "Fraction of decisions with the correct intent withheld."}
    )
    projection_dim: int = field(default=128, metadata={"help": "Width of the decision head."})
    max_train_samples: int | None = field(default=None, metadata={"help": "Limit training examples for a short run."})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Limit evaluation examples."})


@dataclass
class JevaModelArguments(ModelConfig):
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "Text encoder checkpoint."})
    lora_task_type: str = field(default="FEATURE_EXTRACTION", metadata={"help": "PEFT task type for the encoder."})


def prepare_dataset(dataset, num_options, null_fraction, seed):
    intents = dataset.features["label"].names

    def format_example(example, index):
        rng = random.Random(seed + index)
        label = example["label"]
        present = rng.random() >= null_fraction
        candidates = rng.sample([i for i in range(len(intents)) if i != label], num_options - int(present))
        if present:
            candidates.append(label)
        rng.shuffle(candidates)
        return {
            "state": example["text"],
            "question": "Which banking intent best matches this request?",
            "options": [intents[i].replace("_", " ") for i in candidates],
            "allow_null": True,
            "target_index": candidates.index(label) if present else None,
        }

    return dataset.map(format_example, with_indices=True, remove_columns=dataset.column_names)


def compute_metrics(prediction):
    return {"accuracy": float(np.mean(prediction.predictions.argmax(-1) == prediction.label_ids.argmax(-1)))}


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    model_kwargs = dict(training_args.model_init_kwargs or {})
    model_kwargs.setdefault("revision", model_args.model_revision)
    model_kwargs.setdefault("dtype", model_args.dtype)
    if model_args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = model_args.attn_implementation
    model = JevaModel.from_backbone_pretrained(
        model_args.model_name_or_path, model_config={"projection_dim": script_args.projection_dim}, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(script_args.max_train_samples))
    if script_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(script_args.max_eval_samples))
    train_dataset = prepare_dataset(
        train_dataset, script_args.num_options, script_args.null_fraction, training_args.seed
    )
    eval_dataset = prepare_dataset(
        eval_dataset, script_args.num_options, script_args.null_fraction, training_args.seed + 1
    )
    trainer = JevaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        compute_metrics=compute_metrics,
    )
    result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_metrics("train", result.metrics)
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((JevaScriptArguments, JevaConfig, JevaModelArguments))
    main(*parser.parse_args_and_config())
